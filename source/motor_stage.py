"""
The motor stage: drives the tendon motor from the live EMG pipeline.

  acquisition thread: ... -> ControlStage -> grasp activation (0 = rest, 1 = reference grip)
  motor thread:       grasp_signal() -> EMG position controller (source/controllers.py) -> Motors

The position controllers (ProportionalEMGController, ThresholdEMGController)
are used unchanged: their own run() loop polls grasp_signal() at their rate
(50 Hz by default) in a separate thread, so a slow serial bus can never stall
EMG acquisition. Only the motor thread talks to the motor.

grasp_signal() is what the controller sees:
  controller DISARMED or FAULT (not calibrated, recalibrating, faults) -> 0.0, so the finger opens
  no activation yet, or the newest one older than stale_s (source lost) -> None;
      the controller holds the last value for its signal_timeout_s, then opens
  IDLE (no grasp detected) -> 0.0, so rest activity (rest tremor) does not move the finger
  STIFFENED (tremor during a grasp) -> held: the mean grasp activation of the last
      hold_window_s before tremor was detected, so the finger keeps its position
      instead of following the tremor-modulated activation
  GRASP_ARMED -> the grasp activation of the newest block

Stopping the stage stops the controller, whose run() always moves the finger
back to the open position.

The tendon motor is driven in extended position mode (mode 4), so the
open-closed range may cross the 0/4095 wrap of one motor turn. Only the
tendon motor is touched: its torque and mode are set, any other motor on the
bus keeps its mode and torque. The motor counts turns from where it was at
power-up, so the configured positions are shifted by whole turns (4096 ticks)
to the turn the motor is in now; see turn_shift().
"""

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any, Protocol

from .config import MotorConfig
from .control_stage import ControlStage
from .controller import SAFE_STATES, ControllerState


class PositionController(Protocol):
    """What the stage needs from an EMGPositionController (source/controllers.py)."""

    def run(self, get_activation: Callable[[], float | None], duration: float | None = None,
            verbose: bool = True) -> None:
        """Blocking control loop; returns the finger to open when it ends."""

    def stop(self) -> None:
        """Make run() return."""


def grasp_signal(
    get_control: Callable[[], ControlStage | None],
    stale_s: float,
    clock: Callable[[], float] = time.monotonic,
    hold_window_s: float = 1.0,
) -> Callable[[], float | None]:
    """The activation source for the position controller; see the module docstring."""
    history: deque[tuple[float, float]] = deque()  # (clock time, grasp activation) while armed
    held: float | None = None

    def signal() -> float | None:
        nonlocal held
        control = get_control()
        if control is None:
            return None
        output = control.last_output
        if output is None or output.state in SAFE_STATES:
            history.clear()
            held = None
            return 0.0
        activation = control.activation
        now = clock()
        if activation is None or now - activation.receive_time > stale_s:
            return None
        grasp = float(activation.grasp_activation)
        if output.state == ControllerState.STIFFENED:
            if held is None:
                held = sum(value for _, value in history) / len(history) if history else grasp
            return held
        held = None
        if output.state == ControllerState.IDLE:
            history.clear()  # the hold averages over the current grasp only
            return 0.0
        history.append((now, grasp))
        while history and now - history[0][0] > hold_window_s:
            history.popleft()
        return grasp

    return signal


class MotorStage:
    """Runs a position controller in its own thread."""

    def __init__(
        self,
        controller: PositionController,
        signal: Callable[[], float | None],
        message: Callable[[str], None] = print,
    ) -> None:
        self.controller = controller
        self.signal = signal
        self._message = message
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    @property
    def running(self) -> bool:
        """True while the motor thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the motor thread."""
        if self.running:
            return
        self.error = None
        self._thread = threading.Thread(target=self._run, name="motor", daemon=True)
        self._thread.start()
        self._message("Motor control started")

    def _run(self) -> None:
        try:
            self.controller.run(self.signal, verbose=False)
        except Exception as error:  # pylint: disable=broad-exception-caught
            # Any bus or motor error ends motor control; the EMG side keeps running
            self.error = f"{type(error).__name__}: {error}"
            self._message(f"Motor control stopped: {self.error}")

    def stop(self, timeout_s: float = 5.0) -> None:
        """Stop the controller (the finger returns to open) and wait for the thread."""
        self.controller.stop()
        if self._thread is not None:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                self._message("Motor thread did not stop in time")


TICKS_PER_TURN = 4096
# How far outside open-closed the motor may be at start: the larger of the two
MIN_START_MARGIN_TICKS = 30
START_MARGIN_FRACTION = 0.05


def turn_shift(present: int, open_pos: int, closed_pos: int) -> int:
    """
    The whole number of turns (in ticks) to add to open_pos and closed_pos so
    the range lies in the turn the motor counts in now. The range is shorter
    than one turn, so the present position belongs to exactly one copy of it
    (the copy whose middle is nearest).
    """
    middle = (open_pos + closed_pos) / 2
    return round((present - middle) / TICKS_PER_TURN) * TICKS_PER_TURN


def connect_controller(
    config: MotorConfig, motors_factory: Callable[..., Any] | None = None
) -> tuple[Any, PositionController]:
    """
    Connect to the motor, put the tendon motor in extended position mode and
    build the configured controller. Returns (motors, controller); close
    motors after the stage has stopped. Raises ValueError for a bad motor ID
    and whatever Motors raises when the port or motor is not found.
    """
    # pylint: disable=import-outside-toplevel  # dynamixel_sdk only when a motor is used
    from .controllers import ProportionalEMGController, ThresholdEMGController

    if motors_factory is None:
        from .DynamixelHardwareInterface import Motors

        motors_factory = Motors
    # init_mode=None: Motors leaves the mode and torque of every motor alone
    motors = motors_factory(port=config.port, baudrate=config.baudrate, init_mode=None)
    try:
        motor_id = motors.motor_ids[0] if config.motor_id is None else config.motor_id
        if motor_id not in motors.motor_ids:
            raise ValueError(f"Motor ID {motor_id} was not detected (found {motors.motor_ids})")
        assert config.closed_pos is not None  # validate_motor_config
        # The Dynamixel control table requires torque to be disabled while
        # changing the operating mode.
        motors.disable_torque([motor_id])
        motors.set_cont_mode("ext", [motor_id])
        # Read after the mode change, in the turn count the motor uses from now on
        present = int(motors.get_position(motor_id))
        if config.open_pos is None:  # open = here; closed = its copy nearest to here
            open_pos = present
            shift = turn_shift(present, config.closed_pos, config.closed_pos)
            closed_pos = config.closed_pos + shift
        else:
            shift = turn_shift(present, config.open_pos, config.closed_pos)
            open_pos, closed_pos = config.open_pos + shift, config.closed_pos + shift
        # The controller's first command is the present position clipped to the range, sent
        # without a speed limit. Far outside the range that would be a jump at full speed
        # (stale open/closed positions, or the motor was moved), so refuse instead, while
        # torque is still off.
        low, high = sorted((open_pos, closed_pos))
        margin = max(MIN_START_MARGIN_TICKS, round(START_MARGIN_FRACTION * (high - low)))
        if not low - margin <= present <= high + margin:
            raise ValueError(
                f"Motor {motor_id} is at {present} ticks, outside the open-closed range "
                f"{low}-{high}. Put the finger in the open position, or measure open_pos and "
                f"closed_pos again. Torque was left off."
            )
        motors.enable_torque([motor_id])
        common = {
            "open_pos": open_pos,
            "closed_pos": closed_pos,
            "max_speed": config.max_speed,
            "rate_hz": config.rate_hz,
            "signal_timeout_s": config.signal_timeout_s,
        }
        controller: PositionController
        if config.controller == "proportional":
            controller = ProportionalEMGController(  # type: ignore[no-untyped-call]
                motors, motor_id, deadband=config.deadband, gamma=config.gamma, **common
            )
        else:
            controller = ThresholdEMGController(  # type: ignore[no-untyped-call]
                motors,
                motor_id,
                on_threshold=config.on_threshold,
                off_threshold=config.off_threshold,
                mode="toggle" if config.toggle else "hold",
                **common,
            )
    except BaseException:
        motors.close()
        raise
    return motors, controller
