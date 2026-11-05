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
  otherwise -> the grasp activation of the newest block

Stopping the stage stops the controller, whose run() always moves the finger
back to the open position. The motor is driven in position mode (mode 3).
"""

import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

from .config import MotorConfig
from .control_stage import ControlStage
from .controller import SAFE_STATES


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
) -> Callable[[], float | None]:
    """The activation source for the position controller; see the module docstring."""

    def signal() -> float | None:
        control = get_control()
        if control is None:
            return None
        output = control.last_output
        if output is None or output.state in SAFE_STATES:
            return 0.0
        activation = control.activation
        if activation is None or clock() - activation.receive_time > stale_s:
            return None
        return float(activation.grasp_activation)

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


def connect_controller(
    config: MotorConfig, motors_factory: Callable[..., Any] | None = None
) -> tuple[Any, PositionController]:
    """
    Connect to the motor, put it in position mode and build the configured
    controller, like main_controller.py. Returns (motors, controller); close
    motors after the stage has stopped. Raises ValueError for a bad motor ID
    and whatever Motors raises when the port or motor is not found.
    """
    # pylint: disable=import-outside-toplevel  # dynamixel_sdk only when a motor is used
    from .controllers import ProportionalEMGController, ThresholdEMGController

    if motors_factory is None:
        from .DynamixelHardwareInterface import Motors

        motors_factory = Motors
    motors = motors_factory(port=config.port, baudrate=config.baudrate)
    try:
        motor_id = motors.motor_ids[0] if config.motor_id is None else config.motor_id
        if motor_id not in motors.motor_ids:
            raise ValueError(f"Motor ID {motor_id} was not detected (found {motors.motor_ids})")
        # The Dynamixel control table requires torque to be disabled while
        # changing the operating mode.
        motors.disable_torque()
        motors.set_cont_mode("pos")
        motors.enable_torque()
        open_pos = config.open_pos if config.open_pos is not None else motors.get_position(motor_id)
        common = {
            "open_pos": open_pos,
            "closed_pos": config.closed_pos,
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
