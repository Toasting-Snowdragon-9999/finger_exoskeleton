"""Motor stage: grasp signal, controller setup and the motor thread, with a fake motor."""

import time
from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

import applied_config
import run_emg
from source.config import ConfigError, MotorConfig, validate_motor_config
from source.control_stage import ControlStage
from source.controller import ControllerOutput, ControllerState
from source.motor_stage import MotorStage, connect_controller, grasp_signal, turn_shift

OPEN, CLOSED = 1000, 2000
FAST = MotorConfig(closed_pos=CLOSED, open_pos=OPEN, max_speed=1e6, rate_hz=200.0)


class FakeMotors:
    """The part of Motors (DynamixelHardwareInterface) the controllers use.
    Motor 1 is the tendon motor; any other IDs only record mode and torque."""

    POS_CONTROL = 3
    EXT_POS_CONTROL = 4
    MODES = {"pos": 3, "ext": 4, "vel": 1, "cur": 0}

    def __init__(
        self,
        port: str = "COM0",
        baudrate: int = 1,
        init_mode: str | None = "cur",
        fail_after: int | None = None,
        motor_ids: tuple[int, ...] = (1,),
        position: int = 1500,
    ) -> None:
        self.port, self.baudrate = port, baudrate
        self.motor_ids = list(motor_ids)
        self.control_mode = [1] * len(motor_ids)  # velocity, as found on the bench
        self.torques = [False] * len(motor_ids)
        self.position = position
        self.sent: list[int] = []
        self.closed = False
        self.fail_after = fail_after
        if init_mode is not None:  # like Motors: every motor to init_mode, torque on
            self.set_cont_mode(init_mode)
            self.enable_torque()

    @property
    def torque(self) -> bool:
        """Torque of the tendon motor."""
        return self.torques[self.motor_ids.index(1)]

    def _indices(self, motor_ids: list[int] | None) -> list[int]:
        ids = self.motor_ids if motor_ids is None else motor_ids
        return [self.motor_ids.index(motor_id) for motor_id in ids]

    def disable_torque(self, motor_ids: list[int] | None = None) -> None:
        """Torque off."""
        for i in self._indices(motor_ids):
            self.torques[i] = False

    def enable_torque(self, motor_ids: list[int] | None = None) -> None:
        """Torque on."""
        for i in self._indices(motor_ids):
            self.torques[i] = True

    def set_cont_mode(self, mode: str, motor_ids: list[int] | None = None) -> None:
        """Operating mode; like the real control table, only with torque off."""
        for i in self._indices(motor_ids):
            assert not self.torques[i], "mode change with torque on"
            self.control_mode[i] = self.MODES[mode]

    def get_position(self, motor_id: int | None = None) -> int:
        """Present position in ticks."""
        assert motor_id in (None, 1)
        return self.position

    # pylint: disable-next=invalid-name
    def sendMotorCommand(self, motor_id: int, command: int) -> None:  # noqa: N802
        """Goal position; the fake motor gets there at once."""
        assert motor_id == 1
        assert self.control_mode[self.motor_ids.index(1)] == self.EXT_POS_CONTROL
        if self.fail_after is not None and len(self.sent) >= self.fail_after:
            raise RuntimeError("[TxRxResult] There is no status packet!")
        self.sent.append(command)
        self.position = command

    def close(self) -> None:
        """Close the port."""
        self.closed = True


def control_with(state: ControllerState | None, grasp: float = 0.6,
                 received: float = 100.0) -> ControlStage:
    """A ControlStage stand-in with the given controller state and newest activation."""
    output = None if state is None else ControllerOutput(state, 0.0, 0.0, True, (), "")
    activation = SimpleNamespace(grasp_activation=grasp, receive_time=received)
    return cast(ControlStage, SimpleNamespace(last_output=output, activation=activation))


def wait_until(condition: Callable[[], bool], timeout_s: float = 2.0) -> bool:
    """Poll condition until it holds or the timeout passes."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def test_grasp_signal_follows_controller_state_and_freshness() -> None:
    """Safe states and IDLE open the finger, a fresh activation passes through in
    GRASP_ARMED, a stale one is no signal."""

    def clock() -> float:
        return 100.1

    assert grasp_signal(lambda: None, 0.3, clock)() is None
    assert grasp_signal(lambda: control_with(None), 0.3, clock)() == 0.0
    assert grasp_signal(lambda: control_with(ControllerState.DISARMED), 0.3, clock)() == 0.0
    assert grasp_signal(lambda: control_with(ControllerState.FAULT), 0.3, clock)() == 0.0
    assert grasp_signal(lambda: control_with(ControllerState.IDLE), 0.3, clock)() == 0.0
    assert grasp_signal(lambda: control_with(ControllerState.GRASP_ARMED), 0.3, clock)() == 0.6
    stale = control_with(ControllerState.GRASP_ARMED, received=99.0)
    assert grasp_signal(lambda: stale, 0.3, clock)() is None


def test_stiffened_holds_the_mean_grasp_of_the_hold_window() -> None:
    """In STIFFENED the signal is frozen at the mean grasp activation of the last
    hold_window_s in GRASP_ARMED; it follows again once tremor is gone, and a new
    grasp (after IDLE) does not average in the old one."""
    now = [0.0]
    current: list[ControlStage] = []

    def clock() -> float:
        return now[0]

    signal = grasp_signal(lambda: current[0], 0.3, clock, hold_window_s=1.0)

    def at(t: float, state: ControllerState, grasp: float) -> float | None:
        now[0] = t
        current[:] = [control_with(state, grasp, received=t)]
        return signal()

    assert at(0.0, ControllerState.GRASP_ARMED, 0.1) == 0.1  # outside the window later
    for k, grasp in enumerate((0.4, 0.6, 0.4, 0.6)):
        assert at(1.0 + 0.2 * k, ControllerState.GRASP_ARMED, grasp) == grasp
    held = at(2.0, ControllerState.STIFFENED, 0.9)
    assert held == pytest.approx(0.5)
    assert at(2.5, ControllerState.STIFFENED, 0.1) == pytest.approx(0.5)  # tremor swings ignored
    assert at(3.0, ControllerState.GRASP_ARMED, 0.7) == 0.7
    assert at(3.5, ControllerState.IDLE, 0.2) == 0.0
    assert at(3.6, ControllerState.GRASP_ARMED, 0.3) == 0.3
    assert at(3.7, ControllerState.STIFFENED, 0.9) == pytest.approx(0.3)  # only the new grasp


def test_connect_puts_the_motor_in_extended_position_mode() -> None:
    """The controller starts at the present position and the motor is in mode 4 with torque."""
    made: list[FakeMotors] = []

    def factory(**kwargs: Any) -> FakeMotors:
        made.append(FakeMotors(**kwargs))
        return made[-1]

    motors, controller = connect_controller(replace(FAST, open_pos=None, port="COM9"), factory)
    assert motors is made[0] and motors.port == "COM9"
    assert motors.control_mode == [FakeMotors.EXT_POS_CONTROL] and motors.torque
    assert cast(Any, controller).open_pos == 1500.0
    assert type(controller).__name__ == "ProportionalEMGController"

    _, threshold = connect_controller(replace(FAST, controller="threshold", toggle=True), factory)
    assert type(threshold).__name__ == "ThresholdEMGController"
    assert cast(Any, threshold).mode == "toggle"


def test_connect_leaves_other_motors_alone() -> None:
    """Only the tendon motor gets a new mode and torque; the other keeps both."""
    motors = FakeMotors(motor_ids=(1, 2), init_mode=None)
    connect_controller(replace(FAST, motor_id=1), lambda **kwargs: motors)
    assert motors.control_mode == [FakeMotors.EXT_POS_CONTROL, 1]
    assert motors.torques == [True, False]


def test_turn_shift_finds_the_turn_the_motor_is_in() -> None:
    """A range of less than one turn has one copy that holds the present position."""
    assert turn_shift(1236, 1236, -1161) == 0
    assert turn_shift(-1161, 1236, -1161) == 0
    assert turn_shift(2935, 1236, -1161) == 4096  # booted with the grasp closed
    assert turn_shift(1236 + 3 * 4096, 1236, -1161) == 3 * 4096
    assert turn_shift(-500 - 4096, 1236, -1161) == -4096


@pytest.mark.parametrize("present, shift", [(1236, 0), (-1161, 0), (2935, 4096), (900, 0)])
def test_range_across_the_wrap_follows_the_motor_turn_count(present: int, shift: int) -> None:
    """A range across the wrap (open 1236, closed -1161) is shifted to the motor's turn count,
    so closing always turns the motor down from open, never the other way round."""
    config = replace(FAST, open_pos=1236, closed_pos=-1161)
    motors = FakeMotors(position=present)
    _, controller = connect_controller(config, lambda **_: motors)
    assert cast(Any, controller).open_pos == 1236 + shift
    assert cast(Any, controller).closed_pos == -1161 + shift
    stage = MotorStage(controller, lambda: 1.0, lambda _: None)
    stage.start()
    assert wait_until(lambda: motors.position == -1161 + shift)
    stage.stop()
    assert motors.sent[-1] == 1236 + shift


def test_connect_refuses_to_start_far_outside_the_range_with_torque_off() -> None:
    """A motor far from open-closed (stale positions) is not enabled: no jump at start."""
    motors = FakeMotors(position=3500)  # range 1000-2000
    with pytest.raises(ValueError, match="outside the open-closed range"):
        connect_controller(FAST, lambda **_: motors)
    assert not motors.torque and motors.closed and not motors.sent
    just_outside = FakeMotors(position=OPEN - 40)  # 5 % of the range is 50 ticks
    connect_controller(FAST, lambda **_: just_outside)
    assert just_outside.torque


def test_connect_rejects_an_unknown_motor_and_closes_the_port() -> None:
    """A motor ID that was not found raises and leaves the port closed."""
    motors = FakeMotors()
    with pytest.raises(ValueError):
        connect_controller(replace(FAST, motor_id=7), lambda **_: motors)
    assert motors.closed


def test_stage_closes_on_activation_and_opens_on_stop() -> None:
    """Full activation drives the finger to closed; stopping returns it to open."""
    motors = FakeMotors()
    _, controller = connect_controller(FAST, lambda **_: motors)
    messages: list[str] = []
    stage = MotorStage(controller, lambda: 1.0, messages.append)
    stage.start()
    assert wait_until(lambda: motors.position == CLOSED)
    stage.stop()
    assert not stage.running
    assert motors.sent[-1] == OPEN
    assert stage.error is None and messages == ["Motor control started"]


def test_stage_stays_open_while_the_controller_is_disarmed() -> None:
    """Before calibration the signal is 0, so the finger never closes."""
    motors = FakeMotors()
    _, controller = connect_controller(FAST, lambda **_: motors)
    signal = grasp_signal(lambda: control_with(ControllerState.DISARMED, grasp=1.0), 0.3)
    stage = MotorStage(controller, signal, lambda _: None)
    stage.start()
    assert wait_until(lambda: len(motors.sent) > 10)
    stage.stop()
    assert max(motors.sent) <= OPEN + 1


def test_stage_reports_a_motor_error_and_ends() -> None:
    """A bus error ends the motor thread with a message; nothing is raised."""
    motors = FakeMotors(fail_after=5)
    _, controller = connect_controller(FAST, lambda **_: motors)
    messages: list[str] = []
    stage = MotorStage(controller, lambda: 1.0, messages.append)
    stage.start()
    assert wait_until(lambda: not stage.running)
    assert stage.error is not None and "no status packet" in stage.error
    assert messages[-1].startswith("Motor control stopped: RuntimeError")


def test_motor_settings_are_validated() -> None:
    """closed_pos is required; positions and thresholds must make sense."""
    validate_motor_config(FAST)
    validate_motor_config(replace(FAST, open_pos=1236, closed_pos=-1161))  # across the wrap
    for bad in (
        MotorConfig(),
        replace(FAST, closed_pos=OPEN + 4096),  # a full turn or more
        replace(FAST, open_pos=None, closed_pos=2_000_000),
        replace(FAST, open_pos=CLOSED),
        replace(FAST, controller="pid"),
        replace(FAST, on_threshold=0.2, off_threshold=0.3),
        replace(FAST, hold_window_s=0.0),
    ):
        with pytest.raises(ConfigError):
            validate_motor_config(bad)


def test_run_emg_refuses_motor_without_closed_position(monkeypatch: pytest.MonkeyPatch) -> None:
    """--motor without closed_pos is a configuration error (and no port is opened)."""
    monkeypatch.setattr(applied_config, "MOTOR", MotorConfig())
    assert run_emg.main(["--source", "synthetic", "--headless", "--duration", "0",
                         "--motor"]) == run_emg.EXIT_CONFIG
