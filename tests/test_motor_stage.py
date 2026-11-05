"""Motor stage: grasp signal, controller setup and the motor thread, with a fake motor."""

import time
from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

import run_emg
from source.config import ConfigError, MotorConfig, validate_motor_config
from source.control_stage import ControlStage
from source.controller import ControllerOutput, ControllerState
from source.motor_stage import MotorStage, connect_controller, grasp_signal

OPEN, CLOSED = 1000, 2000
FAST = MotorConfig(closed_pos=CLOSED, open_pos=OPEN, max_speed=1e6, rate_hz=200.0)


class FakeMotors:
    """The part of Motors (DynamixelHardwareInterface) the controllers use."""

    POS_CONTROL = 3

    def __init__(
        self, port: str = "COM0", baudrate: int = 1, fail_after: int | None = None
    ) -> None:
        self.port, self.baudrate = port, baudrate
        self.motor_ids = [1]
        self.control_mode = [0]
        self.position = 1500
        self.sent: list[int] = []
        self.torque = False
        self.closed = False
        self.fail_after = fail_after

    def disable_torque(self) -> None:
        """Torque off."""
        self.torque = False

    def enable_torque(self) -> None:
        """Torque on."""
        self.torque = True

    def set_cont_mode(self, mode: str) -> None:
        """Only position mode matters here."""
        self.control_mode = [self.POS_CONTROL if mode == "pos" else 0]

    def get_position(self, motor_id: int | None = None) -> int:
        """Present position in ticks."""
        assert motor_id in (None, 1)
        return self.position

    # pylint: disable-next=invalid-name
    def sendMotorCommand(self, motor_id: int, command: int) -> None:  # noqa: N802
        """Goal position; the fake motor gets there at once."""
        assert motor_id == 1
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
    """Safe states open the finger, a fresh activation passes through, a stale one is no signal."""

    def clock() -> float:
        return 100.1

    assert grasp_signal(lambda: None, 0.3, clock)() is None
    assert grasp_signal(lambda: control_with(None), 0.3, clock)() == 0.0
    assert grasp_signal(lambda: control_with(ControllerState.DISARMED), 0.3, clock)() == 0.0
    assert grasp_signal(lambda: control_with(ControllerState.FAULT), 0.3, clock)() == 0.0
    assert grasp_signal(lambda: control_with(ControllerState.IDLE), 0.3, clock)() == 0.6
    stale = control_with(ControllerState.GRASP_ARMED, received=99.0)
    assert grasp_signal(lambda: stale, 0.3, clock)() is None


def test_connect_puts_the_motor_in_position_mode() -> None:
    """The controller starts at the present position and the motor is in mode 3 with torque."""
    made: list[FakeMotors] = []

    def factory(**kwargs: Any) -> FakeMotors:
        made.append(FakeMotors(**kwargs))
        return made[-1]

    motors, controller = connect_controller(replace(FAST, open_pos=None, port="COM9"), factory)
    assert motors is made[0] and motors.port == "COM9"
    assert motors.control_mode == [FakeMotors.POS_CONTROL] and motors.torque
    assert cast(Any, controller).open_pos == 1500.0
    assert type(controller).__name__ == "ProportionalEMGController"

    _, threshold = connect_controller(replace(FAST, controller="threshold", toggle=True), factory)
    assert type(threshold).__name__ == "ThresholdEMGController"
    assert cast(Any, threshold).mode == "toggle"


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
    for bad in (
        MotorConfig(),
        replace(FAST, closed_pos=5000),
        replace(FAST, open_pos=CLOSED),
        replace(FAST, controller="pid"),
        replace(FAST, on_threshold=0.2, off_threshold=0.3),
    ):
        with pytest.raises(ConfigError):
            validate_motor_config(bad)


def test_run_emg_refuses_motor_without_closed_position() -> None:
    """--motor with the default settings (no closed_pos) is a configuration error."""
    assert run_emg.main(["--source", "synthetic", "--headless", "--duration", "0",
                         "--motor"]) == run_emg.EXIT_CONFIG
