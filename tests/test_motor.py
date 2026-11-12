"""The grasp signal and the motor connection in source/motor.py, with a fake motor."""

from types import SimpleNamespace

import pytest

import settings
from source.calibration import Activation
from source.motor import GraspSignal, connect_motor
from source.stiffness_controller import State
from tests.fakes import FakeMotors


def fake_pipeline(state: State, grasp: float = 0.6, received: float = 100.0) -> SimpleNamespace:
    """Something with the two attributes GraspSignal reads: state and activation."""
    return SimpleNamespace(state=state, activation=Activation(grasp, 0.0, 0.0, received))


def signal_for(pipeline: SimpleNamespace, now: float = 100.0) -> GraspSignal:
    """A GraspSignal whose clock reads `now`."""
    return GraspSignal(pipeline, clock=lambda: now)


def test_disarmed_and_idle_open_the_finger() -> None:
    """Not calibrated, or no grasp detected: the signal is 0."""
    assert signal_for(fake_pipeline(State.DISARMED))() == 0.0
    assert signal_for(fake_pipeline(State.IDLE))() == 0.0


def test_the_finger_follows_the_grasp() -> None:
    """In GRASP_ARMED the signal is the grasp activation."""
    assert signal_for(fake_pipeline(State.GRASP_ARMED, grasp=0.7))() == 0.7


def test_old_emg_gives_no_signal() -> None:
    """EMG older than STALE_S counts as no signal (None)."""
    pipeline = fake_pipeline(State.GRASP_ARMED, received=100.0)
    assert signal_for(pipeline, now=100.0 + settings.STALE_S + 0.01)() is None


def test_tremor_holds_the_mean_of_the_grasp_before_it() -> None:
    """In STIFFENED the signal is held at the mean grasp of the last second, not the new value."""
    pipeline = fake_pipeline(State.GRASP_ARMED)
    clock = {"now": 100.0}
    signal = GraspSignal(pipeline, clock=lambda: clock["now"])
    for grasp in (0.4, 0.6):
        pipeline.activation = Activation(grasp, 0.0, 0.0, clock["now"])
        signal()
        clock["now"] += 0.1
    pipeline.state = State.STIFFENED
    pipeline.activation = Activation(0.9, 0.0, 0.0, clock["now"])
    assert signal() == pytest.approx(0.5)


def test_idle_forgets_the_previous_grasp() -> None:
    """After IDLE, the hold only averages over the new grasp."""
    pipeline = fake_pipeline(State.GRASP_ARMED, grasp=0.2)
    signal = signal_for(pipeline)
    signal()
    pipeline.state = State.IDLE
    signal()
    pipeline.state, pipeline.activation = State.GRASP_ARMED, Activation(0.8, 0.0, 0.0, 100.0)
    signal()
    pipeline.state = State.STIFFENED
    assert signal() == 0.8


def test_connect_sets_only_the_tendon_motor_to_extended_mode() -> None:
    """Motor MOTOR_ID goes to mode 4 with torque on; motor 1 is not touched."""
    motors, controller = connect_motor("COM0", motors_class=FakeMotors)
    assert motors.init_mode is None
    assert motors.control_mode == [1, FakeMotors.EXT_POS_CONTROL]
    assert motors.torque == [False, True]
    assert (controller.open_pos, controller.closed_pos) == (settings.OPEN_POS, settings.CLOSED_POS)


def test_connect_shifts_the_range_to_the_motors_turn() -> None:
    """A motor counted one turn further gets open and closed one turn further too."""
    def one_turn_on(**kwargs) -> FakeMotors:
        return FakeMotors(position=4600 + 4096, **kwargs)

    _, controller = connect_motor("COM0", motors_class=one_turn_on)
    assert controller.open_pos == settings.OPEN_POS + 4096


def test_connect_refuses_a_motor_far_outside_the_range() -> None:
    """Far outside open-closed the start is refused, with torque left off."""
    made = []

    def far_away(**kwargs) -> FakeMotors:
        made.append(FakeMotors(position=settings.CLOSED_POS - 1000, **kwargs))
        return made[0]

    with pytest.raises(ValueError, match="outside the open-closed range"):
        connect_motor("COM0", motors_class=far_away)
    assert made[0].torque == [False, False] and made[0].closed
