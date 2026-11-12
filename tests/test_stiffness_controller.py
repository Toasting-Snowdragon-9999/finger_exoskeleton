"""The state machine in source/stiffness_controller.py."""

import numpy as np

from source.calibration import Activation
from source.stiffness_controller import State, StiffnessController

GRIP = Activation(grasp=0.5, tremor_depth=0.05, tremor_ratio=0.5, receive_time=0.0)
REST = Activation(grasp=0.0, tremor_depth=0.05, tremor_ratio=0.5, receive_time=0.0)
TREMOR = Activation(grasp=0.5, tremor_depth=0.38, tremor_ratio=0.95, receive_time=0.0)
UNRHYTHMIC = Activation(grasp=0.5, tremor_depth=0.38, tremor_ratio=0.5, receive_time=0.0)


def run(controller: StiffnessController, activation: Activation, start: float,
        seconds: float) -> float:
    """Feed the same activation every 10 ms for `seconds`; returns the end time."""
    for now in np.arange(start, start + seconds, 0.01):
        controller.update(activation, float(now))
    return start + seconds


def armed_controller() -> StiffnessController:
    """A controller that has just been armed (IDLE)."""
    controller = StiffnessController()
    controller.arm()
    return controller


def test_a_disarmed_controller_does_nothing() -> None:
    """Before arming, a grip with tremor changes nothing."""
    controller = StiffnessController()
    run(controller, TREMOR, 0.0, 2.0)
    assert controller.state == State.DISARMED and controller.stiffness == 0.0


def test_a_grip_arms_the_grasp_after_the_on_time() -> None:
    """IDLE -> GRASP_ARMED only once the grip has lasted GRASP_ON_TIME_S (0.1 s)."""
    controller = armed_controller()
    run(controller, GRIP, 0.0, 0.05)
    assert controller.state == State.IDLE
    run(controller, GRIP, 0.05, 0.1)
    assert controller.state == State.GRASP_ARMED


def test_releasing_the_grip_goes_back_to_idle() -> None:
    """GRASP_ARMED -> IDLE once the grasp has stayed low for GRASP_OFF_TIME_S (0.3 s)."""
    controller = armed_controller()
    now = run(controller, GRIP, 0.0, 0.2)
    now = run(controller, REST, now, 0.2)
    assert controller.state == State.GRASP_ARMED
    run(controller, REST, now, 0.2)
    assert controller.state == State.IDLE


def test_only_rhythmic_tremor_stiffens() -> None:
    """A high depth without a rhythmic ratio does not count as tremor."""
    controller = armed_controller()
    now = run(controller, GRIP, 0.0, 0.2)
    now = run(controller, UNRHYTHMIC, now, 1.0)
    assert controller.state == State.GRASP_ARMED
    run(controller, TREMOR, now, 0.4)
    assert controller.state == State.STIFFENED


def test_stiffness_follows_the_tremor_depth_and_ramps() -> None:
    """Target = 0.3 + 2 x (depth - 0.08), reached at no more than 20 per second."""
    controller = armed_controller()
    now = run(controller, GRIP, 0.0, 0.2)
    now = run(controller, TREMOR, now, 0.31)
    assert controller.state == State.STIFFENED
    assert controller.stiffness < 0.9  # still ramping up
    run(controller, TREMOR, now, 0.2)
    assert np.isclose(controller.stiffness, 0.3 + 2 * (0.38 - 0.08))


def test_when_tremor_stops_the_stiffness_ramps_down() -> None:
    """STIFFENED -> GRASP_ARMED after TREMOR_OFF_TIME_S, and the stiffness falls slowly."""
    controller = armed_controller()
    now = run(controller, GRIP, 0.0, 0.2)
    now = run(controller, TREMOR, now, 1.0)
    now = run(controller, GRIP, now, 0.8)
    assert controller.state == State.GRASP_ARMED
    assert 0.0 < controller.stiffness < 0.9
    run(controller, GRIP, now, 1.0)
    assert controller.stiffness == 0.0


def test_disarm_drops_the_stiffness_at_once() -> None:
    """disarm() goes to DISARMED with stiffness 0 immediately."""
    controller = armed_controller()
    now = run(controller, GRIP, 0.0, 0.2)
    run(controller, TREMOR, now, 1.0)
    controller.disarm("test")
    assert controller.state == State.DISARMED and controller.stiffness == 0.0
