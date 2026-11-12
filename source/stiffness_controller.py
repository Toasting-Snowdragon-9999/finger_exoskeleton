"""
The stiffness controller: decides when the finger should be stiff because of tremor.

  DISARMED     before calibration (and while recalibrating): stiffness 0
  IDLE         --grasp >= GRASP_ON for GRASP_ON_TIME_S-->                         GRASP_ARMED
  GRASP_ARMED  --grasp < GRASP_OFF for GRASP_OFF_TIME_S-->                        IDLE
               --tremor depth >= TREMOR_ON and ratio >= RATIO_ON
                 for TREMOR_ON_TIME_S-->                                          STIFFENED
  STIFFENED    --grasp < GRASP_OFF for GRASP_OFF_TIME_S-->                        IDLE
               --tremor depth < TREMOR_OFF or ratio < RATIO_OFF
                 for TREMOR_OFF_TIME_S-->                                         GRASP_ARMED

In STIFFENED the stiffness grows with the tremor depth. It rises at most
ENGAGE_RATE_PER_S and falls at most RELEASE_RATE_PER_S. The stiffness is only
shown in the viewer; the motor uses the state (see source/motor.py).
"""

import math
from enum import Enum

import settings
from source.calibration import Activation


class State(Enum):
    """The controller states."""

    DISARMED = "DISARMED"
    IDLE = "IDLE"
    GRASP_ARMED = "GRASP_ARMED"
    STIFFENED = "STIFFENED"


class HoldTimer:
    """Becomes True once a condition has been true for `duration` seconds in a row."""

    def __init__(self, duration: float) -> None:
        self.duration = duration
        self.since: float | None = None

    def update(self, condition: bool, now: float) -> bool:
        """Feed the condition at time `now`; True when it has held long enough."""
        if not condition:
            self.since = None
            return False
        if self.since is None:
            self.since = now
        return now - self.since >= self.duration

    def reset(self) -> None:
        """Forget how long the condition has held."""
        self.since = None


def clamp(value: float, low: float, high: float) -> float:
    """value limited to [low, high]; NaN becomes low."""
    if math.isnan(value):
        return low
    return max(low, min(high, value))


class StiffnessController:
    """The state machine described at the top of this file."""

    def __init__(self) -> None:
        self.state = State.DISARMED
        self.stiffness = 0.0
        self.last_transition = "start-up: disarmed"
        self.last_time: float | None = None
        self.grasp_on = HoldTimer(settings.GRASP_ON_TIME_S)
        self.grasp_off = HoldTimer(settings.GRASP_OFF_TIME_S)
        self.tremor_on = HoldTimer(settings.TREMOR_ON_TIME_S)
        self.tremor_off = HoldTimer(settings.TREMOR_OFF_TIME_S)

    def go_to(self, state: State, reason: str) -> None:
        """Change state and restart all timers."""
        if state != self.state:
            self.last_transition = f"{self.state.value} -> {state.value}: {reason}"
            self.state = state
            for timer in (self.grasp_on, self.grasp_off, self.tremor_on, self.tremor_off):
                timer.reset()

    def arm(self) -> None:
        """Leave DISARMED: from now on the controller follows the activation."""
        if self.state == State.DISARMED:
            self.stiffness = 0.0
            self.go_to(State.IDLE, "armed")

    def disarm(self, reason: str) -> None:
        """Go to DISARMED with stiffness 0 at once."""
        self.stiffness = 0.0
        self.go_to(State.DISARMED, reason)

    def update(self, activation: Activation | None, now: float) -> None:
        """Advance the controller to time `now` (seconds of stream time)."""
        dt = 0.0 if self.last_time is None else clamp(now - self.last_time, 0.0, 0.1)
        self.last_time = now
        if self.state == State.DISARMED or activation is None:
            self.stiffness = 0.0
            return
        self.next_state(activation, now)
        self.ramp_to(self.target_stiffness(activation), dt)

    def next_state(self, activation: Activation, now: float) -> None:
        """Check the conditions and change state when one has held long enough."""
        grasp, tremor, ratio = activation.grasp, activation.tremor_depth, activation.tremor_ratio
        if math.isnan(grasp) or math.isnan(tremor) or math.isnan(ratio):
            grasp, tremor, ratio = 0.0, 0.0, 0.0
        grasp_on = self.grasp_on.update(grasp >= settings.GRASP_ON, now)
        grasp_off = self.grasp_off.update(grasp < settings.GRASP_OFF, now)
        tremor_on = self.tremor_on.update(
            tremor >= settings.TREMOR_ON and ratio >= settings.RATIO_ON, now)
        tremor_off = self.tremor_off.update(
            tremor < settings.TREMOR_OFF or ratio < settings.RATIO_OFF, now)

        if self.state == State.IDLE and grasp_on:
            self.go_to(State.GRASP_ARMED, f"grasp activation {grasp:.2f}")
        elif self.state == State.GRASP_ARMED:
            if grasp_off:
                self.go_to(State.IDLE, f"grasp released ({grasp:.2f})")
            elif tremor_on:
                self.go_to(State.STIFFENED, f"tremor {tremor:.2f}, ratio {ratio:.2f}")
        elif self.state == State.STIFFENED:
            if grasp_off:
                self.go_to(State.IDLE, f"grasp released ({grasp:.2f})")
            elif tremor_off:
                self.go_to(State.GRASP_ARMED, f"tremor subsided ({tremor:.2f}, ratio {ratio:.2f})")

    def target_stiffness(self, activation: Activation) -> float:
        """The stiffness to aim for: 0 unless STIFFENED, then more with more tremor."""
        if self.state != State.STIFFENED:
            return 0.0
        excess = max(0.0, activation.tremor_depth - settings.TREMOR_OFF)
        return clamp(settings.ENGAGE_STIFFNESS + settings.STIFFNESS_GAIN * excess,
                     0.0, settings.MAX_STIFFNESS)

    def ramp_to(self, target: float, dt: float) -> None:
        """Move the stiffness toward the target, no faster than the engage/release rates."""
        if target > self.stiffness:
            self.stiffness = min(target, self.stiffness + settings.ENGAGE_RATE_PER_S * dt)
        else:
            self.stiffness = max(target, self.stiffness - settings.RELEASE_RATE_PER_S * dt)
        self.stiffness = clamp(self.stiffness, 0.0, min(1.0, settings.MAX_STIFFNESS))
