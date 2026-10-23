"""
This module contains the different control systems implemented for the finger exoskeleton.
"""

import time

import numpy as np


# Controller to maintain tension in the tendon at all times
class TendonTensionController:
    """Takes up tendon slack, then lets go. It never holds a position.

    The motor must be in current control mode. There is no tension sensor, so
    the controller works with a small state machine based on motor velocity:

      REEL  Ramp a small pull current up (ramp_s) in the wind-in direction.
            While the tendon is slack the motor spins and reels it in. When
            the tendon goes taut the motor stalls (velocity ~ 0 for stall_s)
            -> go to IDLE. If the user pulls the tendon out while reeling,
            the tendon is obviously taut -> go to IDLE as well.

      IDLE  Command 0 A. The motor applies no pull, so the finger can rest
            in ANY position and nothing is forced back to a set point. The
            gearbox friction keeps the spool where it is, so the tendon stays
            taut. If the user pulls the tendon out (finger extends) an
            optional assist can cancel gearbox friction so movement is free.

    Without a sensor, slack that appears while IDLE (the user flexes the
    finger) cannot be seen directly. So every probe_period_s the controller
    runs a short REEL: if the tendon is taut the motor stalls almost
    immediately and goes back to IDLE; if it is slack, it reels it in first.
    Set probe_period_s=None to disable re-probing (only the first take-up).

    IMPORTANT: tension_ma must be LOW (just above the gearbox friction).
    A taut tendon stalls the motor, but a high current would then pull the
    finger into flexion. Find the lowest current that still reels in slack.

    Parameters
    ----------
    motors          : Motors instance (DynamixelHardwareInterface)
    motor_id        : ID of the tendon motor
    tension_ma      : peak pull current in mA during REEL (keep as low as works)
    wind_direction  : +1 or -1, the sign of current/velocity that reels the tendon IN
    probe_period_s  : seconds between slack checks while IDLE (None = never)
    ramp_s          : seconds to ramp the pull from 0 to tension_ma in REEL
    stall_s         : seconds of ~zero velocity (after the ramp) that means "taut"
    stall_velocity  : rad/s below which the motor counts as not moving
    max_reel_s      : safety timeout for one REEL; gives up and goes IDLE
    damping         : mA removed from the pull per rad/s of reel-in speed
    assist_ma       : optional pay-out assist in IDLE while the user pulls the
                      tendon out (cancels gearbox friction). Default 0 = off.
                      Keep it below the friction level or the motor will
                      unspool the tendon on its own.
    payout_threshold: rad/s of pay-out speed that counts as "user is pulling"
    assist_taper    : rad/s of pay-out speed at which assist has faded to zero
    release_s       : seconds assist stays on after the last pay-out motion
    wind_limit_ticks: optional (min_pos, max_pos) in raw ticks; pull is cut
                      when the position leaves this window in the wind-in direction
    rate_hz         : control loop rate
    """

    CUR_UNIT_MA = 2.69  # 1 raw current unit = 2.69 mA (same constant as Motors)
    REEL = "REEL"
    IDLE = "IDLE"

    def __init__(self, motors, motor_id, tension_ma=30.0, wind_direction=1,
                 probe_period_s=1.0, ramp_s=0.3, stall_s=0.15, stall_velocity=0.03,
                 max_reel_s=5.0, damping=20.0, assist_ma=0.0, payout_threshold=0.05,
                 assist_taper=2.0, release_s=0.15, wind_limit_ticks=None, rate_hz=100.0):
        assert motor_id in motors.motor_ids, f"Motor ID {motor_id} not found."
        assert wind_direction in (1, -1), "wind_direction must be +1 or -1"
        assert motors.control_mode[motors.motor_ids.index(motor_id)] == motors.CUR_CONTROL, \
            "TendonTensionController requires the motor to be in current control mode."

        self.motors = motors
        self.motor_id = motor_id
        self.idx = motors.motor_ids.index(motor_id)
        self.wind_direction = wind_direction
        self.probe_period_s = probe_period_s
        self.ramp_s = ramp_s
        self.stall_s = stall_s
        self.stall_velocity = stall_velocity
        self.max_reel_s = max_reel_s
        self.damping_raw = damping / self.CUR_UNIT_MA   # raw units per rad/s
        self.payout_threshold = payout_threshold
        self.assist_taper = assist_taper
        self.release_s = release_s
        self.wind_limit_ticks = wind_limit_ticks
        self.period = 1.0 / rate_hz
        self.max_raw = float(motors.maxCur[self.idx])    # current limit read from the motor

        self.tension_raw = 0.0
        self.assist_raw = 0.0
        self.set_tension_ma(tension_ma)
        self.set_assist_ma(assist_ma)

        self.state = self.REEL
        self._state_t0 = None
        self._last_motion = None
        self._next_probe = None
        self._assist_until = 0.0
        self._stop = False

    def set_tension_ma(self, tension_ma):
        """Change the peak pull current (mA). Can be called while running."""
        self.tension_raw = float(np.clip(tension_ma / self.CUR_UNIT_MA, 0.0, self.max_raw))

    def set_assist_ma(self, assist_ma):
        """Change the pay-out assist current (mA). Can be called while running."""
        self.assist_raw = float(np.clip(assist_ma / self.CUR_UNIT_MA, 0.0, self.max_raw))

    def request_reel(self, now=None):
        """Start a take-up (e.g. call this after a release in a grasp controller)."""
        self._enter_reel(time.perf_counter() if now is None else now)

    def _enter_reel(self, now):
        self.state = self.REEL
        self._state_t0 = now
        self._last_motion = now

    def _enter_idle(self, now):
        self.state = self.IDLE
        self._next_probe = None if self.probe_period_s is None else now + self.probe_period_s

    def compute_command(self, velocity, position=None, now=None):
        """Raw current command from the measured velocity (rad/s) and position (ticks)."""
        if now is None:
            now = time.perf_counter()
        if self._state_t0 is None:                       # first call: start with a take-up
            self._enter_reel(now)

        v_in = self.wind_direction * velocity            # > 0: tendon being reeled in
        magnitude = 0.0

        # User pulling the tendon out: it is taut. Stop reeling, optionally assist.
        if v_in < -self.payout_threshold:
            self._assist_until = now + self.release_s
            if self.state == self.REEL:
                self._enter_idle(now)
        elif v_in > self.payout_threshold:
            self._assist_until = 0.0

        if self.state == self.IDLE and self._next_probe is not None and now >= self._next_probe \
                and now >= self._assist_until:
            self._enter_reel(now)                        # periodic slack check

        if now < self._assist_until:
            taper = max(0.0, 1.0 - max(-v_in, 0.0) / self.assist_taper)
            magnitude = -self.assist_raw * taper         # negative = pay-out direction
        elif self.state == self.REEL:
            t = now - self._state_t0
            ramp = 1.0 if self.ramp_s <= 0 else min(t / self.ramp_s, 1.0)
            magnitude = max(self.tension_raw * ramp - self.damping_raw * max(v_in, 0.0), 0.0)

            if abs(velocity) > self.stall_velocity:
                self._last_motion = now
            stalled = t >= self.ramp_s and (now - self._last_motion) >= self.stall_s
            timed_out = t > self.max_reel_s
            wound_too_far = False
            if self.wind_limit_ticks is not None and position is not None:
                lo, hi = self.wind_limit_ticks
                wound_too_far = position > hi if self.wind_direction > 0 else position < lo
            if stalled or timed_out or wound_too_far:
                self._enter_idle(now)                    # tendon taut: let go
                magnitude = 0.0

        magnitude = float(np.clip(magnitude, -self.max_raw, self.max_raw))
        return int(round(self.wind_direction * magnitude))

    def step(self):
        """One control iteration: read feedback, compute, send. Returns the raw command."""
        velocity = self.motors.get_velocity(self.motor_id)
        position = self.motors.get_position(self.motor_id)
        command = self.compute_command(velocity, position)
        self.motors.sendMotorCommand(self.motor_id, command)
        print(
            f"[{self.state}] "
            f"velocity={velocity:.3f} rad/s, "
            f"position={position}, "
            f"command={command} raw "
            f"({command * self.CUR_UNIT_MA:.1f} mA)"
        )
        return command

    def stop(self):
        self._stop = True

    def run(self, duration=None):
        """Blocking fixed-rate loop. Ends after `duration` seconds, on stop(), or Ctrl+C."""
        self._stop = False
        self._state_t0 = None
        t_end = None if duration is None else time.perf_counter() + duration
        next_t = time.perf_counter()
        try:
            while not self._stop and (t_end is None or time.perf_counter() < t_end):
                self.step()
                next_t += self.period
                time.sleep(max(0.0, next_t - time.perf_counter()))
        except KeyboardInterrupt:
            pass
        finally:
            self.motors.sendMotorCommand(self.motor_id, 0)  # release the pull