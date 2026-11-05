"""
The control stage of the live pipeline: calibration, activation estimation
and the controller, driven one processed block at a time by StreamCore
(acquisition.py).

  ProcessedBlock -> Calibrator (until calibrated)
                 -> ActivationEstimator -> ExoskeletonController -> ControllerOutput

Calibration starts with the first block: the user rests during the baseline
window and makes the reference grip during the reference window
(CalibrationConfig). When it succeeds the controller is armed automatically
as soon as the hand has relaxed: grasp activation at most grasp_off_threshold
for grasp_off_time_s (ControllerConfig). Until then it stays disarmed (phase
"relax to arm"), so a reference grip that is still held when calibration ends
does not count as a grasp; with run_emg.py --motor the finger stays open. A
failed calibration leaves the controller disarmed; request_recalibration()
(key 'c' in the viewer) starts over from the next block.

The controller is timed in stream time (sample index / sample rate), so a
replayed recording gives the same result as the live run it came from.
"""

import threading
from collections.abc import Callable

from .activation import ActivationEstimate, ActivationEstimator, Calibrator
from .config import AppConfig
from .controller import ControllerOutput, ExoskeletonController
from .streaming import ProcessedBlock


class ControlStage:  # pylint: disable=too-many-instance-attributes
    """Calibration, activation and control for one block at a time."""

    def __init__(
        self, config: AppConfig, sample_rate: float, message: Callable[[str], None]
    ) -> None:
        self.config = config
        self.sample_rate = sample_rate
        self._message = message
        self.calibrator = Calibrator(config.calibration, config.channel_names, sample_rate)
        self.estimator: ActivationEstimator | None = None
        self.controller = ExoskeletonController(config.controller)
        self.activation: ActivationEstimate | None = None
        self.last_output: ControllerOutput | None = None
        self._reported_error: str | None = None
        self._recalibrate = threading.Event()
        self._arm_pending = False  # calibrated, waiting for the hand to relax
        self._relaxed_since: float | None = None

    @property
    def calibrated(self) -> bool:
        """True when activation can be estimated."""
        return self.estimator is not None

    @property
    def phase(self) -> str:
        """What the user should be doing: the calibration phase, or "relax to arm"."""
        return "relax to arm" if self._arm_pending else self.calibrator.phase

    @property
    def tremor_measure(self) -> str:
        """Which tremor measure the controller thresholds apply to."""
        return self.config.controller.tremor_measure

    @property
    def thresholds(self) -> dict[str, float]:
        """The controller thresholds, for the viewer."""
        controller = self.config.controller
        return {
            "grasp on": controller.grasp_on_threshold,
            "grasp off": controller.grasp_off_threshold,
            "tremor on": controller.tremor_on_threshold,
            "tremor off": controller.tremor_off_threshold,
            "ratio gate": controller.tremor_min_ratio,
        }

    def request_recalibration(self) -> None:
        """Start a new calibration from the next block (thread-safe)."""
        self._recalibrate.set()

    def _restart_calibration(self) -> None:
        self.controller.disarm("recalibration")
        self.estimator = None
        self._arm_pending = False
        self._relaxed_since = None
        self.activation = None
        self._reported_error = None
        self.calibrator.restart()
        windows = self.config.calibration
        self._message(
            f"Calibration restarted: rest {windows.baseline_window_s[0]:g}-"
            f"{windows.baseline_window_s[1]:g} s, reference grip "
            f"{windows.reference_window_s[0]:g}-{windows.reference_window_s[1]:g} s"
        )

    def step(self, processed: ProcessedBlock) -> ControllerOutput:
        """Calibrate or estimate activation from one block, then advance the controller."""
        if self._recalibrate.is_set():
            self._recalibrate.clear()
            self._restart_calibration()
        if self.estimator is None:
            self.calibrator.update(processed)
            self._finish_calibration()
        stream_time = (processed.first_sample_index + processed.n_samples) / self.sample_rate
        if self.estimator is not None:
            self.activation = self.estimator.estimate(processed)
            if self._arm_pending:
                self._arm_when_relaxed(self.activation, stream_time)
        self.last_output = self.controller.update(self.activation, (), stream_time)
        return self.last_output

    def _arm_when_relaxed(self, activation: ActivationEstimate, now: float) -> None:
        """Arm once the grasp activation has stayed at rest for grasp_off_time_s."""
        controller = self.config.controller
        if activation.grasp_activation > controller.grasp_off_threshold:
            self._relaxed_since = None
            return
        if self._relaxed_since is None:
            self._relaxed_since = now
        if now - self._relaxed_since >= controller.grasp_off_time_s:
            self._arm_pending = False
            self.controller.arm()
            self._message("Hand relaxed; controller armed")

    def _finish_calibration(self) -> None:
        calibrator = self.calibrator
        if calibrator.result is not None:
            self.estimator = ActivationEstimator(
                calibrator.result, self.config.channel_roles, self.tremor_measure
            )
            self._arm_pending = True
            self._relaxed_since = None
            self._message("Calibration complete; relax the hand to arm the controller")
            levels = " ".join(
                f"{name} {rest * 1000:.0f}/{ref * 1000:.0f}"
                for name, rest, ref in zip(
                    calibrator.channel_names,
                    calibrator.result.rest_slow,
                    calibrator.result.reference_slow,
                )
            )
            self._message(f"  rest/reference uV: {levels}")
        elif calibrator.error is not None and self._reported_error != calibrator.error:
            self._reported_error = calibrator.error
            self._message(f"Calibration failed: {calibrator.error} Press 'c' to repeat it.")
