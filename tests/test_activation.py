"""Calibration and normalized activation."""

from pathlib import Path

import numpy as np
import pytest

from source.activation import ActivationEstimator, Calibration, CalibrationError, Calibrator
from source.config import CalibrationConfig
from source.streaming import ProcessedBlock

FS = 1000.0
NAMES = ("F1", "E1")


def block(
    first: int,
    slow: tuple[float, float],
    fast: tuple[float, float],
    tremor: tuple[float, float] = (0.0, 0.0),
    n: int = 100,
) -> ProcessedBlock:
    """A processed block with constant envelopes (depth = tremor / slow, floor 0.01 mV)."""

    def constant(values: tuple[float, float]) -> np.ndarray:
        return np.repeat(np.array(values, dtype=float)[:, None], n, axis=1)

    depth = tuple(t / max(s, 0.01) for t, s in zip(tremor, slow))
    return ProcessedBlock(
        raw=np.zeros((2, n)),
        filtered=np.zeros((2, n)),
        slow_envelope=constant(slow),
        fast_envelope=constant(fast),
        tremor_amplitude=constant(tremor),
        tremor_ratio=constant((0.9, 0.5)),
        tremor_depth=constant((depth[0], depth[1])),
        first_sample_index=first,
        receive_time=0.0,
        sample_rate=FS,
    )


def run_calibration(rest: tuple[float, float], reference: tuple[float, float]) -> Calibrator:
    """Feed 1 s rest, then 1 s reference contraction."""
    calibrator = Calibrator(
        CalibrationConfig(
            baseline_window_s=(0.0, 1.0),
            reference_window_s=(1.0, 2.0),
            reference_percentile=95.0,
            min_reference_ratio=3.0,
        ),
        NAMES,
        FS,
    )
    for i in range(10):
        calibrator.update(block(i * 100, rest, rest))
    for i in range(10, 21):
        calibrator.update(block(i * 100, reference, reference))
    return calibrator


def test_calibration_from_windows() -> None:
    """Rest and reference levels come from their windows."""
    calibrator = run_calibration((0.01, 0.02), (0.5, 0.4))
    assert calibrator.done and calibrator.result is not None
    np.testing.assert_allclose(calibrator.result.rest_slow, [0.01, 0.02])
    np.testing.assert_allclose(calibrator.result.reference_slow, [0.5, 0.4])
    assert calibrator.phase == "done" and calibrator.seconds_left == 0.0


def test_weak_reference_is_rejected() -> None:
    """A reference contraction barely above rest fails calibration with a clear message."""
    calibrator = run_calibration((0.01, 0.02), (0.02, 0.4))
    assert calibrator.result is None
    assert calibrator.error is not None and "F1" in calibrator.error
    assert calibrator.phase == "failed"


def test_calibration_phase_follows_stream_time() -> None:
    """The phase and the time left tell the user what to do."""
    calibrator = Calibrator(
        CalibrationConfig(baseline_window_s=(0.0, 1.0), reference_window_s=(1.5, 2.0)), NAMES, FS
    )
    assert calibrator.phase == "rest (baseline)"
    assert calibrator.seconds_left == pytest.approx(1.0)
    for i in range(12):
        calibrator.update(block(i * 100, (0.01, 0.01), (0.01, 0.01)))
    assert calibrator.phase == "prepare reference contraction"
    assert calibrator.seconds_left == pytest.approx(1.5 - 1.199)


def test_recalibration_restarts_from_the_next_block() -> None:
    """restart() measures stream time from the next block, not from the stream start."""
    calibrator = run_calibration((0.01, 0.02), (0.5, 0.4))
    calibrator.restart()
    assert not calibrator.done and calibrator.phase == "rest (baseline)"
    for i in range(50, 60):
        calibrator.update(block(i * 100, (0.02, 0.03), (0.02, 0.03)))
    for i in range(60, 71):
        calibrator.update(block(i * 100, (0.5, 0.4), (0.5, 0.4)))
    assert calibrator.result is not None
    np.testing.assert_allclose(calibrator.result.rest_slow, [0.02, 0.03])


def test_normalization_and_summaries() -> None:
    """0 at rest, 1 at reference; grasp uses flexors only; tremor scaled by the fast span."""
    calibration = run_calibration((0.01, 0.02), (0.51, 0.42)).result
    assert calibration is not None
    estimator = ActivationEstimator(calibration, ("flexor", "extensor"), tremor_measure="level")

    rest = estimator.estimate(block(0, (0.01, 0.02), (0.01, 0.02)))
    assert rest.grasp_activation == pytest.approx(0.0)
    full = estimator.estimate(block(0, (0.51, 0.02), (0.51, 0.02), tremor=(0.1, 0.2)))
    assert full.grasp_activation == pytest.approx(1.0)
    np.testing.assert_allclose(full.slow, [1.0, 0.0], atol=1e-12)
    # Extensor tremor 0.2 / (0.42 - 0.02) = 0.5 is the largest level
    assert full.tremor_level == pytest.approx(0.5)
    assert full.tremor_ratio == pytest.approx(0.5)
    assert full.tremor_depth == pytest.approx(0.2 / 0.02)
    assert full.tremor_measure("level") == full.tremor_level
    assert full.tremor_measure("depth") == full.tremor_depth
    below_rest = estimator.estimate(block(0, (0.0, 0.0), (0.0, 0.0)))
    assert below_rest.grasp_activation == 0.0


def test_depth_measure_picks_the_channel_with_the_deepest_tremor() -> None:
    """With tremor_measure="depth" the summaries come from the channel with the largest depth."""
    calibration = run_calibration((0.01, 0.02), (0.51, 0.42)).result
    assert calibration is not None
    estimator = ActivationEstimator(calibration, ("flexor", "extensor"), tremor_measure="depth")
    # Flexor: depth 0.1 / 0.51 = 0.2, level 0.1 / 0.5 = 0.2; extensor: depth 0.05 / 0.4 = 0.125
    estimate = estimator.estimate(block(0, (0.51, 0.4), (0.51, 0.4), tremor=(0.1, 0.05)))
    assert estimate.tremor_depth == pytest.approx(0.1 / 0.51)
    assert estimate.tremor_level == pytest.approx(0.2)
    assert estimate.tremor_ratio == pytest.approx(0.9)


def test_grasp_uses_every_channel_without_a_flexor() -> None:
    """A montage without flexor roles (single sensor of unknown role) still detects grasp."""
    calibration = run_calibration((0.01, 0.02), (0.51, 0.42)).result
    assert calibration is not None
    estimator = ActivationEstimator(calibration, ("other", "other"))
    estimate = estimator.estimate(block(0, (0.51, 0.02), (0.51, 0.02)))
    assert estimate.grasp_activation == pytest.approx(0.5)


def test_save_and_load(tmp_path: Path) -> None:
    """A saved calibration loads back identically."""
    calibration = run_calibration((0.01, 0.02), (0.5, 0.4)).result
    assert calibration is not None
    path = tmp_path / "calibration.json"
    calibration.save(path)
    loaded = Calibration.load(path)
    assert loaded.channel_names == NAMES
    np.testing.assert_allclose(loaded.reference_fast, calibration.reference_fast)
    with pytest.raises(CalibrationError):
        Calibration.load(tmp_path / "missing.json")


def test_estimator_rejects_mismatched_channels() -> None:
    """The calibration must match the channel configuration."""
    calibration = run_calibration((0.01, 0.02), (0.5, 0.4)).result
    assert calibration is not None
    with pytest.raises(CalibrationError):
        ActivationEstimator(calibration, ("flexor",))
