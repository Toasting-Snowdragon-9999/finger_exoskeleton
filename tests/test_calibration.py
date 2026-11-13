"""Calibration, the calibration file and the activation in source/calibration.py."""

import json
from pathlib import Path

import numpy as np

from source.calibration import Calibration, Calibrator, compute_activation, load_calibration
from source.signal_processing import ProcessedBlock

ROOT = Path(__file__).resolve().parent.parent
NAMES = ["FDS", "EDC"]


def block(first: int, n: int, slow: float, fast: float | None = None) -> ProcessedBlock:
    """A processed block with constant envelopes on both channels."""
    level = np.full((2, n), slow)
    fast_level = np.full((2, n), slow if fast is None else fast)
    zero = np.zeros((2, n))
    return ProcessedBlock(zero, zero, level, fast_level, zero, zero, zero, first, 0.0)


def calibrate(grip: float) -> Calibrator:
    """Rest at 0.01 mV for 3.5 s, then grip until the calibration ends."""
    calibrator = Calibrator(NAMES, 2000.0)
    for first in range(0, 21000, 100):
        calibrator.update(block(first, 100, 0.01 if first < 7000 else grip))
    return calibrator


def test_calibration_finds_rest_and_grip_levels() -> None:
    """Rest is the mean of the rest window, grip the 95th percentile of the grip window."""
    calibrator = calibrate(grip=0.1)
    assert calibrator.result is not None and calibrator.phase == "done"
    assert np.allclose(calibrator.result.rest_slow, 0.01)
    assert np.allclose(calibrator.result.reference_slow, 0.1)


def test_a_weak_grip_fails() -> None:
    """A grip below 3x the rest level is not accepted."""
    calibrator = calibrate(grip=0.02)
    assert calibrator.result is None
    assert calibrator.error is not None and "not at least 3x" in calibrator.error


def make_calibration() -> Calibration:
    """Rest 0.01 mV and grip 0.11 mV on both channels."""
    levels = np.array([0.01, 0.01]), np.array([0.11, 0.11])
    return Calibration(NAMES, levels[0], levels[1], levels[0], levels[1])


def test_activation_is_0_at_rest_and_1_at_the_grip() -> None:
    """Activation is measured from rest (0) to the calibration grip (1), never below 0."""
    flexors = np.array([True, True])
    calibration = make_calibration()
    assert compute_activation(block(0, 10, 0.01), calibration, flexors).grasp == 0.0
    assert compute_activation(block(0, 10, 0.11), calibration, flexors).grasp == 1.0
    assert compute_activation(block(0, 10, 0.001), calibration, flexors).grasp == 0.0


def test_grasp_uses_only_the_flexor_channels() -> None:
    """The extensor channel does not count for the grasp."""
    processed = block(0, 10, 0.01)
    processed.slow[0, :] = 0.06  # flexor halfway, extensor at rest
    activation = compute_activation(processed, make_calibration(), np.array([True, False]))
    assert np.isclose(activation.grasp, 0.5)


def test_save_and_load(tmp_path: Path) -> None:
    """A saved calibration loads again with the same levels."""
    path = tmp_path / "calibration.json"
    make_calibration().save(path)
    loaded, message = load_calibration(path, NAMES)
    assert loaded is not None and "loaded" in message
    assert np.allclose(loaded.reference_slow, 0.11)


def test_the_committed_calibration_files_load() -> None:
    """The calibration files used by the bench commands in kommandoer.txt still fit."""
    for path in (ROOT / "calibration_replay.json", ROOT / "measurements" / "plac_calibration.json"):
        names = json.loads(path.read_text())["channel_names"]  # one or two sensors
        loaded, message = load_calibration(path, names)
        assert loaded is not None, message


def test_a_file_for_other_channels_or_settings_is_ignored(tmp_path: Path) -> None:
    """Other channel names or other filter settings make the file unusable."""
    path = tmp_path / "calibration.json"
    make_calibration().save(path)
    loaded, message = load_calibration(path, ["FDS", "APB"])
    assert loaded is None and "channels" in message

    data = json.loads(path.read_text())
    data["processing"]["notch_hz"] = 60.0
    path.write_text(json.dumps(data))
    loaded, message = load_calibration(path, NAMES)
    assert loaded is None and "settings changed" in message
