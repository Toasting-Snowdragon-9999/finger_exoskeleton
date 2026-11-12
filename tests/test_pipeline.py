"""The per-block pipeline in source/pipeline.py: calibrate, arm after relaxing, recalibrate."""

from pathlib import Path

import numpy as np

import settings
from source.calibration import Calibration
from source.pipeline import Pipeline
from source.signal_processing import EMGBlock
from source.stiffness_controller import State
from tests.fakes import RATE, make_emg

CHANNELS = settings.CHANNELS


def feed(pipeline: Pipeline, signal: np.ndarray, start_index: int = 0) -> None:
    """Give a (2, n) signal to the pipeline in blocks of 27 samples."""
    for start in range(0, signal.shape[1], 27):
        pipeline.step(EMGBlock(signal[:, start:start + 27], start_index + start, 0.0))


def test_a_grip_held_after_calibration_does_not_arm() -> None:
    """Calibration ends at 10 s while the grip lasts until 11 s: it arms only after relaxing."""
    pipeline = Pipeline(CHANNELS, RATE)
    signal = make_emg([(3.5, 0.01), (7.5, 0.2), (2.0, 0.01)])
    feed(pipeline, signal[:, :round(10.5 * RATE)])
    assert pipeline.calibration is not None
    assert pipeline.state == State.DISARMED and pipeline.phase == "relax to arm"
    feed(pipeline, signal[:, round(10.5 * RATE):], start_index=round(10.5 * RATE))
    assert pipeline.state == State.IDLE and pipeline.phase == "done"
    assert any(text == "Hand relaxed; controller armed" for _, text in pipeline.messages)


def test_a_saved_calibration_is_used_and_arms_after_relaxing(tmp_path: Path) -> None:
    """With a calibration file no routine runs; resting for 0.3 s arms the controller."""
    path = tmp_path / "calibration.json"
    levels = np.array([0.01, 0.01]), np.array([0.2, 0.2])
    Calibration(["FDS", "EDC"], levels[0], levels[1], levels[0], levels[1]).save(path)
    pipeline = Pipeline(CHANNELS, RATE, calibration_file=path)
    feed(pipeline, make_emg([(0.5, 0.01)]))
    assert pipeline.state == State.IDLE
    assert any(text.startswith("Using the saved calibration") for _, text in pipeline.messages)


def test_recalibrate_disarms_and_starts_over() -> None:
    """recalibrate() disarms on the next block and runs the calibration routine again."""
    pipeline = Pipeline(CHANNELS, RATE)
    signal = make_emg([(3.5, 0.01), (6.5, 0.2), (1.0, 0.01)])
    feed(pipeline, signal)
    assert pipeline.state == State.IDLE
    pipeline.recalibrate()
    feed(pipeline, make_emg([(0.1, 0.01)]), start_index=signal.shape[1])
    assert pipeline.state == State.DISARMED and pipeline.calibration is None
    assert pipeline.phase == "rest (baseline)"
