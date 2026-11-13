"""The GRU / DSP comparison in source/gru/evaluation.py (no PyTorch needed)."""

from dataclasses import replace

import numpy as np

from source.gru.evaluation import evaluate, format_results, intention_metrics, tremor_metrics
from source.gru.synthetic import add_tremor, synthetic_recording
from tests.fakes import FakeModel

NAN = np.nan


def test_intention_metrics_only_count_known_labels() -> None:
    """Unknown labels (NaN) are left out; the errors are over the rest."""
    predicted = np.array([0.0, 0.5, 1.0, 9.0])
    label = np.array([0.0, 1.0, 1.0, NAN])
    metrics = intention_metrics(predicted, label)
    assert metrics["n"] == 3 and np.isclose(metrics["mae"], 0.5 / 3)
    assert intention_metrics(predicted, np.full(4, NAN)) == {"n": 0}


def test_tremor_metrics() -> None:
    """Precision, recall, false alarms per minute and the latency of the detection."""
    #                 calm ........ tremor ............ unknown  calm
    label = np.array([0, 0, 0, 0,   1, 1, 1, 1, 1, 1,   NAN,     0, 0])
    found = np.array([0, 1, 0, 0,   0, 0, 1, 1, 1, 1,   1,       0, 0]) > 0
    metrics = tremor_metrics(found, label, frame_rate=1.0)  # one frame per second
    assert metrics["n"] == 12
    assert np.isclose(metrics["precision"], 4 / 5) and np.isclose(metrics["recall"], 4 / 6)
    assert np.isclose(metrics["false_alarms_per_minute"], 1 / (6 / 60))  # one start in 6 calm s
    assert metrics["median_latency_s"] == 2.0  # tremor starts at frame 4, found at frame 6


def test_evaluate_scores_both_methods_on_the_same_recordings() -> None:
    """Both methods get intention/grasp and tremor metrics, and synthetic data is flagged."""
    roles = ["flexor", "extensor"]
    recordings = [add_tremor(synthetic_recording(seed, roles, seconds=30.0), seed)
                  for seed in (1, 2)]
    result = evaluate(FakeModel(roles, intention=0.0, probability=0.9), recordings)
    assert result["synthetic_data"] and len(result["recordings"]) == 2
    assert result["gru"]["tremor"]["recall"] == 1.0  # the fake model always says "tremor"
    assert result["gru"]["tremor"]["precision"] < 0.8
    assert result["dsp"]["grasp"]["n"] == result["gru"]["grasp"]["n"] > 1000
    assert result["dsp"]["grasp"]["mae"] < result["gru"]["grasp"]["mae"]  # "always 0" is worse
    text = format_results(result)
    assert text.startswith("SYNTHETIC DATA") and "DSP tremor" in text


def test_dsp_grasp_needs_a_calibration_grip_level() -> None:
    """Without a grip level (a public dataset) the DSP grasp is reported as not available."""
    recording = replace(synthetic_recording(3, ["flexor"], seconds=30.0), grip_level=None)
    result = evaluate(FakeModel(["flexor"]), [recording])
    assert result["dsp"]["grasp"] == {"n": 0} and result["dsp"]["tremor"]["n"] > 0
    assert "no labels (or no calibration grip level)" in format_results(result)
