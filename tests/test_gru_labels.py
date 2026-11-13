"""Labelled recordings and synthetic data in source/gru/labels.py and synthetic.py."""

from pathlib import Path

import numpy as np
import pytest

from source.gru.labels import load_npz, load_session, read_labels
from source.gru.synthetic import add_tremor, synthetic_recording
from source.recording import SessionWriter
from source.signal_processing import EMGBlock
from tests.fakes import TWO_SENSORS, make_emg

ROOT = Path(__file__).resolve().parent.parent
MEASUREMENTS = ROOT / "measurements"
BENCH = "20261105-144738_trigno_to-lange-greb"


def test_read_labels_with_unknowns_and_reaction_margin(tmp_path: Path) -> None:
    """Empty = unknown; after a change of a label that label is unknown for the margin."""
    path = tmp_path / "labels.csv"
    path.write_text("start_s,end_s,intention,tremor\n0,2,0,0\n2,4,1,0\n4,6,,1\n")
    times = np.array([1.0, 2.2, 3.0, 4.2, 5.0, 7.0])
    intention, tremor = read_labels(path, times, margin_s=0.5)
    assert np.allclose(intention, [0, np.nan, 1, np.nan, np.nan, np.nan], equal_nan=True)
    # tremor only changes at 4 s: unknown until 4.5 s; 7 s is outside every row
    assert np.allclose(tremor, [0, 0, 0, np.nan, 1, np.nan], equal_nan=True)


def test_bench_recording_loads_with_its_tremor_labels() -> None:
    """The committed tremor recording: roles, tremor labels and the group of its original."""
    recording = load_session(MEASUREMENTS / f"{BENCH}_tremor")
    assert recording.roles == ["flexor", "extensor"] and recording.sample_rate == 2000.0
    assert recording.group == BENCH and recording.synthetic  # same group as the original
    assert np.all(np.isnan(recording.intention))  # nobody recorded what was intended

    def at(seconds: float) -> float:
        return float(recording.tremor[round(seconds * 2000)])

    assert at(5.0) == 0 and at(9.0) == 1 and at(24.0) == 1 and np.isnan(at(7.6))
    assert load_session(MEASUREMENTS / BENCH).group == BENCH


def test_rest_level_comes_from_the_calibration_that_was_used(tmp_path: Path) -> None:
    """calibration file > calibration in session.json > the rest window of the recording."""
    from_window = load_session(MEASUREMENTS / BENCH).rest_level
    assert np.allclose(from_window, [0.0024, 0.0042], atol=0.0002)
    from_file = load_session(MEASUREMENTS / BENCH, ROOT / "calibration_replay.json").rest_level
    assert np.allclose(from_file, [0.0036, 0.0056], atol=0.0001)

    writer = SessionWriter(tmp_path, TWO_SENSORS, 2000.0, "trigno", {})
    writer.write(EMGBlock(make_emg([(4.0, 0.01)]), 0, 0.0))
    writer.finish([], None, calibration={"channel_names": ["FDS", "EDC"],
                                         "rest_slow": [0.5, 0.25], "reference_slow": [5, 4]})
    stored = load_session(tmp_path)
    assert stored.rest_level.tolist() == [0.5, 0.25]
    assert stored.grip_level is not None and stored.grip_level.tolist() == [5, 4]


def test_npz_dataset_is_resampled_and_keeps_its_labels(tmp_path: Path) -> None:
    """A converted dataset at 1000 Hz comes back at 2000 Hz with labels of the same length."""
    path = tmp_path / "subject1.npz"
    np.savez(path, emg=make_emg([(4.0, 0.01)])[:, ::2], sample_rate=1000.0,
             roles=["flexor", "extensor"], tremor=np.ones(4000), group="subject1")
    recording = load_npz(path)
    assert recording.sample_rate == 2000.0 and recording.emg.shape == (2, 8000)
    assert recording.tremor.shape == (8000,) and np.all(recording.tremor == 1)
    assert np.all(np.isnan(recording.intention)) and recording.group == "subject1"
    assert recording.rest_level.shape == (2,) and np.all(recording.rest_level > 0)


def test_select_roles_picks_channels_by_muscle() -> None:
    """A one-sensor model gets the flexor channel; a missing role is an error."""
    recording = load_session(MEASUREMENTS / BENCH)
    flexor = recording.select_roles(["flexor"])
    assert flexor.emg.shape[0] == 1 and np.array_equal(flexor.emg[0], recording.emg[0])
    assert flexor.rest_level.shape == (1,)
    with pytest.raises(ValueError, match="no thumb channel"):
        recording.select_roles(["thumb"])


def test_add_tremor_labels_what_it_adds() -> None:
    """Tremor label 1 where the EMG was changed, the old label elsewhere, and a new name."""
    base = load_session(MEASUREMENTS / BENCH)
    shaken = add_tremor(base, seed=1)
    changed = np.any(shaken.emg != base.emg, axis=0)
    assert np.all(changed[shaken.tremor == 1]) and not np.any(changed[shaken.tremor == 0])
    assert 0.2 < np.mean(shaken.tremor == 1) < 0.7
    assert shaken.synthetic and shaken.group == base.group and shaken.name != base.name


def test_synthetic_recording_is_marked_synthetic() -> None:
    """The smoke-test data: flagged synthetic, opening only with an extensor channel."""
    one = synthetic_recording(0, ["flexor"], seconds=30.0)
    two = synthetic_recording(0, ["flexor", "extensor"], seconds=30.0)
    assert one.synthetic and one.emg.shape[0] == 1 and one.intention.min() == 0.0
    assert two.emg.shape[0] == 2 and two.intention.min() == -1.0
    assert np.all(one.tremor == 0) and 0.001 < one.rest_level[0] < 0.004
