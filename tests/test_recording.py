"""Session folders in source/recording.py: create, write, read back."""

from pathlib import Path

import numpy as np

import settings
from source.recording import RecordedSource, Recording, SessionWriter, new_session_folder
from source.signal_processing import EMGBlock
from tests.fakes import TWO_SENSORS

ROOT = Path(__file__).resolve().parent.parent


def test_session_folder_name(tmp_path: Path) -> None:
    """<date>-<time>_<source>_<label>, the label made safe; a second one gets a number."""
    first = new_session_folder(tmp_path, "trigno", "grip test/1")
    second = new_session_folder(tmp_path, "trigno", "grip test/1")
    assert first.name.endswith("_trigno_grip-test-1")
    assert second.name == first.name + "-1"


def test_write_and_read_back(tmp_path: Path) -> None:
    """What SessionWriter writes, Recording reads back: channels, rate and samples."""
    samples = np.arange(20.0).reshape(2, 10) / 100
    writer = SessionWriter(tmp_path, TWO_SENSORS, 2000.0, "trigno", {}, label="x")
    writer.write(EMGBlock(samples[:, :4], 0, 0.0))
    writer.write(EMGBlock(samples[:, 4:], 4, 0.0))
    writer.finish([(0.0, "hello")], None)

    recording = Recording(tmp_path)
    assert [c.name for c in recording.channels] == ["FDS", "EDC"]
    assert [c.role for c in recording.channels] == ["flexor", "extensor"]
    assert recording.sample_rate == 2000.0
    assert np.allclose(recording.emg, samples)
    assert recording.metadata["status"] == "complete"
    assert recording.metadata["events"] == [{"time_s": 0.0, "message": "hello"}]


def test_the_committed_recording_replays() -> None:
    """The bench recording loads with its channels and rate, and replays in blocks."""
    folder = ROOT / "measurements" / "20261105-144738_trigno_to-lange-greb"
    source = RecordedSource(folder, loop=False, realtime=False)
    assert [(c.name, c.role) for c in source.channels] == [("FDS", "flexor"), ("EDC", "extensor")]
    assert source.sample_rate == 2000.0
    assert source.recording.emg.shape == (2, 57051)
    block = source.read_block()
    assert block is not None and block.samples.shape == (2, settings.BLOCK_SIZE)
