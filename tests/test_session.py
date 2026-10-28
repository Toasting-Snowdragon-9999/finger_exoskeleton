"""Measurement sessions: one folder per run with the raw EMG and its metadata."""

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

import applied_config
import run_emg
from source.import_data import load_columns
from source.session import EMG_FILE, METADATA_FILE, create_session_dir, find_recording

NOW = datetime(2026, 10, 23, 14, 5, 12)


def test_session_dirs_are_unique_and_named(tmp_path: Path) -> None:
    """Folders are named by time, source and a cleaned label, and never reused."""
    first = create_session_dir(tmp_path / "new", "trigno", "grip test/1", NOW)
    second = create_session_dir(tmp_path / "new", "trigno", "grip test/1", NOW)
    assert first.name == "20261023-140512_trigno_grip-test-1"
    assert second.name == "20261023-140512_trigno_grip-test-1-1"
    assert create_session_dir(tmp_path, "synthetic", " ", NOW).name == "20261023-140512_synthetic"


def test_cli_out_saves_each_run_in_its_own_session(tmp_path: Path) -> None:
    """Two runs give two session folders, each with emg.csv and a complete session.json."""
    args = ["--source", "synthetic", "--headless", "--duration", "1.0", "--out", str(tmp_path)]
    assert run_emg.main([*args, "--label", "grip", "--note", "subject 1"]) == 0
    assert run_emg.main(args) == 0

    sessions = sorted(tmp_path.iterdir())
    assert len(sessions) == 2
    labelled = next(path for path in sessions if path.name.endswith("_grip"))
    metadata = json.loads((labelled / METADATA_FILE).read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
    assert metadata["source"] == "synthetic" and metadata["note"] == "subject 1"
    assert metadata["sample_rate_hz"] == 2000.0
    assert [channel["name"] for channel in metadata["channels"]] == ["FDS"]
    assert metadata["config"]["processing"]["notch_hz"] == 50.0
    assert metadata["events"][0] == {
        "time_s": 0.0,
        "message": "Source started: synthetic at 2000 Hz",
    }

    rows = (labelled / EMG_FILE).read_text(encoding="utf-8").splitlines()
    assert rows[0] == "time_s,FDS_mV"
    assert metadata["samples_per_channel"] == len(rows) - 1 > 1000
    times = load_columns(labelled / EMG_FILE, [0])[0]
    assert times[0] == 0.0 and times[1] == pytest.approx(1 / 2000.0)


def test_cli_out_leaves_no_folder_when_the_source_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a TCU no empty session folder is left behind."""
    monkeypatch.setattr(
        applied_config,
        "TRIGNO",
        replace(applied_config.TRIGNO, command_port=1, connect_timeout_s=0.3),
    )
    code = run_emg.main(["--source", "trigno", "--headless", "--out", str(tmp_path)])
    assert code == run_emg.EXIT_SOURCE
    assert not list(tmp_path.iterdir())


def test_cli_out_and_save_are_exclusive(tmp_path: Path) -> None:
    """--out and --save cannot be combined."""
    with pytest.raises(SystemExit):
        run_emg.main(["--out", str(tmp_path), "--save", str(tmp_path / "x.csv")])


def record_four_channels(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A 1 s synthetic session with the four-sensor montage; CHANNELS is then reset."""
    monkeypatch.setattr(applied_config, "CHANNELS", applied_config.FOUR_SENSORS)
    args = ["--source", "synthetic", "--headless", "--duration", "1.0", "--out", str(root)]
    assert run_emg.main(args) == 0
    monkeypatch.setattr(applied_config, "CHANNELS", applied_config.ONE_SENSOR)
    (session,) = root.iterdir()
    return session


def test_session_loads_for_analysis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A session folder, its emg.csv or its session.json all give the recorded channels."""
    session = record_four_channels(tmp_path, monkeypatch)
    for path in (session, session / EMG_FILE, session / METADATA_FILE):
        recording = find_recording(path)
        assert recording is not None and recording.emg_path == session / EMG_FILE
        assert recording.sample_rate_hz == 2000.0
        assert [channel.name for channel in recording.channels] == ["FDS", "EDC", "FDI", "APB"]
        assert [channel.trigno_slot for channel in recording.channels] == [1, 2, 3, 4]
    assert recording is not None
    times, emg = recording.load()
    assert emg.shape == (4, times.size) and times.size > 1000
    np.testing.assert_allclose(emg[1], load_columns(session / EMG_FILE, [2])[0])


def test_cli_replays_a_session_exactly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replaying a session (and saving the replay) reproduces the recorded samples."""
    session = record_four_channels(tmp_path / "live", monkeypatch)
    args = ["--source", "recorded", "--file", str(session), "--no-loop", "--headless"]
    assert run_emg.main([*args, "--out", str(tmp_path / "replay")]) == 0

    (replay,) = (tmp_path / "replay").iterdir()
    assert (replay / EMG_FILE).read_bytes() == (session / EMG_FILE).read_bytes()
    metadata = json.loads((replay / METADATA_FILE).read_text(encoding="utf-8"))
    assert metadata["source"] == "recorded" and metadata["sample_rate_hz"] == 2000.0


def test_plain_csv_and_dataset_files(tmp_path: Path) -> None:
    """A --save CSV gets its rate from the time column; dataset CSVs are not ours."""
    path = tmp_path / "plain.csv"
    rows = [f"{i / 1111.111},{i * 0.001},{-i * 0.001}" for i in range(10)]
    path.write_text("\n".join(["time_s,FDS_mV,XYZ_mV", *rows]), encoding="utf-8")
    recording = find_recording(path, applied_config.FOUR_SENSORS)
    assert recording is not None
    assert recording.sample_rate_hz == pytest.approx(1111.111)
    assert [(c.name, c.role, c.recorded_column) for c in recording.channels] == [
        ("FDS", "flexor", 1),
        ("XYZ", "other", 2),
    ]
    dataset = tmp_path / "10_raw.csv"
    dataset.write_text("0.1,0.2\n0.3,0.4\n", encoding="utf-8")
    assert find_recording(dataset) is None
    assert find_recording(tmp_path / "missing.csv") is None


def test_cli_rejects_a_broken_session(tmp_path: Path) -> None:
    """A session folder without emg.csv cannot be replayed (exit code 3)."""
    code = run_emg.main(["--source", "recorded", "--file", str(tmp_path), "--headless"])
    assert code == run_emg.EXIT_SOURCE
