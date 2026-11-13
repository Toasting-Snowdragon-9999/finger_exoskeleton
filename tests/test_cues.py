"""The cue protocols in protocols/ and source/cues.py."""

from pathlib import Path

import numpy as np
import pytest

from source.cues import CueProtocol
from source.gru.labels import read_labels

PROTOCOLS = sorted((Path(__file__).resolve().parent.parent / "protocols").glob("greb_*.csv"))


def test_there_are_three_protocols_with_different_timetables() -> None:
    """Three files whose cue times differ, so the timetable cannot be learned by heart."""
    assert [path.name for path in PROTOCOLS] == ["greb_1.csv", "greb_2.csv", "greb_3.csv"]
    timetables = {tuple(row["start_s"] for row in CueProtocol(path).rows) for path in PROTOCOLS}
    assert len(timetables) == 3


@pytest.mark.parametrize("path", PROTOCOLS, ids=lambda path: path.name)
def test_protocol_is_complete(path: Path, tmp_path: Path) -> None:
    """Calibration first (no label), then rest (0) and grips (1) without gaps; tremor 0."""
    protocol = CueProtocol(path)
    rows = protocol.rows
    assert [row["end_s"] for row in rows[:-1]] == [row["start_s"] for row in rows[1:]]
    assert rows[0]["start_s"] == "0" and float(rows[1]["end_s"]) == 10.0  # the calibration
    assert all(row["intention"] == "" for row in rows[:2])
    assert {row["intention"] for row in rows[2:]} == {"0", "1"}
    assert all(row["tremor"] == "0" for row in rows[2:])
    assert sum(row["intention"] == "1" for row in rows) == 6
    assert sum("KLEM HÅRDT" in row["cue"] for row in rows) == 3

    assert protocol.cue_at(1.0).startswith("KALIBRERING")
    assert protocol.cue_at(float(rows[-1]["end_s"]) + 1) == "FÆRDIG"

    # What --cues writes can be read back as labels: about half rest, half grip
    protocol.write_labels(tmp_path)
    times = np.arange(0, float(rows[-1]["end_s"]), 0.01)
    intention, tremor = read_labels(tmp_path / "labels.csv", times, margin_s=0.5)
    known = ~np.isnan(intention)
    assert np.all(np.isnan(intention[times < 10])) and 0.6 < known.mean() < 0.8
    assert 0.35 < np.mean(intention[known] == 1) < 0.6 and np.nanmax(tremor) == 0
