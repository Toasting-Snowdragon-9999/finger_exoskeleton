"""run_emg.py from start to end: replay, recording, the fake TCU and the motor."""

import json
from pathlib import Path

import pytest

import run_emg
import settings
from source.motor import connect_motor
from tests.fakes import FakeMotors

ROOT = Path(__file__).resolve().parent.parent
RECORDING = ROOT / "measurements" / "20261105-144738_trigno_to-lange-greb"


def test_replay_with_recording(tmp_path: Path) -> None:
    """A headless replay with --out writes a session folder that can be replayed again."""
    code = run_emg.main(["--source", "recorded", "--file", str(RECORDING), "--headless",
                         "--duration", "1.5", "--out", str(tmp_path), "--label", "test"])
    assert code == 0
    (folder,) = tmp_path.iterdir()
    metadata = json.loads((folder / "session.json").read_text())
    assert metadata["status"] == "complete" and metadata["samples_per_channel"] > 2000
    assert (folder / "emg.csv").read_text().startswith("time_s,FDS_mV,EDC_mV")


def test_motor_ends_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """With --motor the motor runs during the replay and is back at open at the end."""
    made: list[FakeMotors] = []

    def fake_motors(**kwargs) -> FakeMotors:
        made.append(FakeMotors(**kwargs))
        return made[0]

    monkeypatch.setattr(run_emg, "connect_motor",
                        lambda port: connect_motor(port, motors_class=fake_motors))
    code = run_emg.main(["--source", "recorded", "--file", str(RECORDING), "--headless",
                         "--duration", "1.0", "--motor"])
    assert code == 0
    assert made[0].sent and made[0].sent[-1] == settings.OPEN_POS


def test_fake_tcu() -> None:
    """--mock-tcu streams a recording through the real Trigno client."""
    code = run_emg.main(["--source", "trigno", "--mock-tcu", "--headless", "--duration", "1.5"])
    assert code == 0
