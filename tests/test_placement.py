"""Sensor placement comparison (source/placement.py, compare_placement.py)."""

import json
from pathlib import Path

import numpy as np
import pytest

import compare_placement
from source.placement import analyse, grip_spans
from source.session import EMG_FILE, METADATA_FILE

RATE = 100.0  # the analysis works on an envelope, any rate will do


def envelope(seconds: float = 14.0) -> tuple[np.ndarray, np.ndarray]:
    """Two channels: 2 uV at rest; channel 0 has grips of 40 and 60 uV (a 400 uV spike in the
    second); channel 1 only rises to 3 uV during the first grip."""
    times = np.arange(int(seconds * RATE)) / RATE
    env = np.full((2, times.size), 2.0)
    env[0, (times >= 4) & (times < 7)] = 40.0
    env[0, (times >= 9) & (times < 12)] = 60.0
    env[0, (times >= 10) & (times < 10.1)] = 400.0
    env[1, (times >= 4) & (times < 7)] = 3.0
    return times, env


def test_grip_spans_merge_short_gaps_and_drop_short_spans() -> None:
    """Gaps under 0.3 s join two bursts, a burst under 0.3 s is not a grip."""
    env = np.full(1000, 1.0)
    env[100:200] = 20.0
    env[215:300] = 20.0  # 0.15 s gap: one grip 100-300
    env[500:510] = 20.0  # 0.1 s: too short
    env[700:800] = 20.0
    assert grip_spans(env, 1.0, RATE) == [(100, 300), (700, 800)]


def test_analyse_reports_levels_spikes_and_crosstalk() -> None:
    """Rest, grip level, SNR, artifact spikes, repeatability and cross-talk of the example."""
    _, env = envelope()
    reports = analyse(env, ["FDS", "EDC"], RATE)
    flexor, extensor = reports[0], reports[1]
    assert flexor.rest == pytest.approx(2.0) and flexor.grips == 2
    assert flexor.active == pytest.approx(60.0, rel=0.05)  # the 95th percentile is in grip 2
    assert flexor.snr == pytest.approx(flexor.active / 2.0)
    # peak / median per grip: 40/40 = 1 and 400/60; the report takes the median of the two
    assert flexor.spike_ratio == pytest.approx((1.0 + 400 / 60) / 2, rel=0.01)
    assert flexor.repeatability == pytest.approx(np.std([40, 60]) / 50, rel=0.01)
    # the extensor is 3 uV in the first grip and 2 uV in the second: median 2.5 uV = 1.25 x rest
    assert flexor.crosstalk["EDC"] == pytest.approx(1.25)
    assert extensor.grips == 0 and np.isnan(extensor.snr)  # 1.5x rest is not a grip
    assert extensor.repeatability is None


def test_a_recording_without_rest_signal_is_rejected() -> None:
    """A flat zero rest window gives no baseline."""
    with pytest.raises(ValueError, match="rest window"):
        analyse(np.zeros((1, 1000)), ["FDS"], RATE)


def test_cli_prints_one_row_per_channel(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """compare_placement.py on a session whose flexor has two grips."""
    rate = 2000.0
    times = np.arange(int(14 * rate)) / rate
    level = np.full(times.size, 0.003)  # mV
    for start, end in ((4.0, 7.0), (9.0, 12.0)):
        level[(times >= start) & (times < end)] = 0.1
    noise = np.random.default_rng(1).standard_normal((2, times.size))
    emg = noise * np.array([level, np.full(times.size, 0.003)])
    session = tmp_path / "20261106-100000_trigno_plac-A"
    session.mkdir()
    rows = ["time_s,FDS_mV,EDC_mV"] + [f"{t},{a},{b}" for t, a, b in zip(times, *emg)]
    (session / EMG_FILE).write_text("\n".join(rows) + "\n", encoding="utf-8")
    channels = [{"name": "FDS", "role": "flexor", "trigno_slot": 15},
                {"name": "EDC", "role": "extensor", "trigno_slot": 14}]
    (session / METADATA_FILE).write_text(
        json.dumps({"emg_file": EMG_FILE, "sample_rate_hz": rate, "channels": channels}),
        encoding="utf-8",
    )

    assert compare_placement.main([str(session)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3 and lines[1].split()[1] == "FDS" and lines[2].split()[1] == "EDC"
    assert lines[1].split()[6] == "2"  # two grips on the flexor
    assert "no grip found" in lines[2]
    assert compare_placement.main([str(tmp_path / "missing")]) == 2
