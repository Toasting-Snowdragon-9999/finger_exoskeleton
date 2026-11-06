"""Synthetic tremor laid over a recording (source/tremor_overlay.py, add_tremor.py)."""

import json
from pathlib import Path

import numpy as np
import pytest

import add_tremor
from source.session import EMG_FILE, METADATA_FILE, find_recording
from source.tremor_overlay import TremorOverlay, add_tremor as overlay_tremor, span_weight

RATE = 2000.0


def noise(n_channels: int, seconds: float, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Times and white-noise 'EMG' of RMS 0.1 mV."""
    times = np.arange(int(seconds * RATE)) / RATE
    return times, 0.1 * np.random.default_rng(seed).standard_normal((n_channels, times.size))


def test_weight_is_one_inside_and_ramps_at_the_edges() -> None:
    """Zero outside, one in the middle, halfway half a ramp in."""
    times = np.array([0.5, 1.0, 1.25, 3.0, 4.75, 5.0, 6.0])
    weight = span_weight(times, TremorOverlay(((1.0, 5.0),), ramp_s=0.5))
    assert weight.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0])


def test_tremor_modulates_inside_the_span_only_and_keeps_power() -> None:
    """Outside the span the EMG is unchanged; inside, the mean power stays and the
    5 Hz bursts show up with flexor and extensor in antiphase."""
    times, emg = noise(2, 10.0)
    overlay = TremorOverlay(((2.0, 8.0),), frequency_hz=5.0, depth=1.0, ramp_s=0.0)
    out = overlay_tremor(times, emg, ["flexor", "extensor"], RATE, overlay)
    outside = (times < 2.0) | (times >= 8.0)
    assert np.array_equal(out[:, outside], emg[:, outside])
    inside = (times >= 2.0) & (times < 8.0)
    power_ratio = np.mean(out[:, inside] ** 2) / np.mean(emg[:, inside] ** 2)
    assert power_ratio == pytest.approx(1.5, rel=0.1)  # E[(2 burst)^2] = 1.5 at depth 1
    # Envelope per 20 ms: flexor and extensor bursts are in antiphase
    env = np.abs(out[:, inside]).reshape(2, -1, 40).mean(axis=2)
    assert np.corrcoef(env[0], env[1])[0, 1] < -0.5


def test_rest_bursts_add_tremor_where_the_muscle_is_quiet() -> None:
    """With rest_mv, a quiet channel gets bursts of about that RMS inside the span."""
    times = np.arange(int(6 * RATE)) / RATE
    quiet = np.zeros((1, times.size))
    overlay = TremorOverlay(((1.0, 5.0),), rest_mv=0.02, ramp_s=0.0)
    out = overlay_tremor(times, quiet, ["flexor"], RATE, overlay)
    inside = (times >= 1.0) & (times < 5.0)
    assert np.sqrt(np.mean(out[0, inside] ** 2)) == pytest.approx(0.02 * np.sqrt(1.5), rel=0.15)
    assert not np.any(out[0, ~inside])


def test_bad_overlays_are_rejected() -> None:
    """No spans, reversed spans and depths outside (0, 1] raise."""
    for bad in ({"spans": ()}, {"spans": ((5.0, 2.0),)}, {"spans": ((1.0, 2.0),), "depth": 0.0}):
        with pytest.raises(ValueError):
            TremorOverlay(**bad)


def test_cli_writes_a_replayable_session(tmp_path: Path) -> None:
    """add_tremor.py makes a new session folder that find_recording reads back."""
    session = tmp_path / "20261105-144436_trigno_to-greb"
    session.mkdir()
    times, emg = noise(2, 4.0)
    rows = ["time_s,FDS_mV,EDC_mV"] + [f"{t},{a},{b}" for t, a, b in zip(times, *emg)]
    (session / EMG_FILE).write_text("\n".join(rows) + "\n", encoding="utf-8")
    metadata = {"label": "to-greb", "note": "", "emg_file": EMG_FILE, "sample_rate_hz": RATE,
                "channels": [{"name": "FDS", "role": "flexor", "trigno_slot": 15},
                             {"name": "EDC", "role": "extensor", "trigno_slot": 14}]}
    (session / METADATA_FILE).write_text(json.dumps(metadata), encoding="utf-8")

    assert add_tremor.main([str(session), "--span", "1-3"]) == 0
    out = tmp_path / f"{session.name}_tremor"
    recording = find_recording(out)
    assert recording is not None and recording.sample_rate_hz == RATE
    assert [c.role for c in recording.channels] == ["flexor", "extensor"]
    new_times, new_emg = recording.load()
    assert np.allclose(new_times, times) and new_emg.shape == emg.shape
    saved = json.loads((out / METADATA_FILE).read_text(encoding="utf-8"))
    assert saved["label"] == "to-greb+tremor"
    assert saved["tremor_overlay"]["spans_s"] == [[1.0, 3.0]]
    assert add_tremor.main([str(session), "--span", "3-1"]) == 2
