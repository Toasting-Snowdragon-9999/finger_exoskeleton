"""
Synthetic tremor laid over a real recording, for bench tests of tremor
detection and control on real EMG (noise, artifacts, calibration) with a
known tremor.

Within each tremor span the EMG is modulated by rhythmic bursts, as in the
synthetic source (sensor_mock.py): flexors and extensors burst in antiphase
(Parkinsonian tremor, design doc 7.5), and the modulation keeps the mean
power, so the slow envelope (grasp activation) stays about the same:

  emg * ((1 - depth) + depth * 2 * burst),   burst = (1 + sin(2 pi f t + phase)) / 2

Modulation only shows where the muscle is active (grasp + tremor). For rest
tremor, rest_mv adds bursts of band-limited (20-450 Hz) noise of that RMS.
Both fade in and out over ramp_s at the edges of each span.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TremorOverlay:
    """What to add: spans (start_s, end_s), frequency, modulation depth 0-1, rest bursts."""

    spans: tuple[tuple[float, float], ...]
    frequency_hz: float = 5.0
    depth: float = 0.8
    rest_mv: float = 0.0
    ramp_s: float = 0.5
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.spans:
            raise ValueError("At least one tremor span is required")
        for start, end in self.spans:
            if not 0 <= start < end:
                raise ValueError(f"Bad tremor span {start}-{end} s")
        if not 0 < self.depth <= 1:
            raise ValueError("depth must be in (0, 1]")
        if self.frequency_hz <= 0 or self.rest_mv < 0 or self.ramp_s < 0:
            raise ValueError("frequency_hz must be > 0, rest_mv and ramp_s >= 0")


def span_weight(times: FloatArray, overlay: TremorOverlay) -> FloatArray:
    """1 inside the spans, 0 outside, with linear ramps of ramp_s inside each span's edges."""
    weight = np.zeros_like(times)
    for start, end in overlay.spans:
        ramp = min(overlay.ramp_s, (end - start) / 2)
        if ramp > 0:
            rising = np.clip((times - start) / ramp, 0.0, 1.0)
            falling = np.clip((end - times) / ramp, 0.0, 1.0)
            inside = np.minimum(rising, falling)
        else:
            inside = ((times >= start) & (times < end)).astype(float)
        weight = np.maximum(weight, inside)
    return weight


def add_tremor(
    times: FloatArray,
    emg: FloatArray,
    roles: Sequence[str],
    sample_rate: float,
    overlay: TremorOverlay,
) -> FloatArray:
    """
    emg (n_channels, n) in mV with the tremor added; times (n,) in s.
    Extensors burst in antiphase with the other channels.
    """
    if emg.shape != (len(roles), times.size):
        raise ValueError("emg must have shape (len(roles), len(times))")
    phase = np.array([np.pi if role == "extensor" else 0.0 for role in roles])[:, None]
    burst = 0.5 * (1 + np.sin(2 * np.pi * overlay.frequency_hz * times[None, :] + phase))
    depth = overlay.depth * span_weight(times, overlay)[None, :]
    result: FloatArray = emg * ((1 - depth) + depth * 2 * burst)
    if overlay.rest_mv > 0:
        rng = np.random.default_rng(overlay.seed)
        high = min(450.0, 0.45 * sample_rate)
        sos = butter(4, [20.0, high], btype="bandpass", fs=sample_rate, output="sos")
        noise = sosfiltfilt(sos, rng.standard_normal(emg.shape), axis=1)
        noise /= np.sqrt(np.mean(noise**2, axis=1, keepdims=True))
        # 2 * burst has mean 1, so the bursts have rest_mv RMS on average
        result = result + overlay.rest_mv * (depth / overlay.depth) * 2 * burst * noise
    return result
