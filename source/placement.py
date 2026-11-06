"""
Sensor placement comparison from recordings: how well does a channel see a grip?

Each placement is one recording (run_emg.py --out ...) with a rest period at the
start and then several grips. From the slow envelope (200 ms RMS, streaming.py)
of every channel:

  rest_uv      mean envelope in the rest window: the noise floor of that placement
  drift        standard deviation of the rest envelope / its mean (unstable skin contact)
  active_uv    95th percentile of the envelope inside the grips
  snr          active_uv / rest_uv: how far a grip stands out; the calibration needs >= 3
  grips        number of grips found (envelope above grip_factor x rest for min_grip_s)
  spike_ratio  median over the grips of peak / median envelope; high means short bursts
               on top of the grip, typically motion artifacts from the sensor moving
  repeatability  coefficient of variation of the grips' median levels (lower is better)
  crosstalk    for each other channel: its median level in this channel's grips / its rest;
               near 1 means it stays quiet, high means it picks up the same muscle
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ChannelReport:  # pylint: disable=too-many-instance-attributes
    """The numbers for one channel of one recording, levels in the unit of the envelope."""

    name: str
    rest: float
    drift: float
    active: float
    snr: float
    grips: int
    spike_ratio: float
    repeatability: float | None
    crosstalk: dict[str, float] = field(default_factory=dict)


def grip_spans(
    envelope: FloatArray,
    rest: float,
    sample_rate: float,
    grip_factor: float = 4.0,
    min_grip_s: float = 0.3,
    merge_gap_s: float = 0.3,
) -> list[tuple[int, int]]:
    """
    Sample ranges (start, end) of one channel's envelope above grip_factor x rest, with
    gaps shorter than merge_gap_s closed and spans shorter than min_grip_s dropped.
    """
    above = envelope > grip_factor * rest
    spans: list[list[int]] = []
    start: int | None = None
    for i, value in enumerate(np.append(above, False)):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if spans and start - spans[-1][1] <= merge_gap_s * sample_rate:
                spans[-1][1] = i
            else:
                spans.append([start, i])
            start = None
    return [(a, b) for a, b in spans if b - a >= min_grip_s * sample_rate]


def analyse(
    envelope: FloatArray,
    names: Sequence[str],
    sample_rate: float,
    rest_window_s: tuple[float, float] = (0.5, 3.0),
    grip_factor: float = 4.0,
) -> list[ChannelReport]:
    """One ChannelReport per channel of `envelope` (n_channels, n), see the module docstring."""
    start, end = (int(second * sample_rate) for second in rest_window_s)
    rest = np.mean(envelope[:, start:end], axis=1)
    if np.any(rest <= 0):
        raise ValueError("The rest window has no signal; check the rest window and the recording")
    all_spans = [
        grip_spans(envelope[channel], rest[channel], sample_rate, grip_factor)
        for channel in range(len(names))
    ]
    reports = []
    for channel, name in enumerate(names):
        spans = all_spans[channel]
        inside = [envelope[channel, a:b] for a, b in spans]
        medians = np.array([np.median(part) for part in inside])
        spikes = [np.max(part) / np.median(part) for part in inside]
        active = float(np.percentile(np.concatenate(inside), 95)) if inside else float("nan")
        crosstalk = {
            names[other]: float(
                np.median(np.concatenate([envelope[other, a:b] for a, b in spans])) / rest[other]
            )
            for other in range(len(names))
            if other != channel and spans
        }
        reports.append(
            ChannelReport(
                name=name,
                rest=float(rest[channel]),
                drift=float(np.std(envelope[channel, start:end]) / rest[channel]),
                active=active,
                snr=active / float(rest[channel]) if inside else float("nan"),
                grips=len(spans),
                spike_ratio=float(np.median(spikes)) if spikes else float("nan"),
                repeatability=(
                    float(np.std(medians) / np.mean(medians)) if len(spans) >= 2 else None
                ),
                crosstalk=crosstalk,
            )
        )
    return reports
