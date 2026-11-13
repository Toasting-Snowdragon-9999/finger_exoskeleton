"""
Calibration and muscle activation.

The user rests during REST_WINDOW_S and holds a firm grip during GRIP_WINDOW_S
(settings.py). Per channel that gives a rest level and a grip level of the
envelopes, and activation is then measured on that scale:

  activation = (envelope - rest) / (grip - rest)      0 = rest, 1 = the calibration grip

The grasp activation is the mean activation of the flexor channels. A
calibration can be saved to a JSON file and loaded by later runs.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

import settings
from source.signal_processing import ProcessedBlock


@dataclass
class Calibration:
    """Rest and grip levels per channel in mV, of the slow and of the fast envelope."""

    channel_names: list[str]
    rest_slow: np.ndarray
    reference_slow: np.ndarray  # the grip level
    rest_fast: np.ndarray
    reference_fast: np.ndarray

    def problem(self) -> str | None:
        """Why this calibration cannot be used, or None if it is fine."""
        for name, rest, grip in zip(self.channel_names, self.rest_slow, self.reference_slow):
            if rest <= 0 or grip < settings.MIN_GRIP_RATIO * rest:
                return (f"Channel {name}: reference {grip:.4f} mV is not at least "
                        f"{settings.MIN_GRIP_RATIO:g}x the rest level {rest:.4f} mV. Repeat the "
                        f"calibration with a clear reference contraction.")
        return None

    def save(self, path: Path) -> None:
        """Write the calibration and the signal processing settings it was made with."""
        data = {
            "channel_names": list(self.channel_names),
            "rest_slow": self.rest_slow.tolist(),
            "reference_slow": self.reference_slow.tolist(),
            "rest_fast": self.rest_fast.tolist(),
            "reference_fast": self.reference_fast.tolist(),
            "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "processing": settings.processing_settings(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_calibration(path: Path, channel_names: list[str]) -> tuple[Calibration | None, str]:
    """
    The saved calibration in `path` and a message saying what happened. Returns
    (None, message) if the file does not fit this run: other channels, other
    signal processing settings, or levels that fail the grip check.
    """
    if not path.is_file():
        return None, ""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["channel_names"] != list(channel_names):
        return None, (f"Saved calibration ignored: it is for channels {data['channel_names']}, "
                      f"not {list(channel_names)}")
    saved_processing = data.get("processing", {})
    for key, value in settings.processing_settings().items():
        if saved_processing.get(key) != value:
            return None, "Saved calibration ignored: the signal processing settings changed"
    calibration = Calibration(
        channel_names=list(data["channel_names"]),
        rest_slow=np.array(data["rest_slow"], dtype=float),
        reference_slow=np.array(data["reference_slow"], dtype=float),
        rest_fast=np.array(data["rest_fast"], dtype=float),
        reference_fast=np.array(data["reference_fast"], dtype=float),
    )
    problem = calibration.problem()
    if problem is not None:
        return None, f"Saved calibration ignored: {problem}"
    saved_at = data.get("saved_at", "unknown date")
    return calibration, f"Saved calibration loaded from {path} ({saved_at})"


class Calibrator:
    """Collects the envelopes during the rest and grip windows and makes a Calibration."""

    def __init__(self, channel_names: list[str], sample_rate: float) -> None:
        self.channel_names = list(channel_names)
        self.sample_rate = sample_rate
        self.restart()

    def restart(self) -> None:
        """Start a new calibration; time counts from the next block."""
        self.result: Calibration | None = None
        self.error: str | None = None
        self.start_index: int | None = None
        self.time = 0.0  # seconds since the calibration started
        self.rest_slow: list[np.ndarray] = []
        self.rest_fast: list[np.ndarray] = []
        self.grip_slow: list[np.ndarray] = []
        self.grip_fast: list[np.ndarray] = []

    @property
    def phase(self) -> str:
        """What the user should be doing now."""
        if self.result is not None:
            return "done"
        if self.error is not None:
            return "failed"
        if self.time < settings.REST_WINDOW_S[1]:
            return "rest (baseline)"
        if self.time < settings.GRIP_WINDOW_S[0]:
            return "prepare reference contraction"
        return "reference contraction"

    @property
    def seconds_left(self) -> float:
        """Seconds until the current phase ends."""
        if self.result is not None or self.error is not None:
            return 0.0
        ends = (settings.REST_WINDOW_S[1], settings.GRIP_WINDOW_S[0], settings.GRIP_WINDOW_S[1])
        for end in ends:
            if self.time < end:
                return end - self.time
        return 0.0

    def update(self, block: ProcessedBlock) -> None:
        """Feed one processed block. The calibration finishes after the grip window."""
        if self.result is not None or self.error is not None:
            return
        if self.start_index is None:
            self.start_index = block.first_index
        n = block.slow.shape[1]
        times = (block.first_index - self.start_index + np.arange(n)) / self.sample_rate
        self.time = float(times[-1])

        rest = (times >= settings.REST_WINDOW_S[0]) & (times < settings.REST_WINDOW_S[1])
        if rest.any():
            self.rest_slow.append(block.slow[:, rest])
            self.rest_fast.append(block.fast[:, rest])
        grip = (times >= settings.GRIP_WINDOW_S[0]) & (times < settings.GRIP_WINDOW_S[1])
        if grip.any():
            self.grip_slow.append(block.slow[:, grip])
            self.grip_fast.append(block.fast[:, grip])

        if self.time >= settings.GRIP_WINDOW_S[1]:
            self.finish()

    def finish(self) -> None:
        """Compute the levels and check them."""
        if not self.rest_slow or not self.grip_slow:
            self.error = "No samples fell inside the calibration windows"
            return
        percentile = settings.GRIP_PERCENTILE
        rest_slow = np.concatenate(self.rest_slow, axis=1)
        rest_fast = np.concatenate(self.rest_fast, axis=1)
        grip_slow = np.concatenate(self.grip_slow, axis=1)
        grip_fast = np.concatenate(self.grip_fast, axis=1)
        calibration = Calibration(
            channel_names=self.channel_names,
            rest_slow=np.mean(rest_slow, axis=1),
            reference_slow=np.percentile(grip_slow, percentile, axis=1),
            rest_fast=np.mean(rest_fast, axis=1),
            reference_fast=np.percentile(grip_fast, percentile, axis=1),
        )
        self.error = calibration.problem()
        if self.error is None:
            self.result = calibration


@dataclass
class Activation:
    """The muscle activation at the end of one block."""

    grasp: float  # mean activation of the flexor channels: 0 = rest, 1 = calibration grip
    tremor_depth: float  # of the channel with the most tremor
    tremor_ratio: float  # of that same channel
    receive_time: float  # when the block arrived (time.monotonic())
    # Only set by the GRU method (--method gru): the network's tremor probability, 0-1.
    # None means the signal processing (DSP) decides tremor from depth and ratio.
    tremor_probability: float | None = None


def compute_activation(block: ProcessedBlock, calibration: Calibration,
                       flexors: np.ndarray) -> Activation:
    """Activation at the last sample of the block. flexors: True for every flexor channel."""
    span = calibration.reference_slow - calibration.rest_slow
    activation = np.maximum((block.slow[:, -1] - calibration.rest_slow) / span, 0.0)
    depth = block.depth[:, -1]
    strongest = int(np.argmax(depth))
    return Activation(
        grasp=float(np.mean(activation[flexors])),
        tremor_depth=float(depth[strongest]),
        tremor_ratio=float(block.ratio[strongest, -1]),
        receive_time=block.receive_time,
    )
