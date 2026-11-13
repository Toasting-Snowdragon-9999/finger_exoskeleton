"""
Recordings with labels, for training and testing the GRU.

A label says, for every EMG sample, what the right answer is:

  intention   -1 = open actively, 0 = rest, +1 = close fully      NaN = not known
  tremor      1 = tremor, 0 = no tremor                           NaN = not known

"Not known" is normal: a recording may have tremor labels but no intention
labels, or the other way round. Unknown samples are simply not scored.

Where recordings come from:
  load_session(folder)   one of our own session folders, with a labels.csv next to
                         emg.csv (written by run_emg.py --cues, or by hand)
  load_npz(file)         any other dataset, converted to one .npz file with the keys
                         emg (n_channels, n) in mV, sample_rate, roles, and optionally
                         intention, tremor, group and rest_level
"""

import csv
import json
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

import settings
from source.cues import LABELS_FILE
from source.gru.config import GRUConfig
from source.gru.features import FeatureExtractor, normalize
from source.recording import Recording
from source.signal_processing import EMGBlock, SignalProcessor


@dataclass
class LabelledRecording:
    """EMG with a label for every sample."""

    name: str
    group: str  # recordings of the same group (person, session) are never split up
    emg: np.ndarray  # (n_channels, n) in mV
    sample_rate: float
    roles: list[str]  # the muscle role of every channel, e.g. ["flexor", "extensor"]
    rest_level: np.ndarray  # (n_channels,) rest level in mV, for normalizing the features
    intention: np.ndarray  # (n,) NaN = unknown
    tremor: np.ndarray  # (n,) NaN = unknown
    synthetic: bool = False  # True: made up, a software test only
    # Grip level of the calibration per channel in mV; only the DSP baseline needs it
    grip_level: np.ndarray | None = None

    def select_roles(self, roles: list[str]) -> "LabelledRecording":
        """The same recording with only the channels of these roles, in this order."""
        for role in roles:
            if role not in self.roles:
                raise ValueError(f"{self.name} has no {role} channel (it has {self.roles})")
        rows = [self.roles.index(role) for role in roles]
        grip_level = None if self.grip_level is None else self.grip_level[rows]
        return replace(self, emg=self.emg[rows], roles=list(roles),
                       rest_level=self.rest_level[rows], grip_level=grip_level)


def read_labels(path: Path, times: np.ndarray, margin_s: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-sample intention and tremor from a labels.csv (start_s,end_s,intention,tremor;
    an empty value = unknown). For margin_s after every change of a value that
    label is unknown: a person needs a moment to react to a new cue.
    """
    labels = {"intention": np.full(len(times), np.nan), "tremor": np.full(len(times), np.nan)}
    previous = {"intention": "", "tremor": ""}
    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            start, end = float(row["start_s"]), float(row["end_s"])
            inside = (times >= start) & (times < end)
            for key, values in labels.items():
                text = row[key].strip()
                values[inside] = float(text) if text else np.nan
                if text != previous[key]:
                    values[inside & (times < start + margin_s)] = np.nan
                previous[key] = text
    return labels["intention"], labels["tremor"]


def rest_level_of(emg: np.ndarray, sample_rate: float, percentile: float | None = None,
                  start_s: float = 0.0) -> np.ndarray:
    """
    The rest level of every channel: the mean slow envelope in the calibration
    rest window (as the live calibration; start_s = the recording's start delay),
    or with `percentile` that percentile of the whole recording (for datasets
    without a calibration at the start).
    """
    slow = SignalProcessor(emg.shape[0], sample_rate).process(EMGBlock(emg, 0, 0.0)).slow
    if percentile is not None:
        return np.percentile(slow, percentile, axis=1)
    times = np.arange(emg.shape[1]) / sample_rate - start_s
    rest = (times >= settings.REST_WINDOW_S[0]) & (times < settings.REST_WINDOW_S[1])
    return slow[:, rest].mean(axis=1)


def load_session(folder: Path, calibration_file: Path | None = None,
                 margin_s: float = settings.GRU_LABEL_MARGIN_S) -> LabelledRecording:
    """
    One of our session folders. The rest level is the one that was used live:
    from calibration_file if given, else from the calibration stored in
    session.json, else from the rest window at the start of the recording.
    """
    recording = Recording(folder)
    names = [channel.name for channel in recording.channels]
    calibration = recording.metadata.get("calibration")
    if calibration_file is not None:
        calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
    grip_level = None
    if calibration is not None:
        rest = dict(zip(calibration["channel_names"], calibration["rest_slow"]))
        grip = dict(zip(calibration["channel_names"], calibration["reference_slow"]))
        rest_level = np.array([rest[name] for name in names])
        grip_level = np.array([grip[name] for name in names])
    else:
        rest_level = rest_level_of(recording.emg, recording.sample_rate,
                                   start_s=float(recording.metadata.get("start_delay_s", 0.0)))

    n = recording.emg.shape[1]
    intention, tremor = np.full(n, np.nan), np.full(n, np.nan)
    labels_path = recording.emg_path.with_name(LABELS_FILE)
    if labels_path.is_file():
        intention, tremor = read_labels(labels_path, recording.times, margin_s)
    # A copy with synthetic tremor belongs to the same group as the recording it was made from
    overlay = recording.metadata.get("tremor_overlay")
    folder_name = recording.emg_path.parent.name
    return LabelledRecording(
        name=folder_name, group=overlay["based_on"] if overlay else folder_name,
        emg=recording.emg, sample_rate=recording.sample_rate,
        roles=[channel.role for channel in recording.channels], rest_level=rest_level,
        intention=intention, tremor=tremor, synthetic=overlay is not None,
        grip_level=grip_level)


def resample(recording: LabelledRecording, sample_rate: float) -> LabelledRecording:
    """The recording at another sample rate (labels: the nearest earlier sample)."""
    ratio = Fraction(sample_rate / recording.sample_rate).limit_denominator(1000)
    emg = resample_poly(recording.emg, ratio.numerator, ratio.denominator, axis=1)
    source = np.minimum((np.arange(emg.shape[1]) / ratio).astype(int), len(recording.tremor) - 1)
    return replace(recording, emg=emg, sample_rate=sample_rate,
                   intention=recording.intention[source], tremor=recording.tremor[source])


def load_npz(path: Path, sample_rate: float = 2000.0) -> LabelledRecording:
    """A converted dataset file (see the top of this file), resampled to sample_rate."""
    data = np.load(path)
    emg = np.asarray(data["emg"], dtype=float)
    n = emg.shape[1]
    rate = float(data["sample_rate"])
    if "rest_level" in data:
        rest_level = np.asarray(data["rest_level"], dtype=float)
    else:  # no calibration in a public dataset: take a low percentile as "rest"
        rest_level = rest_level_of(emg, rate, percentile=10.0)
    recording = LabelledRecording(
        name=path.stem, group=str(data["group"]) if "group" in data else path.stem,
        emg=emg, sample_rate=rate, roles=[str(role) for role in data["roles"]],
        rest_level=rest_level,
        intention=np.asarray(data["intention"], dtype=float) if "intention" in data
        else np.full(n, np.nan),
        tremor=np.asarray(data["tremor"], dtype=float) if "tremor" in data else np.full(n, np.nan))
    if abs(rate - sample_rate) > 1e-6:
        recording = resample(recording, sample_rate)
    return recording


def recording_frames(recording: LabelledRecording, config: GRUConfig
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Features (n_frames, n_features) and the intention and tremor label of every
    frame (the label at the frame's last sample). Uses the live filter and
    feature code, so training sees exactly what the model sees later.
    """
    recording = recording.select_roles(config.channel_roles)
    if abs(recording.sample_rate - config.sample_rate_hz) > 1e-6:
        raise ValueError(f"{recording.name} is at {recording.sample_rate:g} Hz; "
                         f"the model needs {config.sample_rate_hz:g} Hz")
    n_channels = recording.emg.shape[0]
    processor = SignalProcessor(n_channels, recording.sample_rate)
    filtered = processor.process(EMGBlock(recording.emg, 0, 0.0)).filtered
    extractor = FeatureExtractor(n_channels, recording.sample_rate, config.feature_rate_hz,
                                 config.feature_window_s)
    frames = extractor.process(filtered)
    ends = extractor.frame_end_samples(len(frames))
    intention = recording.intention[ends]
    if "extensor" not in config.channel_roles:
        # Opening is done by the extensor. With a flexor sensor only it cannot be told from
        # rest, so "open" (-1) is trained as "no grasp" (0).
        intention = np.maximum(intention, 0.0)
    return normalize(frames, recording.rest_level), intention, recording.tremor[ends]
