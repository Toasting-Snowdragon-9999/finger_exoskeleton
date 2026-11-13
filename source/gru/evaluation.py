"""
Comparing the GRU and the signal processing (DSP) on recordings with labels.

Both methods are run over the same recording and scored at the same moments
(every 10 ms) against the same labels:

  intention   GRU only: how far the output is from the label (-1 ... +1)
  grasp       the closing part (0 ... 1) that drives the position controller:
              GRU max(intention, 0)  against  DSP grasp activation
  tremor      GRU probability >= GRU_TREMOR_ON  against  DSP depth >= TREMOR_ON
              and ratio >= RATIO_ON, the same conditions as in the live controller

Only moments with a known label count. A result is only worth something on
recordings (or people) that were not used for training, and on real data: with
synthetic tremor both methods are tested on the pattern they were built for.

No PyTorch is needed here; the model only has to offer step() (see model.py).
"""

from typing import Any

import numpy as np

import settings
from source.gru.features import FeatureExtractor
from source.gru.labels import LabelledRecording, recording_frames
from source.signal_processing import EMGBlock, SignalProcessor


def gru_outputs(model: Any, recording: LabelledRecording) -> dict:
    """The GRU's intention and tremor decision for every frame, with the labels."""
    features, intention, tremor = recording_frames(recording, model.config)
    predicted, probability, _ = model.step(features.astype(np.float32), None)
    return {"intention": predicted, "grasp": np.maximum(predicted, 0.0),
            "tremor": probability >= settings.GRU_TREMOR_ON,
            "intention_label": intention, "tremor_label": tremor}


def dsp_outputs(recording: LabelledRecording, config: Any) -> dict:
    """
    The signal processing's grasp activation and tremor decision at the same
    frames. The grasp needs the grip level of a calibration; without one it is
    not available (None).
    """
    n_channels = recording.emg.shape[0]
    processed = SignalProcessor(n_channels, recording.sample_rate).process(
        EMGBlock(recording.emg, 0, 0.0))
    extractor = FeatureExtractor(n_channels, recording.sample_rate, config.feature_rate_hz,
                                 config.feature_window_s)
    n_frames = 1 + (recording.emg.shape[1] - extractor.window) // extractor.hop
    ends = extractor.frame_end_samples(n_frames)

    depth, ratio = processed.depth[:, ends], processed.ratio[:, ends]
    strongest = np.argmax(depth, axis=0)  # the channel with the most tremor, as live
    frames = np.arange(n_frames)
    tremor = (depth[strongest, frames] >= settings.TREMOR_ON) \
        & (ratio[strongest, frames] >= settings.RATIO_ON)

    grasp = None
    if recording.grip_level is not None:
        span = (recording.grip_level - recording.rest_level)[:, None]
        activation = np.maximum((processed.slow[:, ends] - recording.rest_level[:, None]) / span, 0)
        flexors = [role == "flexor" for role in recording.roles]
        if not any(flexors):  # as live: without a flexor channel every channel counts
            flexors = [True] * n_channels
        grasp = np.minimum(activation[flexors].mean(axis=0), 1.0)  # the controller stops at 1
    return {"grasp": grasp, "tremor": tremor}


def intention_metrics(predicted: np.ndarray, label: np.ndarray) -> dict:
    """Mean absolute error, root mean square error and correlation where the label is known."""
    known = ~np.isnan(label)
    if not known.any():
        return {"n": 0}
    error = predicted[known] - label[known]
    result = {"n": int(known.sum()), "mae": float(np.mean(np.abs(error))),
              "rmse": float(np.sqrt(np.mean(error**2)))}
    if np.std(label[known]) > 0 and np.std(predicted[known]) > 0:
        result["correlation"] = float(np.corrcoef(predicted[known], label[known])[0, 1])
    return result


def tremor_metrics(detected: np.ndarray, label: np.ndarray, frame_rate: float) -> dict:
    """
    precision   of the frames called tremor, the share that really is tremor
    recall      of the real tremor frames, the share that was found
    false_alarms_per_minute   detections that start while the label says "no tremor"
    median_latency_s          from the start of a tremor episode to its first detection
    """
    known = ~np.isnan(label)
    truth = label == 1
    if not known.any():
        return {"n": 0}
    hits = int(np.sum(detected & truth))
    false = int(np.sum(detected & known & ~truth))
    missed = int(np.sum(~detected & truth))
    precision = hits / (hits + false) if hits + false else 0.0
    recall = hits / (hits + missed) if hits + missed else 0.0
    result = {"n": int(known.sum()), "tremor_share": float(truth[known].mean()),
              "precision": precision, "recall": recall,
              "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}

    starts = np.flatnonzero(detected[1:] & ~detected[:-1]) + 1  # frames where a detection begins
    calm_minutes = np.sum(known & ~truth) / frame_rate / 60
    if calm_minutes > 0:
        result["false_alarms_per_minute"] = float(np.sum(label[starts] == 0) / calm_minutes)
    latencies = []
    for onset in np.flatnonzero(truth[1:] & ~truth[:-1]) + 1:  # frames where tremor begins
        episode_end = onset + int(np.argmax(~truth[onset:])) if not truth[onset:].all() \
            else len(truth)
        found = np.flatnonzero(detected[onset:episode_end])
        if len(found) > 0:
            latencies.append(found[0] / frame_rate)
    if latencies:
        result["median_latency_s"] = float(np.median(latencies))
    return result


def evaluate(model: Any, recordings: list[LabelledRecording]) -> dict:
    """Metrics of the GRU and of the signal processing over all the recordings together."""
    rate = model.config.feature_rate_hz
    warm_up = round(settings.GRU_WARMUP_S * rate)  # not scored, as the estimates are not valid
    joined: dict[str, list] = {key: [] for key in (
        "intention", "gru_grasp", "gru_tremor", "dsp_grasp", "dsp_tremor",
        "intention_label", "tremor_label")}
    dsp_has_grasp = True
    for recording in recordings:
        gru = gru_outputs(model, recording)
        dsp = dsp_outputs(recording, model.config)
        n = len(gru["intention"])
        if dsp["grasp"] is None:
            dsp_has_grasp = False
        dsp_grasp = np.zeros(n) if dsp["grasp"] is None else dsp["grasp"]
        parts = {"intention": gru["intention"], "gru_grasp": gru["grasp"],
                 "gru_tremor": gru["tremor"], "dsp_grasp": dsp_grasp, "dsp_tremor": dsp["tremor"],
                 "intention_label": gru["intention_label"], "tremor_label": gru["tremor_label"]}
        for key, values in parts.items():
            values = np.array(values[:n], dtype=float)
            if key.endswith("_label"):
                values[:warm_up] = np.nan  # unknown label = not scored
            # One extra "unknown" frame keeps the recordings apart when they are joined
            joined[key].append(np.append(values, np.nan if key.endswith("_label") else 0.0))
    data = {key: np.concatenate(parts) for key, parts in joined.items()}

    grasp_label = np.maximum(data["intention_label"], 0.0)
    result = {
        "recordings": [recording.name for recording in recordings],
        "synthetic_data": any(recording.synthetic for recording in recordings),
        "gru": {"intention": intention_metrics(data["intention"], data["intention_label"]),
                "grasp": intention_metrics(data["gru_grasp"], grasp_label),
                "tremor": tremor_metrics(data["gru_tremor"] > 0, data["tremor_label"], rate)},
        "dsp": {"grasp": intention_metrics(data["dsp_grasp"], grasp_label)
                if dsp_has_grasp else {"n": 0},
                "tremor": tremor_metrics(data["dsp_tremor"] > 0, data["tremor_label"], rate)},
    }
    return result


def format_results(result: dict) -> str:
    """The metrics as a small text table."""
    lines = []
    if result["synthetic_data"]:
        lines.append("SYNTHETIC DATA: a software test. This says nothing about real tremor.")
    for title, method, key in (("GRU intention (-1..+1)", "gru", "intention"),
                               ("GRU grasp (0..1)", "gru", "grasp"),
                               ("DSP grasp (0..1)", "dsp", "grasp"),
                               ("GRU tremor", "gru", "tremor"),
                               ("DSP tremor", "dsp", "tremor")):
        metrics = result[method][key]
        if metrics["n"] == 0:
            lines.append(f"{title:24s} no labels (or no calibration grip level) to compare with")
            continue
        values = "  ".join(f"{name} {value:.3f}" for name, value in metrics.items() if name != "n")
        lines.append(f"{title:24s} {values}  ({metrics['n']} frames)")
    return "\n".join(lines)
