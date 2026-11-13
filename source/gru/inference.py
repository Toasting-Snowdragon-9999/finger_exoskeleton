"""
Real-time use of a trained GRU model: the interface the control system asks.

  estimator = GRUEstimator(model, channels, sample_rate)
  estimate = estimator.update(processed_block, rest_level)     # for every EMG block
  estimate.intention            -1 (open) ... 0 (rest) ... +1 (close)
  estimate.tremor_probability   0 ... 1
  estimate.valid                False: do not use the two numbers (estimate.reason says why)

The GRU has a memory (its "hidden state") that is kept between blocks, so it
can see rhythm and trends over time. reset() empties that memory; it is called
when the signal was interrupted or the calibration starts over. After a reset
the estimates are not valid for GRU_WARMUP_S, until the memory has filled again.

The estimator never touches the motor. It only gives numbers; the pipeline, the
stiffness controller and the motor code decide what to do with them.

This file does not need PyTorch: the model object only has to offer
`config` and `step(frames, hidden)` (see IntentionTremorGRU in model.py).
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import settings
from settings import Channel
from source.gru.config import filter_settings
from source.gru.features import FeatureExtractor, normalize
from source.signal_processing import ProcessedBlock


@dataclass
class GRUEstimate:
    """What the GRU says at the end of one EMG block."""

    intention: float  # -1 = open, 0 = rest, +1 = close
    tremor_probability: float  # 0-1
    valid: bool  # False: the numbers must not be used
    reason: str  # why it is not valid ("" when valid)
    receive_time: float  # when the EMG block arrived (time.monotonic())


class GRUEstimator:
    """Feeds EMG blocks to a GRU model and keeps its memory between blocks."""

    def __init__(self, model: Any, channels: list[Channel], sample_rate: float) -> None:
        self.model = model
        self.config = model.config
        self.rows: list[int] = []  # row of each model input channel in the EMG blocks
        self.problem = self.find_problem(channels, sample_rate)  # None when the model fits
        self.hidden: Any = None  # the GRU's memory; None = empty
        self.frames_since_reset = 0
        self.last: tuple[float, float] | None = None  # newest (intention, tremor probability)
        self.broken = False  # the model gave NaN or infinity
        if self.problem is None:
            self.extractor = FeatureExtractor(len(self.rows), sample_rate,
                                              self.config.feature_rate_hz,
                                              self.config.feature_window_s)

    def find_problem(self, channels: list[Channel], sample_rate: float) -> str | None:
        """Why this model cannot be used with these sensors and settings, or None."""
        roles = [channel.role for channel in channels]
        for role in self.config.channel_roles:
            if role not in roles:
                return f"the model needs a {role} sensor, the sensors are {roles}"
            self.rows.append(roles.index(role))
        if abs(sample_rate - self.config.sample_rate_hz) > 0.5:
            return (f"the model is for {self.config.sample_rate_hz:g} Hz EMG, "
                    f"the source gives {sample_rate:g} Hz")
        if self.config.filters != filter_settings():
            return "the filter settings changed since the model was trained"
        return None

    def reset(self, forget_broken: bool = False) -> None:
        """Empty the GRU's memory (signal interrupted, or a new calibration)."""
        self.hidden = None
        self.frames_since_reset = 0
        self.last = None
        if forget_broken:
            self.broken = False
        if self.problem is None:
            self.extractor.reset()

    def update(self, block: ProcessedBlock, rest_level: np.ndarray) -> GRUEstimate:
        """Add one processed EMG block; returns the newest estimate."""
        if self.problem is not None:
            return GRUEstimate(0.0, 0.0, False, self.problem, block.receive_time)
        if self.broken:
            reason = "the model gave a value that is not a number; calibrate again to retry"
            return GRUEstimate(0.0, 0.0, False, reason, block.receive_time)

        frames = self.extractor.process(block.filtered[self.rows])
        if len(frames) > 0:  # a small block may not complete a new frame
            features = normalize(frames, rest_level[self.rows])
            intention, probability, self.hidden = self.model.step(features, self.hidden)
            self.frames_since_reset += len(frames)
            self.last = (float(intention[-1]), float(probability[-1]))
            if not (math.isfinite(self.last[0]) and math.isfinite(self.last[1])):
                self.broken = True
                return self.update(block, rest_level)

        if self.last is None:
            return GRUEstimate(0.0, 0.0, False, "warming up", block.receive_time)
        warm = self.frames_since_reset >= settings.GRU_WARMUP_S * self.config.feature_rate_hz
        return GRUEstimate(self.last[0], self.last[1], warm, "" if warm else "warming up",
                           block.receive_time)


def load_model(path: Path) -> tuple[Any, str]:
    """
    (model, message): the GRU model in `path`, or (None, why not). PyTorch is only
    imported here, so everything else runs without it.
    """
    if not path.is_file():
        return None, f"no GRU model file {path} (train one with train_gru.py)"
    try:
        from source.gru.model import load_checkpoint  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None, "PyTorch is not installed (pip install -r requirements-ml.txt)"
    model = load_checkpoint(path)
    kind = "SYNTHETIC data: software test only" if model.config.data_kind == "synthetic" \
        else model.config.trained_on
    return model, f"GRU model {path} loaded ({kind})"
