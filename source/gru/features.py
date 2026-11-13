"""
Input features for the GRU, made from the filtered EMG.

Every 10 ms (GRU_FEATURE_RATE_HZ) one "frame" of features is made from the last
50 ms (GRU_FEATURE_WINDOW_S) of every channel:

  RMS  root mean square        how strong the EMG is (as the DSP envelope, but short)
  MAV  mean absolute value     also strength, less sensitive to single spikes

A frame is [RMS of each channel..., MAV of each channel...]. The frames are
then normalized with the rest level of each channel from the calibration:

  feature = log(1 + value / rest level)

so 0 means silence, about 0.7 means "as at rest", and a strong grip is 2-4. That
makes the numbers independent of the sensor placement and of the skin, and
the logarithm keeps a strong grip from producing huge inputs.

The extractor is causal (it only uses samples it has already seen) and gives
the same frames whatever the size of the blocks it is fed.
"""

import numpy as np


def frame_features(window: np.ndarray) -> np.ndarray:
    """RMS and MAV of one (n_channels, n) window, as one (2 * n_channels,) frame."""
    rms = np.sqrt(np.mean(window**2, axis=1))
    mav = np.mean(np.abs(window), axis=1)
    return np.concatenate([rms, mav])


def normalize(frames: np.ndarray, rest_level: np.ndarray) -> np.ndarray:
    """log(1 + value / rest level) of (n_frames, 2 * n_channels) frames."""
    rest = np.maximum(np.tile(rest_level, 2), 1e-6)  # RMS columns, then MAV columns
    return np.log1p(frames / rest)


class FeatureExtractor:
    """Makes frames from filtered EMG that arrives in blocks of any size."""

    def __init__(self, n_channels: int, sample_rate: float, feature_rate_hz: float,
                 feature_window_s: float) -> None:
        hop = sample_rate / feature_rate_hz  # samples between two frames
        if abs(hop - round(hop)) > 1e-6:
            raise ValueError(f"{sample_rate:g} Hz does not give a whole number of samples per "
                             f"frame at {feature_rate_hz:g} frames per second")
        self.n_channels = n_channels
        self.hop = round(hop)
        self.window = round(feature_window_s * sample_rate)
        self.buffer = np.zeros((n_channels, 0))  # samples not used up yet

    def reset(self) -> None:
        """Forget the buffered samples; the next frame needs a full window again."""
        self.buffer = np.zeros((self.n_channels, 0))

    def process(self, filtered: np.ndarray) -> np.ndarray:
        """
        Add a (n_channels, n) block of filtered EMG and return the new frames,
        (n_frames, 2 * n_channels). A small block may give no frame at all.
        """
        self.buffer = np.concatenate([self.buffer, filtered], axis=1)
        frames = []
        while self.buffer.shape[1] >= self.window:
            frames.append(frame_features(self.buffer[:, :self.window]))
            self.buffer = self.buffer[:, self.hop:]
        return np.array(frames).reshape(-1, 2 * self.n_channels)

    def frame_end_samples(self, n_frames: int) -> np.ndarray:
        """Sample number of the last sample of each of the first n_frames frames."""
        return np.arange(n_frames) * self.hop + self.window - 1
