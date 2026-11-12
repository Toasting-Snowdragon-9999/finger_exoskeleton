"""
Real-time EMG processing, one block of samples at a time.

Every filter keeps its state between blocks, so processing a signal in blocks
gives exactly the same result as processing it in one go. Per channel:

  raw -> band-pass 20-450 Hz -> notch 50 Hz             = filtered
  filtered -> moving RMS 0.2 s                         = slow envelope (grasp effort)
  |filtered| -> low-pass 10 Hz                         = fast envelope (tremor bursts)
  fast -> band-pass 4-6 Hz -> moving RMS 0.5 s         = tremor amplitude
  fast -> band-pass 1-9 Hz -> moving RMS 0.5 s         = fluctuation
  tremor / fluctuation                                 = tremor ratio (~1 rhythmic, ~0.5 not)
  tremor / slow envelope                               = tremor depth (independent of effort)
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

import settings


@dataclass
class EMGBlock:
    """A block of raw EMG from a source."""

    samples: np.ndarray  # (n_channels, n_samples) in mV
    first_index: int  # sample number of samples[:, 0], counted from the start
    receive_time: float  # time.monotonic() when the block arrived


class Filter:
    """A digital filter (second-order sections) that keeps its state between blocks."""

    def __init__(self, sos: np.ndarray, steady_start: bool) -> None:
        # steady_start: start as if the first sample had always been there. That avoids a big
        # jump at the start when the electrode has a DC offset.
        self.sos = sos
        self.steady_start = steady_start
        self.state: np.ndarray | None = None

    def process(self, block: np.ndarray) -> np.ndarray:
        """Filter one (n_channels, n) block."""
        if self.state is None:
            n_channels = block.shape[0]
            if self.steady_start:
                self.state = sosfilt_zi(self.sos)[:, None, :] * block[:, 0][None, :, None]
            else:
                self.state = np.zeros((self.sos.shape[0], n_channels, 2))
        output, self.state = sosfilt(self.sos, block, axis=1, zi=self.state)
        return output


class MovingRMS:
    """RMS of the last `window` samples. The first outputs use the samples seen so far."""

    def __init__(self, n_channels: int, window: int) -> None:
        self.n_channels = n_channels
        self.window = window
        self.history = np.zeros((n_channels, 0))  # squares of the last window - 1 samples

    def process(self, block: np.ndarray) -> np.ndarray:
        """The moving RMS at every sample of the block."""
        squares = np.concatenate([self.history, block**2], axis=1)
        sums = np.concatenate([np.zeros((self.n_channels, 1)), np.cumsum(squares, axis=1)], axis=1)
        ends = self.history.shape[1] + 1 + np.arange(block.shape[1])
        starts = np.maximum(ends - self.window, 0)
        mean_square = (sums[:, ends] - sums[:, starts]) / (ends - starts)
        self.history = squares[:, -(self.window - 1):] if self.window > 1 else squares[:, :0]
        return np.sqrt(np.maximum(mean_square, 0.0))


@dataclass
class ProcessedBlock:
    """All signals of one block, each (n_channels, n_samples)."""

    raw: np.ndarray
    filtered: np.ndarray
    slow: np.ndarray
    fast: np.ndarray
    tremor: np.ndarray
    ratio: np.ndarray
    depth: np.ndarray
    first_index: int
    receive_time: float


class SignalProcessor:
    """The processing chain described at the top of this file."""

    def __init__(self, n_channels: int, sample_rate: float) -> None:
        rate = sample_rate
        bandpass = butter(settings.BANDPASS_ORDER, list(settings.BANDPASS_HZ),
                          btype="bandpass", fs=rate, output="sos")
        b, a = iirnotch(settings.NOTCH_HZ, settings.NOTCH_QUALITY, fs=rate)
        lowpass = butter(2, settings.FAST_CUTOFF_HZ, btype="lowpass", fs=rate, output="sos")
        tremor = butter(2, list(settings.TREMOR_BAND_HZ), btype="bandpass", fs=rate, output="sos")
        rhythm = butter(2, list(settings.RHYTHM_BAND_HZ), btype="bandpass", fs=rate, output="sos")
        tremor_window = round(settings.TREMOR_WINDOW_S * rate)

        self.bandpass = Filter(bandpass, steady_start=True)
        self.notch = Filter(tf2sos(b, a), steady_start=True)
        self.slow_rms = MovingRMS(n_channels, round(settings.SLOW_WINDOW_S * rate))
        self.lowpass = Filter(lowpass, steady_start=False)
        self.tremor_band = Filter(tremor, steady_start=False)
        self.tremor_rms = MovingRMS(n_channels, tremor_window)
        self.rhythm_band = Filter(rhythm, steady_start=False)
        self.rhythm_rms = MovingRMS(n_channels, tremor_window)

    def process(self, block: EMGBlock) -> ProcessedBlock:
        """Process one block; the filter states carry over to the next block."""
        raw = np.asarray(block.samples, dtype=np.float64)
        filtered = self.notch.process(self.bandpass.process(raw))
        slow = self.slow_rms.process(filtered)
        fast = self.lowpass.process(np.abs(filtered))
        tremor = self.tremor_rms.process(self.tremor_band.process(fast))
        fluctuation = self.rhythm_rms.process(self.rhythm_band.process(fast))
        ratio = np.divide(tremor, fluctuation, out=np.zeros_like(tremor), where=fluctuation > 1e-12)
        depth = tremor / np.maximum(slow, settings.DEPTH_MIN_ENVELOPE_MV)
        return ProcessedBlock(raw, filtered, slow, fast, tremor, ratio, depth,
                              block.first_index, block.receive_time)
