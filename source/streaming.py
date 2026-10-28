"""
Real-time (streaming) EMG processing for many channels, block by block.

Every stage is causal and keeps its state between blocks, so processing a
signal in blocks gives exactly the same result as processing it in one go.
The offline, zero-phase tools in process_signal.py are not used here because
they need future samples.

Pipeline per channel (design doc 7.4):
  raw -> band-pass (20-450 Hz) -> optional notch (50 Hz) = filtered
  filtered -> moving RMS (200 ms)                          = slow envelope (grasp intent)
  |filtered| -> low-pass (10 Hz)                           = fast envelope (tremor bursts)
  fast envelope -> band-pass (4-6 Hz) -> moving RMS (0.5 s) = tremor amplitude
  fast envelope -> band-pass (1-9 Hz) -> moving RMS (0.5 s) = envelope fluctuation
  tremor amplitude / envelope fluctuation                   = tremor ratio (rhythmicity)
  tremor amplitude / slow envelope                          = tremor depth

The tremor ratio is near 1 when the envelope fluctuation is concentrated in
the 4-6 Hz band (rhythmic tremor bursts) and about 0.5 for broadband,
non-rhythmic voluntary fluctuation (design doc 7.7).

The tremor amplitude grows with effort even without tremor: EMG is noise-like,
so its envelope always fluctuates in proportion to its level, and a fixed share
of that falls in 4-6 Hz. The tremor depth divides by the slow envelope, which
cancels this: about 0.05 at any effort without tremor, about 0.3 or more with
tremor bursting (synthetic data). Below depth_min_envelope_mv (rest) the slow
envelope is replaced by that floor so noise is not amplified.

dominant_frequency() finds where the fast envelope's fluctuation peaks over a
recent window, e.g. to check that a tremor (or a mimicked one) is at 4-6 Hz.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

from .config import ProcessingConfig
from .emg_source import EMGBlock

FloatArray = NDArray[np.float64]


class StreamingFilter:
    """
    A second-order-sections IIR filter applied along time to (n_channels, n)
    blocks, with the filter state carried between blocks.

    steady_state_start: initialize the state as if the first sample had been
    present forever. This avoids a large start-up transient from an electrode
    DC offset. Otherwise the state starts at zero.
    """

    def __init__(self, sos: FloatArray, n_channels: int, steady_state_start: bool) -> None:
        self.sos = sos
        self.n_channels = n_channels
        self.steady_state_start = steady_state_start
        self._state: FloatArray | None = None

    def reset(self) -> None:
        """Forget the filter state; the next block starts fresh."""
        self._state = None

    def process(self, block: FloatArray) -> FloatArray:
        """Filter one (n_channels, n) block."""
        if self._state is None:
            if self.steady_state_start:
                zi = sosfilt_zi(self.sos)  # (n_sections, 2) for a unit step
                self._state = zi[:, None, :] * block[:, 0][None, :, None]
            else:
                self._state = np.zeros((self.sos.shape[0], self.n_channels, 2))
        output, self._state = sosfilt(self.sos, block, axis=1, zi=self._state)
        result: FloatArray = np.asarray(output, dtype=np.float64)
        return result


class MovingRMS:
    """
    Trailing-window RMS per channel. Like SignalProcess.smooth_signal (offline),
    the first outputs average over the samples seen so far.
    """

    def __init__(self, n_channels: int, window: int) -> None:
        if window < 1:
            raise ValueError("The RMS window must be at least one sample")
        self.n_channels = n_channels
        self.window = window
        self._history = np.zeros((n_channels, 0))

    def reset(self) -> None:
        """Forget the window contents."""
        self._history = np.zeros((self.n_channels, 0))

    def process(self, block: FloatArray) -> FloatArray:
        """RMS of the last `window` samples, for every sample of the block."""
        squares = np.concatenate([self._history, block**2], axis=1)
        sums = np.concatenate([np.zeros((self.n_channels, 1)), np.cumsum(squares, axis=1)], axis=1)
        n_history = self._history.shape[1]
        ends = n_history + 1 + np.arange(block.shape[1])  # index into sums, exclusive end
        starts = np.maximum(ends - self.window, 0)
        counts = ends - starts
        mean_square = (sums[:, ends] - sums[:, starts]) / counts
        self._history = squares[:, -(self.window - 1) :] if self.window > 1 else squares[:, :0]
        result: FloatArray = np.sqrt(np.maximum(mean_square, 0.0))
        return result


@dataclass(frozen=True)
class ProcessedBlock:  # pylint: disable=too-many-instance-attributes
    """All signals for one block, each (n_channels, n_samples), in mV except ratio and depth."""

    raw: FloatArray
    filtered: FloatArray
    slow_envelope: FloatArray
    fast_envelope: FloatArray
    tremor_amplitude: FloatArray
    tremor_ratio: FloatArray
    tremor_depth: FloatArray
    first_sample_index: int
    receive_time: float
    sample_rate: float

    @property
    def n_samples(self) -> int:
        """Samples per channel in this block."""
        return int(self.raw.shape[1])


class StreamingProcessor:
    """Multi-channel causal EMG processing; see the module docstring."""

    def __init__(self, config: ProcessingConfig, n_channels: int, sample_rate: float) -> None:
        nyquist = sample_rate / 2
        if config.bandpass_high_hz >= nyquist:
            raise ValueError(
                f"Band-pass upper edge {config.bandpass_high_hz} Hz must be below "
                f"the Nyquist frequency {nyquist} Hz"
            )
        if config.notch_enabled and config.notch_hz >= nyquist:
            raise ValueError(f"Notch at {config.notch_hz} Hz is above the Nyquist frequency")
        self.config = config
        self.n_channels = n_channels
        self.sample_rate = sample_rate

        bandpass = butter(
            config.bandpass_order,
            [config.bandpass_low_hz, config.bandpass_high_hz],
            btype="bandpass",
            fs=sample_rate,
            output="sos",
        )
        self._bandpass = StreamingFilter(bandpass, n_channels, steady_state_start=True)
        self._notch: StreamingFilter | None = None
        if config.notch_enabled:
            # Removes the 50 Hz powerline interference
            b, a = iirnotch(config.notch_hz, config.notch_quality, fs=sample_rate)
            self._notch = StreamingFilter(tf2sos(b, a), n_channels, steady_state_start=True)
        self._slow = MovingRMS(n_channels, max(1, round(config.slow_window_s * sample_rate)))
        fast = butter(2, config.fast_cutoff_hz, btype="lowpass", fs=sample_rate, output="sos")
        self._fast = StreamingFilter(fast, n_channels, steady_state_start=False)
        tremor = butter(
            2, list(config.tremor_band_hz), btype="bandpass", fs=sample_rate, output="sos"
        )
        self._tremor_band = StreamingFilter(tremor, n_channels, steady_state_start=False)
        self._tremor_rms = MovingRMS(
            n_channels, max(1, round(config.tremor_window_s * sample_rate))
        )
        rhythm = butter(
            2, list(config.rhythm_band_hz), btype="bandpass", fs=sample_rate, output="sos"
        )
        self._rhythm_band = StreamingFilter(rhythm, n_channels, steady_state_start=False)
        self._rhythm_rms = MovingRMS(
            n_channels, max(1, round(config.tremor_window_s * sample_rate))
        )

    def reset(self) -> None:
        """Reset every filter, e.g. after a reconnection."""
        for stage in (
            self._bandpass,
            self._notch,
            self._slow,
            self._fast,
            self._tremor_band,
            self._tremor_rms,
            self._rhythm_band,
            self._rhythm_rms,
        ):
            if stage is not None:
                stage.reset()

    def process(self, block: EMGBlock) -> ProcessedBlock:
        """Process one block; the state carries over to the next call."""
        raw = np.asarray(block.samples, dtype=np.float64)
        if raw.shape[0] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {raw.shape[0]}")
        filtered = self._bandpass.process(raw)
        if self._notch is not None:
            filtered = self._notch.process(filtered)  # band-stop filter
        slow = self._slow.process(filtered)
        fast = self._fast.process(np.abs(filtered))
        tremor = self._tremor_rms.process(self._tremor_band.process(fast))
        fluctuation = self._rhythm_rms.process(self._rhythm_band.process(fast))
        ratio = np.divide(tremor, fluctuation, out=np.zeros_like(tremor), where=fluctuation > 1e-12)
        depth = tremor / np.maximum(slow, self.config.depth_min_envelope_mv)
        return ProcessedBlock(
            raw=raw,
            filtered=filtered,
            slow_envelope=slow,
            fast_envelope=fast,
            tremor_amplitude=tremor,
            tremor_ratio=ratio,
            tremor_depth=depth,
            first_sample_index=block.first_sample_index,
            receive_time=block.receive_time,
            sample_rate=self.sample_rate,
        )


def dominant_frequency(
    signal: FloatArray, sample_rate: float, band_hz: tuple[float, float]
) -> FloatArray:
    """
    Frequency (Hz) where each row of an (n_channels, n) signal fluctuates most
    within band_hz: peak of the Hann-windowed spectrum of the mean-removed
    signal, zero-padded and refined by parabolic interpolation. NaN for a row
    that is too short (under two periods of the band's lower edge) or flat.
    """
    n_channels, n = signal.shape
    result = np.full(n_channels, np.nan)
    if n < 2 * sample_rate / band_hz[0]:
        return result
    centred = signal - signal.mean(axis=1, keepdims=True)
    n_fft = max(1 << int(np.ceil(np.log2(n))), int(sample_rate / 0.05))  # <= 0.05 Hz bins
    spectrum = np.abs(np.fft.rfft(centred * np.hanning(n), n=n_fft, axis=1))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    in_band = np.flatnonzero((freqs >= band_hz[0]) & (freqs <= band_hz[1]))
    for row in range(n_channels):
        values = spectrum[row, in_band]
        if values.max() <= 0.0:
            continue
        peak = int(np.argmax(values))
        offset = 0.0
        if 0 < peak < len(values) - 1:
            left, centre, right = values[peak - 1], values[peak], values[peak + 1]
            denominator = left - 2 * centre + right
            if denominator != 0.0:
                offset = 0.5 * (left - right) / denominator
        result[row] = freqs[in_band[peak]] + offset * (freqs[1] - freqs[0])
    return result
