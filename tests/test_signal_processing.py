"""The filter chain in source/signal_processing.py."""

import numpy as np

from source.signal_processing import EMGBlock, MovingRMS, SignalProcessor
from tests.fakes import RATE, make_emg


def process_all(signal: np.ndarray, block_size: int = 27):
    """Process a (n_channels, n) signal in blocks; returns the list of processed blocks."""
    processor = SignalProcessor(signal.shape[0], RATE)
    return [processor.process(EMGBlock(signal[:, start:start + block_size], start, 0.0))
            for start in range(0, signal.shape[1], block_size)]


def joined(blocks, name: str) -> np.ndarray:
    """One signal of all blocks, joined in time."""
    return np.concatenate([getattr(block, name) for block in blocks], axis=1)


def sine(frequency: float, seconds: float = 2.0, amplitude: float = 1.0) -> np.ndarray:
    """A (1, n) sine wave."""
    times = np.arange(round(seconds * RATE)) / RATE
    return amplitude * np.sin(2 * np.pi * frequency * times)[None, :]


def test_bandpass_keeps_emg_and_removes_slow_movement() -> None:
    """100 Hz passes the band-pass, 5 Hz does not."""
    kept = joined(process_all(sine(100)), "filtered")[0, 2000:]
    removed = joined(process_all(sine(5)), "filtered")[0, 2000:]
    assert 0.6 < np.sqrt(np.mean(kept**2)) < 0.8
    assert np.sqrt(np.mean(removed**2)) < 0.01


def test_notch_removes_power_line() -> None:
    """50 Hz is removed by the notch filter."""
    filtered = joined(process_all(sine(50)), "filtered")[0, 2000:]
    assert np.sqrt(np.mean(filtered**2)) < 0.05


def test_blocks_give_the_same_result_as_one_call() -> None:
    """The filters keep their state between blocks."""
    signal = make_emg([(3.0, 0.1)])
    small = process_all(signal, block_size=27)
    large = process_all(signal, block_size=signal.shape[1])
    for name in ("filtered", "slow", "fast", "depth", "ratio"):
        assert np.allclose(joined(small, name), joined(large, name))


def test_moving_rms_uses_the_samples_seen_so_far() -> None:
    """At the start the window is not full yet; then it is the RMS of the last `window` samples."""
    rms = MovingRMS(1, window=2)
    assert np.allclose(rms.process(np.array([[3.0, 4.0]])), [[3.0, np.sqrt(12.5)]])
    assert np.allclose(rms.process(np.array([[0.0]])), [[np.sqrt(8.0)]])


def test_slow_envelope_of_a_sine_is_its_rms() -> None:
    """A 100 Hz sine of amplitude 0.1 mV has an RMS of 0.0707 mV."""
    slow = joined(process_all(sine(100, amplitude=0.1)), "slow")
    assert abs(slow[0, -1] - 0.1 / np.sqrt(2)) < 0.005


def test_tremor_bursts_raise_depth_and_ratio() -> None:
    """EMG bursting at 5 Hz has a higher tremor depth and ratio than steady EMG."""
    steady = process_all(make_emg([(4.0, 0.2)], n_channels=1))
    tremor = process_all(make_emg([(4.0, 0.2)], n_channels=1, tremor_hz=5.0))
    assert joined(tremor, "depth")[0, -1] > 2 * joined(steady, "depth")[0, -1]
    assert joined(tremor, "ratio")[0, -1] > 0.8 > joined(steady, "ratio")[0, -1]
