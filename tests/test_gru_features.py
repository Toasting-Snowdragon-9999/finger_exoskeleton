"""The GRU input features in source/gru/features.py."""

import numpy as np
import pytest

from source.gru.features import FeatureExtractor, frame_features, normalize
from tests.fakes import make_emg


def extractor(n_channels: int = 2, sample_rate: float = 2000.0) -> FeatureExtractor:
    """An extractor with the default 100 frames per second and 50 ms window."""
    return FeatureExtractor(n_channels, sample_rate, 100.0, 0.05)


def test_frame_is_rms_then_mav_per_channel() -> None:
    """One frame holds the RMS of every channel, then the MAV of every channel."""
    window = np.array([[3.0, -3.0, 3.0, -3.0], [0.0, 2.0, 0.0, -2.0]])
    assert np.allclose(frame_features(window), [3.0, np.sqrt(2.0), 3.0, 1.0])


def test_one_frame_every_10_ms_after_the_first_window() -> None:
    """At 2000 Hz: the first frame after 100 samples, then one every 20 samples."""
    frames = extractor().process(np.zeros((2, 2000)))
    assert frames.shape == (1 + (2000 - 100) // 20, 4)
    assert extractor().frame_end_samples(3).tolist() == [99, 119, 139]


@pytest.mark.parametrize("block_size", [1, 27, 500])
def test_any_block_size_gives_the_same_frames(block_size: int) -> None:
    """Small blocks, Trigno-sized blocks and big blocks give identical frames."""
    signal = make_emg([(1.0, 0.1)])
    expected = extractor().process(signal)
    streaming = extractor()
    parts = [streaming.process(signal[:, start:start + block_size])
             for start in range(0, signal.shape[1], block_size)]
    assert np.array_equal(np.concatenate(parts), expected)


def test_the_input_block_is_not_changed() -> None:
    """The viewer keeps the filtered blocks, so the extractor must not modify them."""
    block = make_emg([(0.1, 0.1)])
    copy = block.copy()
    extractor().process(block)
    assert np.array_equal(block, copy)


def test_reset_needs_a_full_window_again() -> None:
    """After reset() the buffered samples are gone."""
    features = extractor()
    features.process(np.ones((2, 90)))
    features.reset()
    assert features.process(np.ones((2, 90))).shape == (0, 4)


def test_normalize_is_relative_to_the_rest_level() -> None:
    """Silence gives 0, the rest level gives log(2), and the scale of the EMG drops out."""
    rest = np.array([0.004, 0.002])
    frames = np.array([[0.0, 0.0, 0.0, 0.0], [0.004, 0.002, 0.004, 0.002]])
    assert np.allclose(normalize(frames, rest), [[0, 0, 0, 0], [np.log(2)] * 4])
    assert np.allclose(normalize(5 * frames, 5 * rest), normalize(frames, rest))


def test_a_sample_rate_without_whole_frames_is_refused() -> None:
    """1111.11 Hz does not give a whole number of samples per 10 ms."""
    with pytest.raises(ValueError, match="whole number of samples"):
        extractor(sample_rate=15 / 0.0135)
