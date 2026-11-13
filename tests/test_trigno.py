"""The Trigno client in source/trigno.py against the fake TCU in source/mock_tcu.py."""

import numpy as np
import pytest

import settings
from settings import Channel
from source.mock_tcu import MockTCU
from source.trigno import TrignoSource


def started(slots: list[int], signal: np.ndarray) -> tuple[MockTCU, TrignoSource]:
    """A running fake TCU and a Trigno client connected to it."""
    mock = MockTCU(slots, signal)
    mock.start()
    source = TrignoSource(settings.CHANNELS, "127.0.0.1", mock.command_port, mock.emg_port)
    return mock, source


def test_reads_the_configured_slots_in_mv() -> None:
    """2000 Hz, column = STARTINDEX - 1, volts become mV."""
    slots = [channel.slot for channel in settings.CHANNELS]
    signal = np.vstack([np.linspace(0, 0.001 * (i + 1) * (-1) ** i, 1000) for i in range(len(slots))])
    mock, source = started(slots, signal)
    try:
        source.start()
        assert source.sample_rate == 2000.0
        assert source.columns == [slot - 1 for slot in slots]
        samples: list[np.ndarray] = []
        while sum(block.shape[1] for block in samples) < 100:
            samples.append(source.read_block().samples)
        received = np.concatenate(samples, axis=1)[:, :100]
        assert np.allclose(received, signal[:, :100] * 1000, atol=1e-6)
    finally:
        source.stop()
        mock.stop()
    assert mock.commands[-2:] == ["STOP", "QUIT"]


def test_an_unpaired_slot_is_refused() -> None:
    """A configured slot without a paired sensor stops the start with a clear message."""
    mock = MockTCU([15], np.zeros((1, 100)))
    mock.start()
    channels = [Channel("FDS", "flexor", 15), Channel("EDC", "extensor", 14)]
    source = TrignoSource(channels, "127.0.0.1", mock.command_port, mock.emg_port)
    try:
        with pytest.raises(ValueError, match="No sensor is paired in TCU slot 14"):
            source.start()
    finally:
        source.stop()
        mock.stop()
