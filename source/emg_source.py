"""
The EMG source interface shared by every source: the real Delsys Trigno
(trigno.py) and the stand-ins without hardware (sensor_mock.py). Each delivers
EMGBlock objects, so the downstream code cannot tell them apart.

Units: all sources deliver samples in mV.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EMGBlock:
    """
    A block of consecutive samples from all channels.

    samples:            shape (n_channels, n_samples), in mV
    first_sample_index: index of samples[:, 0] since the source started; gaps
                        in the index mean samples were lost
    receive_time:       time.monotonic() when the block was received
    """

    samples: NDArray[np.float64]
    first_sample_index: int
    receive_time: float

    @property
    def n_samples(self) -> int:
        """Number of samples per channel."""
        return int(self.samples.shape[1])


class SourceError(RuntimeError):
    """Base class for acquisition errors."""


class SourceConnectionError(SourceError):
    """The source could not connect, or the connection was lost."""


class SourceTimeoutError(SourceError):
    """No data arrived within the timeout. The connection may still be alive."""


class EMGSource(ABC):
    """Interface shared by all EMG sources."""

    kind = "abstract"

    def __init__(self, channel_names: Sequence[str], sample_rate: float) -> None:
        self.channel_names: tuple[str, ...] = tuple(channel_names)
        self.sample_rate: float = sample_rate
        self.connected: bool = False

    @property
    def n_channels(self) -> int:
        """Number of EMG channels delivered."""
        return len(self.channel_names)

    @abstractmethod
    def start(self) -> None:
        """Connect and start streaming. Raises SourceConnectionError on failure."""

    @abstractmethod
    def read_block(self) -> EMGBlock | None:
        """
        Block until the next EMGBlock is available. Returns None when a finite
        source has no more data. Raises SourceTimeoutError or
        SourceConnectionError.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop streaming and release all connections. Safe to call twice."""

    def info(self) -> dict[str, str]:
        """Human-readable facts about the source, for the monitor."""
        return {"source": self.kind, "sample rate": f"{self.sample_rate:g} Hz"}

    def __enter__(self) -> "EMGSource":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()
