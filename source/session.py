"""
Measurement sessions: every run with `run_emg.py --out <folder>` gets its own
sub-folder, so recordings are never overwritten:

  <folder>/<YYYYMMDD-HHMMSS>_<source>[_<label>]/
      emg.csv        raw EMG (EMGRecorder): time_s and one column per channel in mV
      session.json   metadata: times, source, sample rate, channels and TCU slots,
                     source details, counters, errors and the full configuration

session.json is written when streaming starts (status "recording") and again
when the run ends ("complete" or "error"), so a crash still leaves the
metadata of the samples already on disk.

find_recording() reads such a session (or a plain `--save` CSV) back, for
replay with `--source recorded --file <session folder>` or for analysis:

  recording = find_recording(Path("measurements/20261023-140512_trigno_grip"))
  times, emg = recording.load()   # (n,) in s and (n_channels, n) in mV
"""

import csv
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import AppConfig, ChannelConfig
from .emg_source import EMGSource

EMG_FILE = "emg.csv"
METADATA_FILE = "session.json"


def _safe_label(label: str) -> str:
    """A label usable in a folder name on every OS."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", label.strip()).strip("-")


def create_session_dir(
    root: Path, source_kind: str, label: str | None = None, now: datetime | None = None
) -> Path:
    """Create and return a new, empty session folder under root."""
    now = now or datetime.now()
    name = f"{now:%Y%m%d-%H%M%S}_{source_kind}"
    if label and _safe_label(label):
        name += f"_{_safe_label(label)}"
    root.mkdir(parents=True, exist_ok=True)
    for suffix in range(1000):
        path = root / (name if suffix == 0 else f"{name}-{suffix}")
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return path
    raise FileExistsError(f"Too many sessions named {name} in {root}")


class SessionLog:
    """Writes session.json for one measurement."""

    def __init__(
        self,
        directory: Path,
        config: AppConfig,
        label: str | None = None,
        note: str | None = None,
    ) -> None:
        self.directory = directory
        self.path = directory / METADATA_FILE
        self.emg_path = directory / EMG_FILE
        self.data: dict[str, Any] = {
            "status": "starting",
            "label": label or "",
            "note": note or "",
            "emg_file": EMG_FILE,
            "emg_units": "mV",
            "channels": [asdict(channel) for channel in config.channels],
            "config": asdict(config),
        }

    def started(self, source: EMGSource) -> None:
        """Streaming has started: record the source and the sample rate."""
        self.data.update(
            status="recording",
            start_time=datetime.now().astimezone().isoformat(timespec="seconds"),
            source=source.kind,
            sample_rate_hz=source.sample_rate,
            source_info=source.info(),
        )
        self._write()

    def finished(
        self,
        samples: int,
        counters: dict[str, int],
        events: Sequence[tuple[float, str]],
        error: str | None,
    ) -> None:
        """The run has ended. events: (recording time in s, message), see StreamCore."""
        rate = float(self.data.get("sample_rate_hz", 0.0))
        self.data.update(
            status="error" if error else "complete",
            end_time=datetime.now().astimezone().isoformat(timespec="seconds"),
            samples_per_channel=samples,
            duration_s=samples / rate if rate > 0 else 0.0,
            counters=counters,
            events=[{"time_s": round(time_s, 4), "message": text} for time_s, text in events],
            error=error or "",
        )
        self._write()

    def _write(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.path)


class RecordingError(ValueError):
    """A session or recording of ours that cannot be read."""


@dataclass(frozen=True)
class Recording:
    """
    A recording made by this program, ready for replay or analysis.

    channels: as recorded; recorded_column is the channel's column in emg_path
    metadata: session.json, or empty for a plain --save CSV
    """

    emg_path: Path
    sample_rate_hz: float
    channels: tuple[ChannelConfig, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def load(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """time_s with shape (n,) and the EMG with shape (n_channels, n), in mV."""
        data = np.loadtxt(self.emg_path, delimiter=",", skiprows=1, ndmin=2)
        columns = [channel.recorded_column for channel in self.channels]
        return data[:, 0], data[:, columns].T


def _read_header_and_rate(path: Path) -> tuple[list[str], float]:
    """The header row and the sample rate implied by the first two time values (0 if unknown)."""
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader, [])
        rows = [row for _, row in zip(range(2), reader)]
    try:
        step = float(rows[1][0]) - float(rows[0][0])
    except (IndexError, ValueError):
        return header, 0.0
    return header, 1.0 / step if step > 0 else 0.0


def find_recording(
    path: Path, known_channels: Sequence[ChannelConfig] = ()
) -> Recording | None:
    """
    Our recording at path: a session folder, its emg.csv or session.json, or a
    --save CSV. Returns None for any other file (such as the dataset CSVs, which
    have no time_s column). Raises RecordingError if it is ours but unreadable.

    The sample rate and channel roles come from session.json; for a plain CSV
    the rate follows from the time column and the roles from known_channels.
    """
    if path.is_dir():
        path = path / EMG_FILE
        if not path.is_file():
            raise RecordingError(f"No {EMG_FILE} in the session folder {path.parent}")
    elif path.name == METADATA_FILE:
        path = path.with_name(EMG_FILE)
    if not path.is_file():
        return None

    header, csv_rate = _read_header_and_rate(path)
    if not header or header[0] != "time_s":
        return None
    names = [column.removesuffix("_mV") for column in header]

    metadata: dict[str, Any] = {}
    metadata_path = path.with_name(METADATA_FILE)
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise RecordingError(f"Cannot read {metadata_path}: {error}") from error
        if metadata.get("emg_file", EMG_FILE) != path.name:
            metadata = {}

    recorded = {channel["name"]: channel for channel in metadata.get("channels", [])}
    known = {channel.name: channel for channel in known_channels}
    channels = []
    for column, name in enumerate(names[1:], start=1):
        role = recorded.get(name, {}).get("role") or getattr(known.get(name), "role", "other")
        slot = recorded.get(name, {}).get("trigno_slot", column)
        channels.append(ChannelConfig(name, role, int(slot), column))
    if not channels:
        raise RecordingError(f"{path} has no EMG columns")

    rate = float(metadata.get("sample_rate_hz", 0.0)) or csv_rate
    if rate <= 0:
        raise RecordingError(f"Cannot tell the sample rate of {path}")
    return Recording(path, rate, tuple(channels), metadata)
