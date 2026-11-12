"""
Measurement sessions on disk, and replaying them.

Every run with `run_emg.py --out <folder>` gets its own sub-folder:

  <folder>/<YYYYMMDD-HHMMSS>_<source>[_<label>]/
      emg.csv        raw EMG: a header, then time_s and one column per channel in mV
      session.json   start and end time, source, sample rate, channels, messages, error

`run_emg.py --source recorded --file <session folder>` replays such a folder
through the same pipeline as live EMG.
"""

import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import settings
from settings import Channel
from source.signal_processing import EMGBlock

EMG_FILE = "emg.csv"
METADATA_FILE = "session.json"


def new_session_folder(root: Path, source_name: str, label: str | None = None) -> Path:
    """Create a new, empty session folder in root, named after the time, source and label."""
    name = f"{datetime.now():%Y%m%d-%H%M%S}_{source_name}"
    if label:
        safe_label = re.sub(r"[^A-Za-z0-9_-]+", "-", label.strip()).strip("-")
        if safe_label:
            name += f"_{safe_label}"
    root.mkdir(parents=True, exist_ok=True)
    folder = root / name
    number = 1
    while folder.exists():  # two runs in the same second
        folder = root / f"{name}-{number}"
        number += 1
    folder.mkdir()
    return folder


def now_text() -> str:
    """The current time as text, e.g. 2026-11-05T14:47:38+01:00."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


class SessionWriter:
    """Writes emg.csv and session.json for one run."""

    def __init__(self, folder: Path, channels: list[Channel], sample_rate: float,
                 source_name: str, source_info: dict, label: str | None = None) -> None:
        self.folder = folder
        self.sample_rate = sample_rate
        self.samples = 0
        self.last_flush = time.monotonic()
        self.file = open(folder / EMG_FILE, "w", newline="", encoding="utf-8")  # pylint: disable=consider-using-with
        self.writer = csv.writer(self.file)
        self.writer.writerow(["time_s"] + [f"{channel.name}_mV" for channel in channels])
        self.metadata = {
            "status": "recording",
            "label": label or "",
            "emg_file": EMG_FILE,
            "emg_units": "mV",
            "channels": [{"name": c.name, "role": c.role, "trigno_slot": c.slot} for c in channels],
            "processing": settings.processing_settings(),
            "start_time": now_text(),
            "source": source_name,
            "sample_rate_hz": sample_rate,
            "source_info": source_info,
        }
        self.write_metadata()  # written now too, so a crash still leaves the metadata

    def write(self, block: EMGBlock) -> None:
        """Append one block; time_s is the sample number divided by the sample rate."""
        n = block.samples.shape[1]
        times = (block.first_index + np.arange(n)) / self.sample_rate
        self.writer.writerows(np.column_stack((times, block.samples.T)).tolist())
        self.samples += n
        if time.monotonic() - self.last_flush >= 1.0:  # keep the data on disk if the PC dies
            self.file.flush()
            self.last_flush = time.monotonic()

    def finish(self, messages: list[tuple[float, str]], error: str | None) -> None:
        """Close emg.csv and write the end of the run to session.json."""
        self.file.close()
        self.metadata.update(
            status="error" if error else "complete",
            end_time=now_text(),
            samples_per_channel=self.samples,
            duration_s=self.samples / self.sample_rate,
            events=[{"time_s": round(t, 4), "message": text} for t, text in messages],
            error=error or "",
        )
        self.write_metadata()

    def write_metadata(self) -> None:
        """Write session.json."""
        text = json.dumps(self.metadata, indent=2, ensure_ascii=False)
        (self.folder / METADATA_FILE).write_text(text, encoding="utf-8")


class Recording:
    """A session folder read back from disk."""

    def __init__(self, path: Path) -> None:
        # Accept the session folder, its emg.csv or its session.json
        folder = path if path.is_dir() else path.parent
        self.emg_path = folder / EMG_FILE
        metadata_path = folder / METADATA_FILE
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8")) \
            if metadata_path.is_file() else {}

        with open(self.emg_path, newline="", encoding="utf-8") as file:
            header = next(csv.reader(file))
        data = np.loadtxt(self.emg_path, delimiter=",", skiprows=1, ndmin=2)
        self.times = data[:, 0]
        self.emg = data[:, 1:].T  # (n_channels, n) in mV

        saved = {channel["name"]: channel for channel in self.metadata.get("channels", [])}
        self.channels = []
        for column, title in enumerate(header[1:], start=1):
            name = title.removesuffix("_mV")
            info = saved.get(name, {})
            self.channels.append(Channel(name, info.get("role", "other"),
                                         int(info.get("trigno_slot", column))))
        self.sample_rate = float(self.metadata.get("sample_rate_hz", 0.0)) \
            or 1.0 / (self.times[1] - self.times[0])


class RecordedSource:
    """Replays a session folder block by block, at the speed it was recorded."""

    name = "recorded"

    def __init__(self, path: Path, loop: bool = True, realtime: bool = True) -> None:
        self.recording = Recording(path)
        self.channels = self.recording.channels
        self.sample_rate = self.recording.sample_rate
        self.loop = loop
        self.realtime = realtime
        self.position = 0  # next column of the recording to send
        self.next_index = 0  # sample number of the next block (keeps counting when looping)
        self.start_time = time.monotonic()

    def start(self) -> None:
        """Start the replay clock."""
        self.start_time = time.monotonic()

    def read_block(self) -> EMGBlock | None:
        """The next block, or None at the end of the recording (without loop)."""
        emg = self.recording.emg
        if self.position >= emg.shape[1]:
            if not self.loop:
                return None
            self.position = 0
        end = min(self.position + settings.BLOCK_SIZE, emg.shape[1])
        samples = emg[:, self.position:end].copy()
        first = self.next_index
        if self.realtime:  # wait until the last sample of the block would have been measured
            due = self.start_time + (first + samples.shape[1]) / self.sample_rate
            time.sleep(max(0.0, due - time.monotonic()))
        self.position = end
        self.next_index += samples.shape[1]
        return EMGBlock(samples, first, time.monotonic())

    def stop(self) -> None:
        """Nothing to close for a replay."""

    def info(self) -> dict:
        """What is being replayed, for session.json."""
        return {"file": str(self.recording.emg_path)}
