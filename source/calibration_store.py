"""
A saved calibration, so the rest/reference routine is only done when the user
asks for it: the first run (no file yet), `run_emg.py --calibrate`, or the 'c'
key in the viewer. A successful calibration is written to the file, and later
runs load it instead of calibrating.

The file is only used when it fits the run: the same channels (names, in
order), the same signal processing settings, and levels that pass the same
reference check as a fresh calibration. Otherwise it is ignored with a message
and the routine runs (the file is replaced when that succeeds).

It holds levels of one sensor placement and one day's skin condition, so
calibrate again after moving the sensors; the file's date is in the message.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .activation import Calibration, CalibrationError
from .config import AppConfig

DEFAULT_PATH = Path("calibration.json")


def _processing(config: AppConfig) -> dict[str, Any]:
    """The processing settings as saved in JSON (tuples as lists), to compare with a saved file."""
    return dict(json.loads(json.dumps(asdict(config.processing))))


@dataclass(frozen=True)
class CalibrationStore:
    """
    path:      the calibration file
    use_saved: False (--calibrate): ignore an existing file, but still save the new one
    """

    path: Path = DEFAULT_PATH
    use_saved: bool = True

    def load(self, config: AppConfig) -> tuple[Calibration | None, str]:
        """
        (calibration, message): the saved calibration if there is one that fits this
        run, else None. The message says what was loaded or why a file was ignored
        (empty when there is simply no file or it is not to be used).
        """
        if not self.use_saved or not self.path.is_file():
            return None, ""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            calibration = Calibration.load(self.path)
        except (OSError, ValueError, CalibrationError) as error:
            return None, f"Saved calibration ignored: {error}"
        ignored = "Saved calibration ignored: "
        if calibration.channel_names != config.channel_names:
            return None, (
                f"{ignored}it is for channels {list(calibration.channel_names)}, "
                f"not {list(config.channel_names)}"
            )
        if data.get("processing") != _processing(config):
            return None, f"{ignored}the signal processing settings changed since it was saved"
        try:
            calibration.validate(config.calibration.min_reference_ratio)
        except CalibrationError as error:
            return None, f"{ignored}{error}"
        saved_at = data.get("saved_at", "unknown date")
        return calibration, f"Saved calibration loaded from {self.path} ({saved_at})"

    def save(self, calibration: Calibration, config: AppConfig) -> None:
        """Write the calibration with what it was made with. Raises OSError."""
        calibration.save(
            self.path,
            extra={
                "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "processing": _processing(config),
            },
        )
