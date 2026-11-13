"""
A cue protocol: tells the person what to do and when, and turns that into labels.

The GRU needs a reference for the movement intention that does not come from the
EMG itself. A cue protocol gives one: the screen says "GRIB objektet" from 14 to 18 s,
so the intention in that interval is known to be +1.

The protocol is a CSV file (see protocols/greb_1.csv; no commas in the cue text):

  start_s,end_s,cue,intention,tremor
  10,14,HVIL: slip og slap af,0,0
  14,18,GRIB objektet med normalt greb og hold,1,0
  23,27,KLEM HÅRDT om objektet og hold,1,0

  intention   -1 = open actively, 0 = rest, +1 = close fully (0.5 = half); empty = unknown
  tremor      1 = tremor, 0 = no tremor; empty = unknown

`run_emg.py --cues <file> --out <folder>` shows the cues during the recording and
writes them as labels.csv into the session folder. Times are seconds of EMG
received, the same clock as time_s in emg.csv.
"""

import csv
from pathlib import Path

LABELS_FILE = "labels.csv"


class CueProtocol:
    """The rows of a cue protocol file."""

    def __init__(self, path: Path) -> None:
        with open(path, newline="", encoding="utf-8") as file:
            self.rows = list(csv.DictReader(file))

    def cue_at(self, time_s: float) -> str:
        """What the person should do at this time ("" before the first row)."""
        for row in self.rows:
            if float(row["start_s"]) <= time_s < float(row["end_s"]):
                return row["cue"]
        if self.rows and time_s >= float(self.rows[-1]["end_s"]):
            return "FÆRDIG"
        return ""

    def write_labels(self, folder: Path) -> None:
        """Write the protocol's intention and tremor as labels.csv in a session folder."""
        with open(folder / LABELS_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["start_s", "end_s", "intention", "tremor"])
            for row in self.rows:
                writer.writerow([row["start_s"], row["end_s"], row["intention"], row["tremor"]])
