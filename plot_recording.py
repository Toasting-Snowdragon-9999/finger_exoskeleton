"""
Show a recording with its labels as colours, to check that it is usable for training.

  python plot_recording.py measurements/<session folder>           opens a window
  python plot_recording.py measurements/<session folder> --save    writes labels.png there

For every channel the filtered EMG and its slow envelope are drawn, and the
background is coloured by what the person was asked to do (from labels.csv):

  blue    HVIL          rest
  green   GRIB          normal grip
  red     KLEM          hard squeeze
  grey    KALIBRERING / VENT, or a part without a label

A purple bar along the top marks tremor (label 1), e.g. synthetic tremor.
The hatched start of every coloured part is the reaction time (GRU_LABEL_MARGIN_S):
the label is unknown there and not used for training.

What to look for: the envelope is flat in the blue parts, rises shortly after
every green or red part begins, and falls again when it ends. A grip that
comes late, is missing or continues into the rest makes the labels wrong.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import settings
from source.cues import LABELS_FILE
from source.recording import Recording
from source.signal_processing import EMGBlock, SignalProcessor

COLORS = {"rest": "tab:blue", "grip": "tab:green", "squeeze": "tab:red", "other": "lightgrey"}
NAMES = {"rest": "HVIL (rest)", "grip": "GRIB (normal grip)", "squeeze": "KLEM (hard squeeze)",
         "other": "calibration / waiting / no label"}
TREMOR_COLOR = "purple"


def kind_of(row: dict) -> str:
    """What a labels.csv row is: "rest", "grip", "squeeze" or "other"."""
    cue = (row.get("cue") or "").upper()
    if cue.startswith("HVIL"):
        return "rest"
    if cue.startswith("GRIB"):
        return "grip"
    if cue.startswith("KLEM"):
        return "squeeze"
    if cue:  # KALIBRERING, VENT, ...
        return "other"
    # A labels.csv without cue texts (written by hand): go by the intention
    intention = row["intention"].strip()
    if intention == "":
        return "other"
    return "rest" if float(intention) == 0 else "grip"


def read_rows(folder: Path) -> list[dict]:
    """The rows of the recording's labels.csv ([] if it has none)."""
    path = folder / LABELS_FILE
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def plot_recording(folder: Path) -> Figure:
    """The figure described at the top of this file."""
    recording = Recording(folder)
    rows = read_rows(recording.emg_path.parent)
    n_channels = recording.emg.shape[0]
    processed = SignalProcessor(n_channels, recording.sample_rate).process(
        EMGBlock(recording.emg, 0, 0.0))
    times = recording.times

    figure, axes = plt.subplots(n_channels, 1, sharex=True, squeeze=False,
                                figsize=(14, 3 * n_channels + 1.5))
    for channel, ax in zip(recording.channels, axes[:, 0]):
        row_number = recording.channels.index(channel)
        for row in rows:
            start, end = float(row["start_s"]), float(row["end_s"])
            kind = kind_of(row)
            ax.axvspan(start, end, color=COLORS[kind], alpha=0.3, linewidth=0)
            if kind != "other":  # the reaction time: not used for training
                ax.axvspan(start, min(start + settings.GRU_LABEL_MARGIN_S, end), facecolor="none",
                           edgecolor="white", hatch="//", linewidth=0)
            if row["tremor"].strip() == "1":  # a bar along the top of the plot
                ax.axvspan(start, end, ymin=0.95, ymax=1.0, color=TREMOR_COLOR, linewidth=0)
        # The raw activity, faint, so the colours behind it stay visible
        ax.plot(times, np.abs(processed.filtered[row_number]) * 1000, color="grey",
                linewidth=0.3, alpha=0.4)
        ax.plot(times, processed.slow[row_number] * 1000, color="black", linewidth=1.5)
        top = max(float(np.percentile(processed.slow[row_number], 99.9)) * 1000 * 1.5, 1.0)
        ax.set_ylim(0, top)
        ax.set_ylabel(f"{channel.name} ({channel.role})\nuV")
        ax.grid(alpha=0.3)

    handles: list[Artist] = [Patch(color=color, alpha=0.3, label=NAMES[kind])
                             for kind, color in COLORS.items()]
    handles.append(Patch(facecolor="lightgrey", edgecolor="white", hatch="//",
                         label="reaction time (not used)"))
    handles.append(Patch(color=TREMOR_COLOR, label="tremor (bar at the top)"))
    handles.append(Line2D([], [], color="black", label="slow envelope"))
    handles.append(Line2D([], [], color="grey", alpha=0.4, label="|filtered EMG|"))
    axes[-1, 0].set_xlabel("Seconds (time_s of emg.csv)")
    note = "" if rows else "   (no labels.csv)"
    figure.suptitle(f"{recording.emg_path.parent.name}{note}")
    figure.legend(handles=handles, loc="lower center", fontsize=8, ncol=len(handles))
    figure.tight_layout(rect=(0, 0.05, 1, 1))  # leave room for the legend at the bottom
    return figure


def main(argv: list[str] | None = None) -> int:
    """Show the figure, or save it with --save."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="a session folder")
    parser.add_argument("--save", action="store_true",
                        help="write labels.png into the folder instead of opening a window")
    args = parser.parse_args(argv)
    figure = plot_recording(args.folder)
    if args.save:
        path = args.folder / "labels.png"
        figure.savefig(path, dpi=110)
        print(f"Saved {path}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
