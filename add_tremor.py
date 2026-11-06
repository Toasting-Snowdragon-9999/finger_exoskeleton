"""
Lay synthetic tremor over a recorded session and save it as a new session,
for bench tests of tremor detection and control on real EMG.

Examples:
  python add_tremor.py measurements/20261105-144436_trigno_to-greb --span 20-28
  python add_tremor.py <session> --span 20-28 --span 35-43 --frequency 5 --depth 0.8
  python add_tremor.py <session> --span 50-58 --rest-mv 0.01      (rest tremor)

Then replay it:
  python run_emg.py --source recorded --no-loop --file <new session folder>

See source/tremor_overlay.py for how the tremor is made.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from source.session import EMG_FILE, METADATA_FILE, RecordingError, find_recording
from source.tremor_overlay import TremorOverlay, add_tremor


def parse_span(text: str) -> tuple[float, float]:
    """'20-28' -> (20.0, 28.0)."""
    try:
        start, end = (float(part) for part in text.split("-"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"span must look like 20-28, got {text!r}") from error
    return start, end


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("session", type=Path, help="session folder recorded with run_emg.py --out")
    parser.add_argument("--span", type=parse_span, action="append", required=True,
                        help="tremor from START to END seconds, e.g. 20-28 (repeatable)")
    parser.add_argument("--frequency", type=float, default=5.0, help="tremor Hz (default 5)")
    parser.add_argument("--depth", type=float, default=0.8,
                        help="burst modulation depth 0-1 (default 0.8)")
    parser.add_argument("--rest-mv", type=float, default=0.0,
                        help="RMS of added bursts in mV, for rest tremor (default 0: none)")
    parser.add_argument("--ramp", type=float, default=0.5, help="fade in/out seconds (default 0.5)")
    parser.add_argument("--out", type=Path, help="new session folder (default: <session>_tremor)")
    return parser.parse_args(argv)


def describe(overlay: TremorOverlay) -> str:
    """One line for the session note and the console."""
    spans = ", ".join(f"{start:g}-{end:g} s" for start, end in overlay.spans)
    text = f"synthetic {overlay.frequency_hz:g} Hz tremor, depth {overlay.depth:g}, at {spans}"
    return text + (f", rest bursts {overlay.rest_mv:g} mV" if overlay.rest_mv else "")


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = parse_args(argv)
    try:
        recording = find_recording(args.session)
        if recording is None:
            raise RecordingError(f"{args.session} is not a recording made by run_emg.py")
        overlay = TremorOverlay(
            tuple(args.span), args.frequency, args.depth, args.rest_mv, args.ramp
        )
    except (RecordingError, ValueError) as error:
        print(f"Cannot add tremor: {error}", file=sys.stderr)
        return 2

    times, emg = recording.load()
    roles = [channel.role for channel in recording.channels]
    emg = add_tremor(times, emg, roles, recording.sample_rate_hz, overlay)

    source_dir = recording.emg_path.parent
    out = args.out or source_dir.with_name(f"{source_dir.name}_tremor")
    out.mkdir(parents=True, exist_ok=False)
    with open(out / EMG_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s", *(f"{channel.name}_mV" for channel in recording.channels)])
        writer.writerows(np.column_stack((times, emg.T)).tolist())

    metadata = dict(recording.metadata)
    metadata.update(
        label=f"{metadata.get('label', '')}+tremor".lstrip("+"),
        note=f"{metadata.get('note', '')} [{describe(overlay)}]".strip(),
        tremor_overlay={"based_on": source_dir.name, "spans_s": [list(s) for s in overlay.spans],
                        "frequency_hz": overlay.frequency_hz, "depth": overlay.depth,
                        "rest_mv": overlay.rest_mv, "ramp_s": overlay.ramp_s},
        # The channels are now the columns of the new emg.csv, in order
        channels=[{"name": c.name, "role": c.role, "trigno_slot": c.trigno_slot,
                   "recorded_column": column}
                  for column, c in enumerate(recording.channels, start=1)],
        sample_rate_hz=recording.sample_rate_hz,
        emg_file=EMG_FILE,
    )
    (out / METADATA_FILE).write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print(f"{describe(overlay)}\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
