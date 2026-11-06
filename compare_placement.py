"""
Compare sensor placements from recordings made with run_emg.py --out.

For each placement record one session: rest for the first 3 s, then several grips
(and a few fast ones), e.g.

  python run_emg.py --source trigno --out measurements --label plac-A

then compare them, one folder per placement:

  python compare_placement.py measurements/<plac-A folder> measurements/<plac-B folder>

Per channel it prints the noise floor, how much a grip stands out (SNR), motion
artifacts (spike ratio), repeatability and cross-talk; see source/placement.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import applied_config
from source.emg_source import EMGBlock
from source.placement import ChannelReport, analyse
from source.session import RecordingError, find_recording
from source.streaming import StreamingProcessor

BLOCK = 27
MAX_SPIKE = 4.0  # median peak / median level in a grip above this: likely motion artifacts
MAX_CV = 0.5  # grip levels varying more than this between grips: unrepeatable
MIN_SNR = 3.0  # what the calibration requires (min_reference_ratio)


def parse_window(text: str) -> tuple[float, float]:
    """'0.5-3' -> (0.5, 3.0)."""
    try:
        start, end = (float(part) for part in text.split("-"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"window must look like 0.5-3, got {text!r}") from error
    return start, end


def slow_envelope_uv(path: Path) -> tuple[list[str], float, np.ndarray]:
    """Channel names, sample rate and the slow envelope (n_channels, n) in uV of a session."""
    recording = find_recording(path, applied_config.CHANNELS)
    if recording is None:
        raise RecordingError(f"{path} is not a recording made by run_emg.py")
    _, emg = recording.load()
    rate = recording.sample_rate_hz
    processor = StreamingProcessor(applied_config.PROCESSING, emg.shape[0], rate)
    envelope = np.zeros_like(emg)
    for first in range(0, emg.shape[1], BLOCK):
        samples = np.ascontiguousarray(emg[:, first : first + BLOCK])
        block = processor.process(EMGBlock(samples, first, 0.0))
        envelope[:, first : first + samples.shape[1]] = block.slow_envelope
    return [channel.name for channel in recording.channels], rate, envelope * 1000.0


def flags(report: ChannelReport) -> str:
    """Short warnings for a channel."""
    notes = []
    if report.grips == 0:
        return "no grip found"
    if report.snr < MIN_SNR:
        notes.append(f"SNR < {MIN_SNR:g}")
    if report.spike_ratio > MAX_SPIKE:
        notes.append("artifact spikes")
    if report.repeatability is not None and report.repeatability > MAX_CV:
        notes.append("grips vary")
    return ", ".join(notes) or "ok"


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sessions", type=Path, nargs="+", help="one session folder per placement")
    parser.add_argument("--rest", type=parse_window, default=(0.5, 3.0),
                        help="rest window in seconds (default 0.5-3)")
    parser.add_argument("--factor", type=float, default=4.0,
                        help="a grip is the envelope above this x rest (default 4)")
    args = parser.parse_args(argv)

    print(f"{'session':44s} {'ch':4s} {'rest uV':>7s} {'drift':>6s} {'grip uV':>8s} {'SNR':>6s} "
          f"{'grips':>5s} {'spike':>6s} {'CV':>5s}  crosstalk (x rest)   notes")
    for path in args.sessions:
        try:
            names, rate, envelope = slow_envelope_uv(path)
            reports = analyse(envelope, names, rate, args.rest, args.factor)
        except (RecordingError, ValueError) as error:
            print(f"{path}: {error}", file=sys.stderr)
            return 2
        for report in reports:
            cv = "-" if report.repeatability is None else f"{report.repeatability:.2f}"
            cross = " ".join(f"{name} {value:.1f}" for name, value in report.crosstalk.items())
            print(f"{path.name[-44:]:44s} {report.name:4s} {report.rest:7.1f} {report.drift:6.2f} "
                  f"{report.active:8.0f} {report.snr:6.1f} {report.grips:5d} "
                  f"{report.spike_ratio:6.1f} {cv:>5s}  {cross:20s} {flags(report)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
