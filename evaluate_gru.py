"""
Test a trained GRU model, and the signal processing (DSP), on recordings with labels.

Examples:
  python evaluate_gru.py --model models/gru.pt --sessions measurements/<folder> ...
  python evaluate_gru.py --model models/gru.pt --npz data/converted/pinheiro_*.npz
  python evaluate_gru.py --model models/gru.pt --smoke-test     (made-up data: software test)

Recordings that were used to train the model are left out: a test on them would
look far better than the model really is. See GRU_README.md for what the
numbers mean.

Needs PyTorch: pip install -r requirements-ml.txt
"""

import argparse
import json
from pathlib import Path

from source.gru.evaluation import evaluate, format_results
from source.gru.labels import load_npz, load_session
from source.gru.model import load_checkpoint
from source.gru.synthetic import add_tremor, synthetic_recording


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command-line options."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="the model file to test")
    parser.add_argument("--sessions", nargs="*", type=Path, default=[],
                        help="session folders with a labels.csv")
    parser.add_argument("--calibration", type=Path,
                        help="the calibration file that was used for these sessions")
    parser.add_argument("--npz", nargs="*", type=Path, default=[],
                        help="converted dataset files (see source/gru/labels.py)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="test on new made-up recordings: a software test only")
    parser.add_argument("--out", type=Path, help="also write the results to this JSON file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load the model and the recordings, print the comparison."""
    args = parse_args(argv)
    model = load_checkpoint(args.model)
    config = model.config
    print(f"Model {args.model}: sensors {config.channel_roles}, {config.hidden_units} hidden "
          f"units, trained on {config.trained_on or 'unknown data'} ({config.data_kind})")

    recordings = [load_session(folder, args.calibration) for folder in args.sessions]
    recordings += [load_npz(path) for path in args.npz]
    if args.smoke_test:  # other seeds than train_gru.py --smoke-test uses
        recordings += [add_tremor(synthetic_recording(seed, config.channel_roles), seed)
                       for seed in range(100, 104)]
    seen = [recording for recording in recordings if recording.group in config.training_groups]
    for recording in seen:
        print(f"Left out: {recording.name} was used to train this model")
    recordings = [recording for recording in recordings if recording not in seen]
    if not recordings:
        print("No recordings to test on.")
        return 1

    results = evaluate(model, recordings)
    print(f"\nOn {len(recordings)} recordings the model has not seen:")
    print(format_results(results))
    if args.out is not None:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
