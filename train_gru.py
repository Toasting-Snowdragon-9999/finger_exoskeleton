"""
Train a GRU model (the alternative to the signal processing) and save it.

Examples:
  python train_gru.py --smoke-test
      SOFTWARE TEST ONLY: trains on made-up data to check that everything runs.
  python train_gru.py --sessions measurements/<folder> measurements/<folder> ...
      our own recordings with a labels.csv (made with run_emg.py --cues)
  python train_gru.py --sessions ... --add-synthetic-tremor
      also lay synthetic tremor over the recordings, to get tremor examples
  python train_gru.py --npz data/converted/*.npz --roles flexor extensor
      converted public datasets, for a two-sensor model

Whole recordings (or people) are set aside for validation and for the test, so
at least 3 independent recordings are needed. The model is saved to --out
together with the settings it was trained with; a .json file next to it holds
the results. See GRU_README.md.

Needs PyTorch: pip install -r requirements-ml.txt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import settings
from source.gru.config import GRUConfig
from source.gru.evaluation import evaluate, format_results
from source.gru.labels import LabelledRecording, load_npz, load_session
from source.gru.model import save_checkpoint
from source.gru.synthetic import add_tremor, synthetic_recording
from source.gru.training import split_by_group, train


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command-line options."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions", nargs="*", type=Path, default=[],
                        help="session folders with a labels.csv")
    parser.add_argument("--calibration", type=Path,
                        help="the calibration file that was used for these sessions")
    parser.add_argument("--npz", nargs="*", type=Path, default=[],
                        help="converted dataset files (see source/gru/labels.py)")
    parser.add_argument("--add-synthetic-tremor", action="store_true",
                        help="add copies of the tremor-free recordings with synthetic tremor")
    parser.add_argument("--smoke-test", action="store_true",
                        help="train on made-up data: a software test, not a usable model")
    parser.add_argument("--roles", nargs="+", default=settings.GRU_CHANNEL_ROLES,
                        help="the sensors the model uses, e.g. flexor, or flexor extensor")
    parser.add_argument("--hidden-units", type=int, default=settings.GRU_HIDDEN_UNITS)
    parser.add_argument("--learning-rate", type=float, default=settings.GRU_LEARNING_RATE)
    parser.add_argument("--sequence-seconds", type=float, default=settings.GRU_SEQUENCE_SECONDS)
    parser.add_argument("--epochs", type=int, default=settings.GRU_EPOCHS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(settings.GRU_MODEL_FILE))
    return parser.parse_args(argv)


def with_synthetic_tremor(recordings: list[LabelledRecording], seed: int
                          ) -> list[LabelledRecording]:
    """The recordings, plus a copy with synthetic tremor of every one known to be tremor-free."""
    copies = [add_tremor(recording, seed + number) for number, recording in enumerate(recordings)
              if np.nanmax(recording.tremor, initial=0.0) == 0
              and not np.all(np.isnan(recording.tremor))]
    return recordings + copies


def main(argv: list[str] | None = None) -> int:
    """Load the data, train, test on the held-out recordings and save the model."""
    args = parse_args(argv)
    if args.smoke_test:
        print("SMOKE TEST on made-up data. The model that comes out is NOT usable on real EMG.")
        recordings = [synthetic_recording(seed, args.roles) for seed in range(8)]
        args.add_synthetic_tremor = True
    else:
        recordings = [load_session(folder, args.calibration) for folder in args.sessions]
        recordings += [load_npz(path) for path in args.npz]
    training, validation, test = split_by_group(recordings, args.seed)
    if args.add_synthetic_tremor:  # after the split, so a copy never leaves its original
        training, validation, test = (with_synthetic_tremor(part, args.seed + 100 * number)
                                      for number, part in enumerate((training, validation, test)))
    for name, part in (("training", training), ("validation", validation), ("test", test)):
        print(f"{name:10s} {len(part):3d} recordings: {sorted({r.group for r in part})}")

    config = GRUConfig(
        channel_roles=list(args.roles), hidden_units=args.hidden_units,
        data_kind="synthetic" if args.smoke_test else "real",
        trained_on=f"{len(training)} recordings" + (", tremor labels from synthetic tremor"
                                                    if args.add_synthetic_tremor else ""),
        training_groups=sorted({recording.group for recording in training + validation}))
    torch.manual_seed(args.seed)  # the same seed gives the same model again
    model, history = train(training, validation, config, epochs=args.epochs,
                           learning_rate=args.learning_rate,
                           sequence_seconds=args.sequence_seconds)

    results = evaluate(model, test)
    print(f"\nOn the {len(test)} held-out test recordings:")
    print(format_results(results))
    model.config.metrics = {"test": results, "history": history}
    save_checkpoint(args.out, model)
    args.out.with_suffix(".json").write_text(json.dumps(model.config.metrics, indent=2),
                                             encoding="utf-8")
    print(f"\nModel saved to {args.out} (results in {args.out.with_suffix('.json')})")
    if args.smoke_test:
        print("Remember: this model was trained on made-up data. It is a software test only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
