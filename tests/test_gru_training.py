"""Training the GRU in source/gru/training.py (needs PyTorch)."""

from dataclasses import replace

import numpy as np
import pytest

from source.gru.config import GRUConfig
from source.gru.synthetic import add_tremor, synthetic_recording

torch = pytest.importorskip("torch")

# pylint: disable=wrong-import-position
from source.gru.model import IntentionTremorGRU  # noqa: E402
from source.gru.labels import recording_frames  # noqa: E402
from source.gru.training import make_sequences, masked_losses, split_by_group, train  # noqa: E402

TWO = ["flexor", "extensor"]


def test_frames_have_features_and_a_label_every_10_ms() -> None:
    """A 30 s recording gives about 3000 frames with 4 features and one label pair each."""
    recording = synthetic_recording(0, TWO, seconds=30.0)
    features, intention, tremor = recording_frames(recording, GRUConfig(channel_roles=TWO))
    assert features.shape == (len(intention), 4) and len(intention) == len(tremor)
    assert abs(len(features) - recording.emg.shape[1] / 20) < 10
    assert np.isfinite(features).all() and features.min() >= 0.0
    assert set(np.unique(intention)) <= {-1.0, 0.0, 0.5, 1.0}


def test_one_flexor_sensor_trains_opening_as_no_grasp() -> None:
    """With the flexor only, an opening label (-1) becomes 0: it cannot be told from rest."""
    recording = synthetic_recording(0, TWO, seconds=30.0)
    assert recording.intention.min() == -1.0
    features, intention, _ = recording_frames(recording, GRUConfig(channel_roles=["flexor"]))
    assert features.shape[1] == 2 and intention.min() == 0.0


def test_sequences_skip_pieces_without_any_label() -> None:
    """Examples are (n, 400, features); a recording without labels gives none."""
    config = GRUConfig(channel_roles=TWO)
    recording = synthetic_recording(1, TWO, seconds=20.0)
    features, intention, tremor = make_sequences([recording], config, sequence_seconds=4.0)
    assert features.shape[1:] == (400, 4) and intention.shape == tremor.shape == features.shape[:2]
    unlabelled = replace(recording, intention=np.full_like(recording.intention, np.nan),
                         tremor=np.full_like(recording.tremor, np.nan))
    with pytest.raises(ValueError, match="No training examples"):
        make_sequences([unlabelled], config, sequence_seconds=4.0)


def test_split_keeps_groups_together() -> None:
    """No group is in two parts, a copy with tremor follows its original, and 2 groups fail."""
    recordings = [synthetic_recording(seed, TWO, seconds=15.0) for seed in range(6)]
    recordings += [add_tremor(recording, seed=9) for recording in recordings]
    parts = split_by_group(recordings)
    groups = [{recording.group for recording in part} for part in parts]
    assert all(groups) and not groups[0] & groups[1] and not groups[0] & groups[2]
    assert not groups[1] & groups[2] and sum(len(part) for part in parts) == 12
    with pytest.raises(ValueError, match="at least 3"):
        split_by_group(recordings[:2])


def test_unknown_labels_are_not_trained_on() -> None:
    """All intention labels unknown: the intention loss is 0 and its head gets no gradient."""
    torch.manual_seed(0)
    model = IntentionTremorGRU(GRUConfig(channel_roles=TWO))
    features = torch.randn(4, 100, 4)
    unknown = torch.full((4, 100), float("nan"))
    tremor = torch.ones(4, 100)
    intention_loss, tremor_loss = masked_losses(model, features, unknown, tremor, burn_in=50)
    assert intention_loss.item() == 0.0 and tremor_loss.item() > 0.0
    (intention_loss + tremor_loss).backward()
    intention_gradient = model.intention_head.weight.grad
    tremor_gradient = model.tremor_head.weight.grad
    assert intention_gradient is not None and intention_gradient.abs().sum() == 0
    assert tremor_gradient is not None and tremor_gradient.abs().sum() > 0


def test_the_start_of_every_example_is_not_scored() -> None:
    """Labels inside the burn-in do not count: a wrong label there changes nothing."""
    torch.manual_seed(0)
    model = IntentionTremorGRU(GRUConfig(channel_roles=TWO))
    features = torch.randn(2, 100, 4)
    intention = torch.zeros(2, 100)
    changed = intention.clone()
    changed[:, :50] = 1.0
    tremor = torch.zeros(2, 100)
    first = masked_losses(model, features, intention, tremor, burn_in=50)[0]
    second = masked_losses(model, features, changed, tremor, burn_in=50)[0]
    assert first.item() == second.item()


def test_smoke_training_on_synthetic_data_learns_something() -> None:
    """SOFTWARE TEST ONLY: on made-up data the loss goes down. Says nothing about real EMG."""
    recordings = [add_tremor(synthetic_recording(seed, TWO, seconds=30.0), seed)
                  for seed in range(4)]
    config = GRUConfig(channel_roles=TWO)
    torch.manual_seed(0)
    model, history = train(recordings[:3], recordings[3:], config, epochs=6,
                           learning_rate=0.01, log=lambda _: None)
    assert history[-1]["train_loss"] < 0.9 * history[0]["train_loss"]
    assert all(np.isfinite(step["validation_loss"]) for step in history)
    assert model.config.data_kind == "synthetic"
