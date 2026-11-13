"""The GRU network and its model file in source/gru/model.py (needs PyTorch)."""

from pathlib import Path

import numpy as np
import pytest

from settings import Channel
from source.gru.config import GRUConfig
from source.gru.inference import GRUEstimator, load_model

torch = pytest.importorskip("torch")

# pylint: disable=wrong-import-position
from source.gru.model import IntentionTremorGRU, load_checkpoint, save_checkpoint  # noqa: E402


def make_model(roles: list[str]) -> IntentionTremorGRU:
    """A model with random weights."""
    torch.manual_seed(0)
    return IntentionTremorGRU(GRUConfig(channel_roles=roles))


@pytest.mark.parametrize("roles", [["flexor"], ["flexor", "extensor"]])
def test_output_shapes_and_ranges(roles: list[str]) -> None:
    """One or two sensors: (batch, time) outputs, intention in -1..+1, probability in 0..1."""
    model = make_model(roles)
    features = torch.randn(3, 50, 2 * len(roles)) * 5
    intention, tremor_logit, hidden = model(features)
    assert intention.shape == (3, 50) and tremor_logit.shape == (3, 50)
    assert hidden.shape == (1, 3, 64)  # (layers, batch, hidden units)
    assert intention.abs().max() <= 1.0
    probability = torch.sigmoid(tremor_logit)
    assert probability.min() >= 0.0 and probability.max() <= 1.0


def test_streaming_gives_the_same_as_the_whole_sequence() -> None:
    """Feeding frames a few at a time, keeping the hidden state, equals one long call."""
    model = make_model(["flexor", "extensor"])
    frames = np.random.default_rng(0).random((60, 4)).astype(np.float32)
    whole_intention, whole_probability, _ = model.step(frames, None)

    hidden = None
    intention, probability = [], []
    for start in range(0, 60, 7):
        part_i, part_p, hidden = model.step(frames[start:start + 7], hidden)
        intention.append(part_i)
        probability.append(part_p)
    assert np.allclose(np.concatenate(intention), whole_intention, atol=1e-6)
    assert np.allclose(np.concatenate(probability), whole_probability, atol=1e-6)


def test_an_empty_memory_gives_a_different_answer() -> None:
    """The hidden state matters: starting from None is not the same as continuing."""
    model = make_model(["flexor"])
    frames = np.ones((20, 2), dtype=np.float32)
    _, _, hidden = model.step(frames, None)
    continued, _, _ = model.step(frames, hidden)
    restarted, _, _ = model.step(frames, None)
    assert not np.allclose(continued, restarted)


def test_model_file_round_trip(tmp_path: Path) -> None:
    """A saved model loads with the same configuration and gives the same outputs."""
    model = make_model(["flexor", "extensor"])
    model.config.data_kind = "real"
    model.config.trained_on = "test"
    path = tmp_path / "gru.pt"
    save_checkpoint(path, model)
    loaded = load_checkpoint(path)
    assert loaded.config == model.config
    frames = np.random.default_rng(1).random((10, 4)).astype(np.float32)
    assert np.allclose(loaded.step(frames, None)[0], model.step(frames, None)[0])

    found, message = load_model(path)
    assert found is not None and "loaded (test)" in message


def test_a_loaded_model_refuses_the_wrong_sensors(tmp_path: Path) -> None:
    """A two-sensor model is not valid with one sensor."""
    path = tmp_path / "gru.pt"
    save_checkpoint(path, make_model(["flexor", "extensor"]))
    estimator = GRUEstimator(load_checkpoint(path), [Channel("FDS", "flexor", 15)], 2000.0)
    assert estimator.problem is not None and "extensor" in estimator.problem
