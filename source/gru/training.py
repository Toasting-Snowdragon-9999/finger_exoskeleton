"""
Training the GRU (needs PyTorch).

  recordings with labels
    -> recording_frames (labels.py): the same filter and features as live, labels every 10 ms
    -> make_sequences: pieces of GRU_SEQUENCE_SECONDS, the training examples
    -> train: show the examples to the network many times (epochs); after each
       example the weights are nudged so the outputs come closer to the labels

Two things keep the result honest:
  - split_by_group puts whole recordings (or people) in training, validation or
    test. Pieces of one recording never end up on both sides, because
    neighbouring pieces are almost identical and would make the test too easy.
  - Unknown labels (NaN) are left out of the loss ("masked"). The network is
    never trained toward an invented answer.

The loss is the number that training makes small:
  intention   smooth L1: about the distance between the output and the label
  tremor      binary cross-entropy: small when the probability is near the label
"""

from collections.abc import Callable

import numpy as np
import torch
from torch import nn

import settings
from source.gru.config import GRUConfig
from source.gru.labels import LabelledRecording, recording_frames
from source.gru.model import IntentionTremorGRU


def make_sequences(recordings: list[LabelledRecording], config: GRUConfig,
                   sequence_seconds: float, step_seconds: float = 1.0
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Training examples: features (n, length, n_features) and the intention and
    tremor labels (n, length), cut from the recordings every step_seconds.
    Pieces without a single known label are left out.
    """
    length = round(sequence_seconds * config.feature_rate_hz)
    step = round(step_seconds * config.feature_rate_hz)
    features, intentions, tremors = [], [], []
    for recording in recordings:
        frames, intention, tremor = recording_frames(recording, config)
        for start in range(0, len(frames) - length + 1, step):
            piece = slice(start, start + length)
            if np.all(np.isnan(intention[piece])) and np.all(np.isnan(tremor[piece])):
                continue
            features.append(frames[piece])
            intentions.append(intention[piece])
            tremors.append(tremor[piece])
    if not features:
        raise ValueError("No training examples: the recordings are too short or have no labels")
    return (torch.tensor(np.array(features), dtype=torch.float32),
            torch.tensor(np.array(intentions), dtype=torch.float32),
            torch.tensor(np.array(tremors), dtype=torch.float32))


def split_by_group(recordings: list[LabelledRecording], seed: int = 0
                   ) -> tuple[list[LabelledRecording], list[LabelledRecording],
                              list[LabelledRecording]]:
    """
    (training, validation, test): about 60 / 20 / 20 % of the groups. Every
    recording of a group goes to the same part.
    """
    groups = sorted({recording.group for recording in recordings})
    if len(groups) < 3:
        raise ValueError(f"Need at least 3 independent recordings or people (one each for "
                         f"training, validation and test); got {len(groups)}: {groups}")
    order = np.random.default_rng(seed).permutation(len(groups))
    n_held = max(1, round(0.2 * len(groups)))
    test = {groups[i] for i in order[:n_held]}
    validation = {groups[i] for i in order[n_held:2 * n_held]}
    return ([r for r in recordings if r.group not in test | validation],
            [r for r in recordings if r.group in validation],
            [r for r in recordings if r.group in test])


def masked_losses(model: IntentionTremorGRU, features: torch.Tensor, intention: torch.Tensor,
                  tremor: torch.Tensor, burn_in: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    (intention loss, tremor loss) over the time steps whose label is known.
    The first burn_in steps of every example are not scored: the GRU starts
    each example with an empty memory, which never happens in a long live run.
    """
    predicted_intention, tremor_logit, _ = model(features)
    scored = torch.ones_like(intention, dtype=torch.bool)
    scored[:, :burn_in] = False
    known_intention = scored & ~torch.isnan(intention)
    known_tremor = scored & ~torch.isnan(tremor)

    # Without a single known label the loss is 0 (times the output, so it is still a tensor
    # PyTorch can take the gradient of: a gradient of zero, which changes nothing)
    intention_loss = predicted_intention.sum() * 0.0
    if known_intention.any():
        intention_loss = nn.functional.smooth_l1_loss(predicted_intention[known_intention],
                                                      intention[known_intention])
    tremor_loss = tremor_logit.sum() * 0.0
    if known_tremor.any():
        tremor_loss = nn.functional.binary_cross_entropy_with_logits(tremor_logit[known_tremor],
                                                                     tremor[known_tremor])
    return intention_loss, tremor_loss


def train_one_epoch(model: IntentionTremorGRU, optimizer: torch.optim.Optimizer,
                    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor], burn_in: int) -> float:
    """Show every training example to the network once; returns the mean loss."""
    model.train()
    features, intention, tremor = data
    order = torch.randperm(len(features))  # a new order of the examples every epoch
    total = 0.0
    for start in range(0, len(order), settings.GRU_BATCH_SIZE):
        batch = order[start:start + settings.GRU_BATCH_SIZE]
        intention_loss, tremor_loss = masked_losses(
            model, features[batch], intention[batch], tremor[batch], burn_in)
        loss = intention_loss + tremor_loss
        optimizer.zero_grad()  # forget the gradients of the previous batch
        loss.backward()  # how should every weight change to make the loss smaller?
        optimizer.step()  # change the weights a little in that direction
        total += loss.item() * len(batch)
    return total / len(order)


def train(training: list[LabelledRecording], validation: list[LabelledRecording],
          config: GRUConfig, epochs: int = settings.GRU_EPOCHS,
          learning_rate: float = settings.GRU_LEARNING_RATE,
          sequence_seconds: float = settings.GRU_SEQUENCE_SECONDS,
          log: Callable[[str], None] = print) -> tuple[IntentionTremorGRU, list[dict]]:
    """
    Train a new model. After every epoch the loss on the validation recordings is
    measured; the weights with the lowest validation loss are kept, so the
    model does not just memorize the training recordings. Returns the model
    and the losses per epoch. Call torch.manual_seed(...) first for a repeatable result.
    """
    burn_in = round(settings.GRU_BURN_IN_SECONDS * config.feature_rate_hz)
    train_data = make_sequences(training, config, sequence_seconds)
    validation_data = make_sequences(validation, config, sequence_seconds)
    model = IntentionTremorGRU(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float("inf")
    best_weights = model.state_dict()
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_data, burn_in)
        model.eval()
        with torch.no_grad():  # only measuring: no gradients needed
            intention_loss, tremor_loss = masked_losses(model, *validation_data, burn_in)
        validation_loss = (intention_loss + tremor_loss).item()
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "validation_loss": validation_loss})
        log(f"epoch {epoch:3d}: training loss {train_loss:.4f}, "
            f"validation loss {validation_loss:.4f}")
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_weights = {name: value.clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_weights)
    model.eval()
    return model, history
