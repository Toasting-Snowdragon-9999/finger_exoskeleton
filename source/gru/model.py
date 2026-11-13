"""
The GRU network (needs PyTorch).

  features (RMS and MAV per channel, every 10 ms)
      -> GRU, one layer of 64 units          the memory: what happened the last seconds
          -> intention head: Linear -> tanh      one number in -1 ... +1
          -> tremor head:    Linear -> sigmoid   one number in 0 ... 1 (a probability)

A GRU ("gated recurrent unit") reads the features one time step after the other.
At every step it combines the new features with its hidden state (64 numbers that
summarize the past) and produces a new hidden state. The two "heads" are small
linear layers that turn those 64 numbers into the two outputs.

The model file holds the weights and the GRUConfig the model was trained with.
"""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from source.gru.config import GRUConfig


class IntentionTremorGRU(nn.Module):
    """One GRU layer with an intention head and a tremor head."""

    def __init__(self, config: GRUConfig) -> None:
        super().__init__()
        self.config = config
        # batch_first: inputs are (batch, time, features)
        self.gru = nn.GRU(config.n_features, config.hidden_units, batch_first=True)
        self.intention_head = nn.Linear(config.hidden_units, 1)
        self.tremor_head = nn.Linear(config.hidden_units, 1)

    def forward(self, features: torch.Tensor, hidden: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        features: (batch, time, n_features). hidden: the memory to start from, or
        None for an empty memory. Returns for every time step the intention
        (batch, time) in -1...+1 and the tremor LOGIT (batch, time), and the new
        hidden state. A logit is the number before the sigmoid: 0 means 50 %,
        positive means "tremor more likely than not". Training uses the logit
        (it is numerically safer); sigmoid(logit) is the probability.
        """
        memory, hidden = self.gru(features, hidden)  # memory: (batch, time, hidden_units)
        intention = torch.tanh(self.intention_head(memory)).squeeze(-1)  # tanh keeps it in -1..+1
        tremor_logit = self.tremor_head(memory).squeeze(-1)
        return intention, tremor_logit, hidden

    def step(self, frames: np.ndarray, hidden: torch.Tensor | None
             ) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """
        Real-time use: feed the new (n_frames, n_features) frames, starting from the
        hidden state of the previous call (None = empty memory). Returns the
        intention and the tremor probability of every frame, and the hidden
        state to pass in next time.
        """
        self.eval()
        with torch.inference_mode():  # no training here, so no gradients are needed
            features = torch.as_tensor(frames, dtype=torch.float32).unsqueeze(0)  # batch of 1
            intention, tremor_logit, hidden = self(features, hidden)
            probability = torch.sigmoid(tremor_logit)
        return intention[0].numpy(), probability[0].numpy(), hidden


def save_checkpoint(path: Path, model: IntentionTremorGRU) -> None:
    """Write the weights and the configuration to a model file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "weights": model.state_dict()}, path)


def load_checkpoint(path: Path) -> IntentionTremorGRU:
    """Read a model file written by save_checkpoint, ready for step()."""
    # weights_only=True: the file can only contain numbers and text, never code
    content = torch.load(path, map_location="cpu", weights_only=True)
    model = IntentionTremorGRU(GRUConfig(**content["config"]))
    model.load_state_dict(content["weights"])
    model.eval()
    torch.set_num_threads(1)  # a network this small is fastest on one CPU core
    return model
