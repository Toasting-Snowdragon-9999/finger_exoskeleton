"""
The settings a GRU model is built and trained with. They are saved inside the
model file, so a model is always used with the features it was trained on.
"""

from dataclasses import dataclass, field

import settings

# The signal processing settings that change the filtered EMG the GRU sees
FILTER_KEYS = ("bandpass_low_hz", "bandpass_high_hz", "bandpass_order",
               "notch_enabled", "notch_hz", "notch_quality", "slow_window_s")


def filter_settings() -> dict:
    """The current filter settings that matter for the GRU."""
    processing = settings.processing_settings()
    return {key: processing[key] for key in FILTER_KEYS}


@dataclass
class GRUConfig:
    """Everything needed to rebuild a model and its input features."""

    channel_roles: list[str]  # e.g. ["flexor"] or ["flexor", "extensor"], in input order
    feature_rate_hz: float = settings.GRU_FEATURE_RATE_HZ
    feature_window_s: float = settings.GRU_FEATURE_WINDOW_S
    hidden_units: int = settings.GRU_HIDDEN_UNITS
    sample_rate_hz: float = 2000.0  # EMG sample rate the features are made for
    filters: dict = field(default_factory=filter_settings)
    data_kind: str = "synthetic"  # "synthetic" = software test only, "real" = real recordings
    trained_on: str = ""  # description of the training data
    training_groups: list[str] = field(default_factory=list)  # recordings used for training
    metrics: dict = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        """Inputs per time step: RMS and MAV of every channel."""
        return 2 * len(self.channel_roles)
