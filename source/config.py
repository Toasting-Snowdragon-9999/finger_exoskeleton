"""
The configuration blueprint: setting types, defaults and validation for live
EMG acquisition. The values in use are in applied_config.py; validate_config()
rejects inconsistent combinations.
"""

from dataclasses import dataclass

CHANNEL_ROLES = ("flexor", "extensor", "other")
SOURCE_KINDS = ("synthetic", "recorded", "trigno")


class ConfigError(ValueError):
    """Raised for an inconsistent configuration."""


@dataclass(frozen=True)
class ChannelConfig:
    """One EMG channel: its name, muscle role and where each source finds it."""

    name: str
    role: str
    trigno_slot: int
    recorded_column: int


@dataclass(frozen=True)
class TrignoConfig:
    """Delsys Trigno Control Utility TCP/IP SDK settings (MAN-025-3-5)."""

    host: str = "127.0.0.1"
    command_port: int = 50040
    emg_port: int = 50043
    channels_on_port: int = 16
    connect_timeout_s: float = 3.0
    command_timeout_s: float = 5.0
    data_timeout_s: float = 0.5
    start_index_base: int = 1
    required_units: str = "V"
    expected_sample_rate_hz: float = 0.0


@dataclass(frozen=True)
class RecordedConfig:
    """Replay of a recorded CSV file."""

    file: str = "10_raw.csv"
    sample_rate_hz: float = 2000.0
    block_size: int = 27
    loop: bool = True
    realtime: bool = True


@dataclass(frozen=True)
class SyntheticConfig:
    """Synthetic EMG generator."""

    sample_rate_hz: float = 2000.0
    block_size: int = 27
    seed: int = 1
    realtime: bool = True
    rest_noise_mv: float = 0.005
    max_contraction_mv: float = 0.4
    tremor_frequency_hz: float = 5.0
    powerline_mv: float = 0.002
    powerline_hz: float = 50.0


@dataclass(frozen=True)
class ProcessingConfig:
    """Streaming (causal) signal processing."""

    bandpass_low_hz: float = 20.0
    bandpass_high_hz: float = 450.0
    bandpass_order: int = 4
    notch_enabled: bool = True
    notch_hz: float = 50.0
    notch_quality: float = 30.0
    slow_window_s: float = 2.0
    fast_cutoff_hz: float = 10.0
    tremor_band_hz: tuple[float, float] = (4.0, 6.0)
    tremor_window_s: float = 0.5
    rhythm_band_hz: tuple[float, float] = (1.0, 9.0)
    depth_min_envelope_mv: float = 0.01
    frequency_window_s: float = 2.0


@dataclass(frozen=True)
class RuntimeConfig:
    """Live view and reconnection."""

    monitor_seconds: float = 5.0
    max_reconnect_attempts: int = 3
    reconnect_delay_s: float = 1.0


@dataclass(frozen=True)
class AppConfig:
    """The complete configuration."""

    channels: tuple[ChannelConfig, ...]
    trigno: TrignoConfig
    recorded: RecordedConfig
    synthetic: SyntheticConfig
    processing: ProcessingConfig
    runtime: RuntimeConfig

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Channel names in order."""
        return tuple(channel.name for channel in self.channels)

    @property
    def channel_roles(self) -> tuple[str, ...]:
        """Channel roles in order."""
        return tuple(channel.role for channel in self.channels)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def validate_config(config: AppConfig) -> None:
    """Reject inconsistent configurations."""
    names = config.channel_names
    _check(len(names) > 0, "At least one [[channels]] entry is required")
    _check(len(set(names)) == len(names), "Channel names must be unique")
    _check(
        len(names) <= config.trigno.channels_on_port,
        "More channels than the Trigno EMG port carries",
    )
    for channel in config.channels:
        _check(
            channel.role in CHANNEL_ROLES,
            f"Channel {channel.name}: role must be one of {CHANNEL_ROLES}",
        )
        _check(1 <= channel.trigno_slot <= 16, f"Channel {channel.name}: trigno_slot must be 1-16")
        _check(
            channel.recorded_column >= 0, f"Channel {channel.name}: recorded_column must be >= 0"
        )
    slots = [channel.trigno_slot for channel in config.channels]
    _check(len(set(slots)) == len(slots), "Each channel needs its own trigno_slot")

    _check(config.trigno.start_index_base in (0, 1), "trigno.start_index_base must be 0 or 1")

    processing = config.processing
    _check(
        0 < processing.bandpass_low_hz < processing.bandpass_high_hz,
        "processing.bandpass_low_hz must be below bandpass_high_hz",
    )
    _check(
        len(processing.tremor_band_hz) == 2
        and 0 < processing.tremor_band_hz[0] < processing.tremor_band_hz[1],
        "processing.tremor_band_hz must be [low, high]",
    )
    _check(
        processing.tremor_band_hz[1] < processing.fast_cutoff_hz,
        "The tremor band must lie below the fast envelope cutoff",
    )
    _check(
        len(processing.rhythm_band_hz) == 2
        and 0 < processing.rhythm_band_hz[0] <= processing.tremor_band_hz[0]
        and processing.tremor_band_hz[1]
        <= processing.rhythm_band_hz[1]
        < processing.fast_cutoff_hz,
        "processing.rhythm_band_hz must contain the tremor band and lie below fast_cutoff_hz",
    )
    _check(processing.depth_min_envelope_mv > 0, "depth_min_envelope_mv must be positive")
    _check(
        0 < processing.frequency_window_s <= config.runtime.monitor_seconds,
        "frequency_window_s must be positive and at most runtime.monitor_seconds",
    )

    _check(config.runtime.monitor_seconds > 0, "monitor_seconds must be positive")
    _check(config.runtime.max_reconnect_attempts >= 0, "max_reconnect_attempts must be >= 0")
    _check(config.runtime.reconnect_delay_s >= 0, "reconnect_delay_s must be >= 0")
