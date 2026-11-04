"""
The configuration blueprint: setting types, defaults and validation for live
EMG acquisition and the controller. The values in use are in applied_config.py;
validate_config() rejects inconsistent combinations.
"""

from dataclasses import dataclass

CHANNEL_ROLES = ("flexor", "extensor", "other")
SOURCE_KINDS = ("synthetic", "recorded", "trigno")
TREMOR_MEASURES = ("depth", "level")


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
class CalibrationConfig:
    """
    Rest baseline and reference contraction windows, in seconds of stream time
    from the start (or from the last recalibration). The reference is a
    submaximal "firm grip" (RVC), not a maximum contraction.
    """

    baseline_window_s: tuple[float, float] = (0.5, 3.0)
    reference_window_s: tuple[float, float] = (3.5, 6.5)
    reference_percentile: float = 95.0
    min_reference_ratio: float = 3.0


@dataclass(frozen=True)
class ControllerConfig:  # pylint: disable=too-many-instance-attributes
    """
    State machine thresholds and stiffness shaping. Grasp activation is
    normalized (0 = rest, 1 = reference grip). tremor_measure picks what the
    tremor thresholds apply to: "depth" (tremor amplitude / slow envelope,
    independent of effort) or "level" (tremor amplitude / reference span).
    """

    grasp_on_threshold: float = 0.25
    grasp_off_threshold: float = 0.15
    grasp_on_time_s: float = 0.10
    grasp_off_time_s: float = 0.30
    tremor_measure: str = "depth"
    tremor_on_threshold: float = 0.15
    tremor_off_threshold: float = 0.08
    tremor_min_ratio: float = 0.8
    tremor_off_ratio: float = 0.6
    tremor_on_time_s: float = 0.05
    tremor_off_time_s: float = 0.75
    engage_stiffness: float = 0.3
    stiffness_gain: float = 2.0
    max_stiffness: float = 1.0
    engage_rate_per_s: float = 20.0
    release_rate_per_s: float = 1.5


@dataclass(frozen=True)
class RuntimeConfig:
    """Live view and reconnection."""

    monitor_seconds: float = 5.0
    max_reconnect_attempts: int = 3
    reconnect_delay_s: float = 1.0


@dataclass(frozen=True)
class AppConfig:  # pylint: disable=too-many-instance-attributes
    """The complete configuration."""

    channels: tuple[ChannelConfig, ...]
    trigno: TrignoConfig
    recorded: RecordedConfig
    synthetic: SyntheticConfig
    processing: ProcessingConfig
    runtime: RuntimeConfig
    calibration: CalibrationConfig = CalibrationConfig()
    controller: ControllerConfig = ControllerConfig()

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
    _validate_channels(config)
    _validate_processing(config)
    _validate_calibration(config.calibration)
    _validate_controller(config.controller)

    _check(config.runtime.monitor_seconds > 0, "monitor_seconds must be positive")
    _check(config.runtime.max_reconnect_attempts >= 0, "max_reconnect_attempts must be >= 0")
    _check(config.runtime.reconnect_delay_s >= 0, "reconnect_delay_s must be >= 0")


def _validate_channels(config: AppConfig) -> None:
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


def _validate_processing(config: AppConfig) -> None:
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


def _validate_calibration(calibration: CalibrationConfig) -> None:
    for window in (calibration.baseline_window_s, calibration.reference_window_s):
        _check(
            len(window) == 2 and 0 <= window[0] < window[1],
            "Calibration windows must be (start, end) with 0 <= start < end",
        )
    _check(
        calibration.baseline_window_s[1] <= calibration.reference_window_s[0],
        "The baseline window must end before the reference window starts",
    )
    _check(
        0 < calibration.reference_percentile <= 100,
        "reference_percentile must be in (0, 100]",
    )
    _check(calibration.min_reference_ratio > 1, "min_reference_ratio must be above 1")


def _validate_controller(controller: ControllerConfig) -> None:
    _check(
        controller.tremor_measure in TREMOR_MEASURES,
        f"controller.tremor_measure must be one of {TREMOR_MEASURES}",
    )
    _check(
        controller.grasp_off_threshold < controller.grasp_on_threshold,
        "grasp_off_threshold must be below grasp_on_threshold (hysteresis)",
    )
    _check(
        controller.tremor_off_threshold < controller.tremor_on_threshold,
        "tremor_off_threshold must be below tremor_on_threshold (hysteresis)",
    )
    _check(0 <= controller.tremor_min_ratio <= 1.0, "tremor_min_ratio must be in [0, 1]")
    _check(
        controller.tremor_off_ratio <= controller.tremor_min_ratio,
        "tremor_off_ratio must not exceed tremor_min_ratio (hysteresis)",
    )
    _check(0 < controller.max_stiffness <= 1.0, "max_stiffness must be in (0, 1]")
    _check(
        0 <= controller.engage_stiffness <= controller.max_stiffness,
        "engage_stiffness must be within [0, max_stiffness]",
    )
    _check(
        controller.engage_rate_per_s > 0 and controller.release_rate_per_s > 0,
        "Stiffness rate limits must be positive",
    )
    for name in ("grasp_on_time_s", "grasp_off_time_s", "tremor_on_time_s", "tremor_off_time_s"):
        _check(getattr(controller, name) >= 0, f"controller.{name} must be >= 0")
