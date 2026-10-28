"""
The applied configuration: the settings values in use. The blueprint (setting
types, defaults and validation) is source/config.py; edit the values here.
"""

from collections.abc import Sequence

from source.config import (
    AppConfig,
    ChannelConfig,
    ProcessingConfig,
    RecordedConfig,
    RuntimeConfig,
    SyntheticConfig,
    TrignoConfig,
    validate_config,
)

# EMG channels. Each channel has a name, a role and where to find it:
#   role             "flexor", "extensor" or "other"; only the synthetic source uses
#                    it (flexors and extensors get different simulated activity).
#   trigno_slot      TCU sensor slot (1-16) the sensor is paired to.
#   recorded_column  Column index in the recorded CSV. The dataset (source/import_data.py)
#                    has 0 = ECU, 1 = FCU, 2 = ECR, 3 = FCR, recorded from these forearm
#                    muscles, so the names are only labels there.

# One sensor, as paired in the TCU for the first hardware tests.
ONE_SENSOR = (ChannelConfig(name="FDS", role="flexor", trigno_slot=15, recorded_column=1),)

# The planned montage (design doc 7.2 / 8.3), sensors paired to slots 1-4.
FOUR_SENSORS = (
    ChannelConfig(name="FDS", role="flexor", trigno_slot=1, recorded_column=1),  # dataset FCU
    ChannelConfig(name="EDC", role="extensor", trigno_slot=2, recorded_column=0),  # dataset ECU
    ChannelConfig(name="FDI", role="flexor", trigno_slot=3, recorded_column=3),  # dataset FCR
    ChannelConfig(name="APB", role="extensor", trigno_slot=4, recorded_column=2),  # dataset ECR
)

CHANNELS = ONE_SENSOR  # the channels in use: ONE_SENSOR or FOUR_SENSORS

# Live Delsys Trigno (DS-T03 / SP-W02 / SP-W06, Trigno Avanti) through the
# Trigno Control Utility (TCU) TCP/IP SDK, Delsys MAN-025-3-5.
TRIGNO = TrignoConfig(
    host="127.0.0.1",
    command_port=50040,  # SDK 6.1
    emg_port=50043,  # SDK 6.1: EMG Data port, 16 channels
    channels_on_port=16,  # SDK 6.1: data channels on the EMG Data port
    connect_timeout_s=3.0,
    command_timeout_s=5.0,  # TCU 3.6.0 takes ~2.6 s to reply to STOP (seen on hardware)
    data_timeout_s=0.5,  # no data for this long -> SourceTimeoutError
    # SDK 6.1.3 shows STARTINDEX as 0-based but the appendix example as 1-based.
    # Verified on hardware (TCU 3.6.0): the sensor in slot 4 reports STARTINDEX 4
    # and its data is in 0-based column 3.
    start_index_base=1,
    required_units="V",  # TCU 3.6.0 replies "V" to "SENSOR n CHANNEL 1 UNITS?"
    expected_sample_rate_hz=0.0,  # if > 0, refuse to start at any other EMG sample rate
)

# Recorded EMG (CSV, one column per channel, in mV).
RECORDED = RecordedConfig(
    file="10_raw.csv",  # relative to data/sEMG_online, or an absolute path
    sample_rate_hz=2000.0,  # dataset paper: 2 kHz
    block_size=27,  # samples per block, matching a Trigno 13.5 ms frame at 2 kHz
    loop=True,
    realtime=True,
)

# Synthetic EMG (no hardware). Scenario in source/sensor_mock.py: rest, reference
# contraction, then repeating cycles of grasp, grasp with 5 Hz tremor, extension
# without grasp and rest tremor without grasp.
SYNTHETIC = SyntheticConfig(
    sample_rate_hz=2000.0,
    block_size=27,
    seed=1,
    realtime=True,
    rest_noise_mv=0.005,  # RMS at rest
    max_contraction_mv=0.4,  # RMS at full drive
    tremor_frequency_hz=5.0,  # design doc 2.1: PD tremor 4-6 Hz
    powerline_mv=0.002,  # 50 Hz interference amplitude
    powerline_hz=50.0,
)

# Streaming signal processing (design doc 7.4). All filters are causal. The live
# view shows the filtered EMG and its moving-RMS envelope (slow_window_s).
PROCESSING = ProcessingConfig(
    bandpass_low_hz=20.0,
    bandpass_high_hz=450.0,
    bandpass_order=4,
    notch_enabled=True,
    notch_hz=50.0,  # Europe; 60.0 in the Americas
    notch_quality=30.0,
    slow_window_s=0.2,  # envelope, design doc 7.4: 150-250 ms
    fast_cutoff_hz=10.0,  # tremor envelope, design doc 7.4: 8-10 Hz low-pass
    tremor_band_hz=(4.0, 6.0),  # design doc 8.1 option C: 4-6 Hz band power
    tremor_window_s=0.5,  # PROVISIONAL
    # Envelope fluctuation band for the tremor ratio (rhythmicity, design doc 7.7):
    # ratio = 4-6 Hz amplitude / 1-9 Hz amplitude of the fast envelope.
    rhythm_band_hz=(1.0, 9.0),  # PROVISIONAL
    # Tremor depth = tremor amplitude / slow envelope; below this envelope (rest) the
    # floor is used instead so noise is not amplified. About 2x the resting RMS.
    depth_min_envelope_mv=0.01,  # PROVISIONAL
    frequency_window_s=2.0,  # dominant frequency of the fast envelope over this window
)

# Live view and reconnection.
RUNTIME = RuntimeConfig(
    monitor_seconds=5.0,  # seconds of EMG shown in the live view
    max_reconnect_attempts=3,  # after a lost connection
    reconnect_delay_s=1.0,
)


def build_config(channels: Sequence[ChannelConfig] | None = None) -> AppConfig:
    """The settings above as one validated configuration (CHANNELS unless given)."""
    config = AppConfig(
        channels=tuple(CHANNELS if channels is None else channels),
        trigno=TRIGNO,
        recorded=RECORDED,
        synthetic=SYNTHETIC,
        processing=PROCESSING,
        runtime=RUNTIME,
    )
    validate_config(config)
    return config
