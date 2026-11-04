"""
The applied configuration: the settings values in use. The blueprint (setting
types, defaults and validation) is source/config.py; edit the values here.
"""

from collections.abc import Sequence

from source.config import (
    AppConfig,
    CalibrationConfig,
    ChannelConfig,
    ControllerConfig,
    ProcessingConfig,
    RecordedConfig,
    RuntimeConfig,
    SyntheticConfig,
    TrignoConfig,
    validate_config,
)

# EMG channels. Each channel has a name, a role and where to find it:
#   role             "flexor", "extensor" or "other". Flexor channels drive grasp
#                    detection (all channels when no flexor is configured); the
#                    synthetic source gives flexors and extensors different activity.
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

# Calibration windows in seconds of stream time from the start: rest during the
# baseline window, make the reference grip (firm, not maximal) during the
# reference window. One entry per source; the synthetic scenario and the dataset
# have fixed timelines, the Trigno windows are what the user follows live.
CALIBRATION = {
    "trigno": CalibrationConfig(
        baseline_window_s=(0.5, 3.0),
        reference_window_s=(3.5, 6.5),
        reference_percentile=95.0,  # PROVISIONAL: of the slow envelope in the reference window
        min_reference_ratio=3.0,  # PROVISIONAL: the reference must exceed rest this many times
    ),
    # Synthetic scenario (sensor_mock.py): rest 0-3 s, reference contraction 3-5 s.
    "synthetic": CalibrationConfig(baseline_window_s=(0.5, 2.5), reference_window_s=(3.3, 4.7)),
    # Dataset timeline (paper Fig. 4): rest 0-4 s, then gestures until 104 s.
    "recorded": CalibrationConfig(baseline_window_s=(0.5, 3.5), reference_window_s=(4.0, 104.0)),
}

# Controller (design doc 8.1 option C and 8.4): IDLE -> GRASP_ARMED -> STIFFENED.
# Grasp activation is normalized: 0 = rest baseline, 1 = reference grip. The
# tremor thresholds apply to the tremor depth (~0.05 without tremor at any effort,
# ~0.3 or more with tremor), or to the tremor level with tremor_measure="level".
# The stiffness (0 = released, 1 = maximum) is the control signal; the viewer
# shows it, nothing physical is driven. All values PROVISIONAL.
CONTROLLER = ControllerConfig(
    grasp_on_threshold=0.25,
    grasp_off_threshold=0.15,  # hysteresis: must be < on
    grasp_on_time_s=0.10,
    grasp_off_time_s=0.30,
    tremor_measure="depth",
    tremor_on_threshold=0.15,
    tremor_off_threshold=0.08,  # hysteresis: must be < on
    # Engage only when the tremor-band fluctuation is rhythmic: ~1 for a narrow
    # 4-6 Hz peak, ~0.5 for broadband voluntary fluctuation. 0 disables the gate.
    tremor_min_ratio=0.8,
    tremor_off_ratio=0.6,  # release when the ratio stays below this; 0 = amplitude only
    tremor_on_time_s=0.05,  # design doc 8.4: engage within tens of ms
    tremor_off_time_s=0.75,  # design doc 8.4: release over 0.5-1 s
    engage_stiffness=0.3,  # stiffness when tremor is first detected
    stiffness_gain=2.0,  # extra stiffness per unit tremor above the off threshold
    max_stiffness=1.0,  # command ceiling, at most 1.0
    engage_rate_per_s=20.0,  # max increase of the command per second
    release_rate_per_s=1.5,  # max decrease per second (~0.7 s from 1 to 0)
)

# Live view and reconnection.
RUNTIME = RuntimeConfig(
    monitor_seconds=5.0,  # seconds of EMG shown in the live view
    max_reconnect_attempts=3,  # after a lost connection
    reconnect_delay_s=1.0,
)


def build_config(
    channels: Sequence[ChannelConfig] | None = None, source: str = "synthetic"
) -> AppConfig:
    """
    The settings above as one validated configuration: CHANNELS unless given,
    and the calibration windows of `source` (synthetic, recorded or trigno).
    """
    if source not in CALIBRATION:
        raise ValueError(f"Unknown source {source!r}; expected one of {tuple(CALIBRATION)}")
    config = AppConfig(
        channels=tuple(CHANNELS if channels is None else channels),
        trigno=TRIGNO,
        recorded=RECORDED,
        synthetic=SYNTHETIC,
        processing=PROCESSING,
        runtime=RUNTIME,
        calibration=CALIBRATION[source],
        controller=CONTROLLER,
    )
    validate_config(config)
    return config
