"""The per-block pipeline in source/pipeline.py: calibrate, arm after relaxing, recalibrate."""

import shutil
from pathlib import Path

import numpy as np

import settings
from source.calibration import Calibration
from source.gru.inference import GRUEstimator
from source.pipeline import Pipeline
from source.recording import RecordedSource
from source.signal_processing import EMGBlock
from source.stiffness_controller import State
from tests.fakes import RATE, FakeModel, make_emg

CHANNELS = settings.CHANNELS
ROOT = Path(__file__).resolve().parent.parent


def feed(pipeline: Pipeline, signal: np.ndarray, start_index: int = 0) -> None:
    """Give a (2, n) signal to the pipeline in blocks of 27 samples."""
    for start in range(0, signal.shape[1], 27):
        pipeline.step(EMGBlock(signal[:, start:start + 27], start_index + start, 0.0))


def test_a_grip_held_after_calibration_does_not_arm() -> None:
    """Calibration ends at 10 s while the grip lasts until 11 s: it arms only after relaxing."""
    pipeline = Pipeline(CHANNELS, RATE)
    signal = make_emg([(3.5, 0.01), (7.5, 0.2), (2.0, 0.01)])
    feed(pipeline, signal[:, :round(10.5 * RATE)])
    assert pipeline.calibration is not None
    assert pipeline.state == State.DISARMED and pipeline.phase == "relax to arm"
    feed(pipeline, signal[:, round(10.5 * RATE):], start_index=round(10.5 * RATE))
    assert pipeline.state == State.IDLE and pipeline.phase == "done"
    assert any(text == "Hand relaxed; controller armed" for _, text in pipeline.messages)


def test_start_delay_moves_the_calibration_routine() -> None:
    """With a start delay the rest and grip windows begin that much later."""
    pipeline = Pipeline(CHANNELS, RATE, start_delay_s=2.0)
    signal = make_emg([(2.0 + 3.5, 0.01), (6.5, 0.2), (2.0, 0.01)])
    feed(pipeline, signal[:, :round(1.5 * RATE)])
    assert pipeline.phase == "waiting to start" and pipeline.calibrator.start_index is None
    feed(pipeline, signal[:, round(1.5 * RATE):], start_index=round(1.5 * RATE))
    assert pipeline.calibration is not None and pipeline.state == State.IDLE
    assert np.all(pipeline.calibration.rest_slow < 0.02)  # the rest window saw rest only

    without_delay = Pipeline(CHANNELS, RATE)  # the same signal: the grip comes 2 s "late"
    feed(without_delay, signal)
    assert without_delay.calibration is not None
    assert pipeline.calibration.reference_slow[0] >= without_delay.calibration.reference_slow[0]


def test_a_saved_calibration_does_not_wait_for_the_start_delay(tmp_path: Path) -> None:
    """The delay is for the calibration routine; a loaded calibration is used at once."""
    path = tmp_path / "calibration.json"
    levels = np.array([0.01, 0.01]), np.array([0.2, 0.2])
    Calibration(["FDS", "EDC"], levels[0], levels[1], levels[0], levels[1]).save(path)
    pipeline = Pipeline(CHANNELS, RATE, calibration_file=path, start_delay_s=20.0)
    feed(pipeline, make_emg([(0.5, 0.01)]))
    assert pipeline.state == State.IDLE and not pipeline.waiting_to_start


def test_a_saved_calibration_is_used_and_arms_after_relaxing(tmp_path: Path) -> None:
    """With a calibration file no routine runs; resting for 0.3 s arms the controller."""
    path = tmp_path / "calibration.json"
    levels = np.array([0.01, 0.01]), np.array([0.2, 0.2])
    Calibration(["FDS", "EDC"], levels[0], levels[1], levels[0], levels[1]).save(path)
    pipeline = Pipeline(CHANNELS, RATE, calibration_file=path)
    feed(pipeline, make_emg([(0.5, 0.01)]))
    assert pipeline.state == State.IDLE
    assert any(text.startswith("Using the saved calibration") for _, text in pipeline.messages)


def test_the_bench_recording_gives_the_same_states_as_before(tmp_path: Path) -> None:
    """
    Replaying the committed recording with synthetic tremor gives exactly the state changes
    of the code before the simplification (sample number at the end of the block, new state).
    """
    calibration = tmp_path / "calibration.json"
    shutil.copy(ROOT / "calibration_replay.json", calibration)
    folder = ROOT / "measurements" / "20261105-144738_trigno_to-lange-greb_tremor"
    source = RecordedSource(folder, loop=False, realtime=False)
    pipeline = Pipeline(source.channels, source.sample_rate, calibration)
    changes: list[tuple[int, str]] = []
    while (block := source.read_block()) is not None:
        pipeline.step(block)
        if not changes or changes[-1][1] != pipeline.state.value:
            changes.append((block.first_index + block.samples.shape[1], pipeline.state.value))
    assert changes == [(27, "DISARMED"), (648, "IDLE"), (16011, "GRASP_ARMED"),
                       (17415, "STIFFENED"), (22950, "IDLE"), (32751, "GRASP_ARMED"),
                       (36504, "IDLE")]


def test_recalibrate_disarms_and_starts_over() -> None:
    """recalibrate() disarms on the next block and runs the calibration routine again."""
    pipeline = Pipeline(CHANNELS, RATE)
    signal = make_emg([(3.5, 0.01), (6.5, 0.2), (1.0, 0.01)])
    feed(pipeline, signal)
    assert pipeline.state == State.IDLE
    pipeline.recalibrate()
    feed(pipeline, make_emg([(0.1, 0.01)]), start_index=signal.shape[1])
    assert pipeline.state == State.DISARMED and pipeline.calibration is None
    assert pipeline.phase == "rest (baseline)"


# --- with a GRU estimator (--method gru) -------------------------------------------------------

REST_THEN_GRIP = [(3.5, 0.01), (6.5, 0.2), (2.0, 0.01)]  # calibrates and arms at about 10.4 s


def gru_pipeline(model: FakeModel) -> Pipeline:
    """A pipeline that asks a (fake) GRU model."""
    return Pipeline(CHANNELS, RATE, estimator=GRUEstimator(model, CHANNELS, RATE))


def test_gru_drives_the_controller_when_valid() -> None:
    """A GRU that says "closing, tremor" brings the controller to STIFFENED without EMG tremor."""
    pipeline = gru_pipeline(FakeModel(intention=0.8, probability=0.9))
    feed(pipeline, make_emg(REST_THEN_GRIP + [(1.0, 0.01)]))
    assert pipeline.using_gru and pipeline.state == State.STIFFENED
    assert pipeline.activation is not None and pipeline.activation.grasp == 0.8
    assert pipeline.activation.tremor_probability == 0.9
    assert pipeline.dsp_activation is not None and pipeline.dsp_activation.grasp < 0.1
    recent = pipeline.recent()
    assert recent["gru_intention"][-1] == 0.8 and recent["gru_probability"][-1] == 0.9


def test_arming_follows_the_signal_processing_not_the_gru() -> None:
    """A GRU stuck at "no grasp" must not arm the controller while the grip is still held."""
    pipeline = gru_pipeline(FakeModel(intention=0.0))
    signal = make_emg(REST_THEN_GRIP)
    feed(pipeline, signal[:, :round(10.2 * RATE)])
    assert pipeline.calibration is not None and pipeline.state == State.DISARMED
    feed(pipeline, signal[:, round(10.2 * RATE):], start_index=round(10.2 * RATE))
    assert pipeline.state == State.IDLE


def test_opening_intention_is_no_grasp_for_the_position_controller() -> None:
    """A negative intention (open) gives grasp 0."""
    pipeline = gru_pipeline(FakeModel(intention=-0.7))
    feed(pipeline, make_emg(REST_THEN_GRIP + [(1.0, 0.01)]))
    assert pipeline.using_gru and pipeline.activation is not None
    assert pipeline.activation.grasp == 0.0 and pipeline.state == State.IDLE


def test_an_invalid_gru_falls_back_at_once_and_says_so_once() -> None:
    """A model for a missing sensor is never used; the reason is reported one time."""
    model = FakeModel(["flexor", "other"])
    pipeline = gru_pipeline(model)
    feed(pipeline, make_emg(REST_THEN_GRIP))
    assert not pipeline.using_gru and pipeline.activation is pipeline.dsp_activation
    notes = [text for _, text in pipeline.messages if text.startswith("GRU not used")]
    assert len(notes) == 1 and "needs a other sensor" in notes[0]


def test_gru_takes_over_only_while_the_finger_is_open() -> None:
    """After an interruption during a grasp the DSP keeps control until the grasp ends."""
    pipeline = gru_pipeline(FakeModel(intention=0.8))
    signal = make_emg(REST_THEN_GRIP + [(1.0, 0.01), (3.0, 0.2), (1.0, 0.01)])
    grip_start = round(13.5 * RATE)
    feed(pipeline, signal[:, :grip_start])
    assert pipeline.using_gru and pipeline.state == State.GRASP_ARMED
    pipeline.signal_interrupted()  # the EMG stalled: the GRU has to warm up again
    feed(pipeline, signal[:, grip_start:grip_start + round(2.0 * RATE)], start_index=grip_start)
    assert not pipeline.using_gru and pipeline.state == State.GRASP_ARMED  # DSP sees the grip
    feed(pipeline, signal[:, grip_start + round(2.0 * RATE):],
         start_index=grip_start + round(2.0 * RATE))
    assert pipeline.using_gru  # back to the GRU once the DSP grasp ended (IDLE)


def test_recalibration_resets_the_gru() -> None:
    """Recalibrating clears the GRU's memory and stops using it until armed again."""
    pipeline = gru_pipeline(FakeModel())
    signal = make_emg(REST_THEN_GRIP)
    feed(pipeline, signal)
    assert pipeline.using_gru
    pipeline.recalibrate()
    feed(pipeline, make_emg([(0.1, 0.01)]), start_index=signal.shape[1])
    assert not pipeline.using_gru and pipeline.gru_estimate is None
    assert pipeline.estimator is not None and pipeline.estimator.hidden is None
