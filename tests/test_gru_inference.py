"""The real-time GRU interface in source/gru/inference.py (no PyTorch needed)."""

from pathlib import Path

import numpy as np

import settings
from settings import Channel
from source.gru.inference import GRUEstimator, load_model
from source.signal_processing import ProcessedBlock
from tests.fakes import FakeModel

TWO_SENSORS = [Channel("FDS", "flexor", 15), Channel("EDC", "extensor", 14)]
REST = np.array([0.004, 0.004])


def block(n: int = 200) -> ProcessedBlock:
    """A processed block of n samples; only `filtered` and `receive_time` matter here."""
    data = np.full((2, n), 0.01)
    return ProcessedBlock(data, data, data, data, data, data, data, 0, 123.0)


def warmed_up(model: FakeModel) -> GRUEstimator:
    """An estimator that has seen more than GRU_WARMUP_S of EMG."""
    estimator = GRUEstimator(model, TWO_SENSORS, 2000.0)
    estimator.update(block(2400), REST)
    return estimator


def test_estimate_is_valid_after_the_warm_up() -> None:
    """During GRU_WARMUP_S (1 s) the estimate is not valid; after it, it is."""
    estimator = GRUEstimator(FakeModel(), TWO_SENSORS, 2000.0)
    early = estimator.update(block(1000), REST)
    assert not early.valid and early.reason == "warming up"
    later = estimator.update(block(1400), REST)
    assert later.valid and later.reason == ""
    assert (later.intention, later.tremor_probability, later.receive_time) == (0.5, 0.1, 123.0)


def test_hidden_state_is_kept_between_blocks_and_cleared_by_reset() -> None:
    """Every call gets the hidden state of the call before; reset() starts from None."""
    model = FakeModel()
    estimator = GRUEstimator(model, TWO_SENSORS, 2000.0)
    estimator.update(block(), REST)
    estimator.update(block(), REST)
    assert model.hidden_seen[0] is None and model.hidden_seen[1] is not None
    estimator.reset()
    assert not estimator.update(block(), REST).valid  # warming up again
    assert model.hidden_seen[2] is None


def test_a_block_without_a_new_frame_repeats_the_last_estimate() -> None:
    """5 samples do not complete a frame: the last estimate is given with the new time."""
    estimator = warmed_up(FakeModel())
    tiny = block(5)
    tiny.receive_time = 124.0
    estimate = estimator.update(tiny, REST)
    assert estimate.valid and estimate.receive_time == 124.0


def test_one_sensor_model_uses_only_the_flexor_row() -> None:
    """A model for ["flexor"] gets 2 features per frame, taken from the flexor channel."""
    seen = []

    class Recorder(FakeModel):
        """Remembers the features it was given."""

        def step(self, frames: np.ndarray, hidden: object) -> tuple:
            seen.append(frames)
            return super().step(frames, hidden)

    data = block()
    data.filtered = np.vstack([np.full(200, 0.004), np.full(200, 0.5)])
    GRUEstimator(Recorder(["flexor"]), TWO_SENSORS, 2000.0).update(data, REST)
    assert seen[0].shape[1] == 2 and np.allclose(seen[0], np.log(2))


def test_a_model_that_does_not_fit_is_never_valid() -> None:
    """Missing sensor, another sample rate or other filters: not valid, with the reason."""
    one_sensor = [Channel("FDS", "flexor", 15)]
    needs_two = GRUEstimator(FakeModel(["flexor", "extensor"]), one_sensor, 2000.0)
    assert "needs a extensor sensor" in needs_two.update(block(), REST).reason

    wrong_rate = GRUEstimator(FakeModel(), TWO_SENSORS, 15 / 0.0135)
    assert "2000 Hz" in wrong_rate.update(block(), REST).reason

    model = FakeModel()
    model.config.filters["notch_hz"] = 60.0
    other_filters = GRUEstimator(model, TWO_SENSORS, 2000.0)
    assert "filter settings changed" in other_filters.update(block(), REST).reason


def test_a_value_that_is_not_a_number_disables_the_model() -> None:
    """NaN from the model: not valid until the calibration starts over."""
    model = FakeModel(intention=float("nan"))
    estimator = GRUEstimator(model, TWO_SENSORS, 2000.0)
    assert "not a number" in estimator.update(block(2400), REST).reason
    model.intention = 0.2
    assert not estimator.update(block(2400), REST).valid  # still disabled
    estimator.reset(forget_broken=True)
    assert estimator.update(block(2400), REST).valid


def test_missing_model_file_gives_a_message(tmp_path: Path) -> None:
    """load_model does not raise for a missing file; it says what to do."""
    model, message = load_model(tmp_path / "none.pt")
    assert model is None and "train_gru.py" in message
    assert settings.GRU_WARMUP_S > 0
