"""A saved calibration: saved after a successful routine, loaded instead of it on later runs."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import offline_synthetic

import run_emg
from source.acquisition import StreamCore
from source.calibration_store import DEFAULT_PATH, CalibrationStore
from source.config import AppConfig
from source.controller import ControllerState
from source.sensor_mock import SyntheticSource

RATE = 2000.0


def run_until(core: StreamCore, source: SyntheticSource, seconds: float) -> None:
    """Feed blocks until stream time reaches `seconds`."""
    while True:
        block = source.read_block()
        assert block is not None
        core.handle_block(block)
        if (block.first_sample_index + block.n_samples) / RATE >= seconds:
            return


def messages(core: StreamCore) -> list[str]:
    """The event log's texts."""
    return [text for _, text in core.event_log()]


def calibrated_run(config: AppConfig, store: CalibrationStore, seconds: float) -> StreamCore:
    """A fresh synthetic run with the store, fed for `seconds`."""
    source = offline_synthetic(config)
    source.start()
    core = StreamCore(config, RATE, calibration_store=store)
    run_until(core, source, seconds)
    return core


def test_calibration_is_saved_then_loaded_instead_of_calibrating(
    config: AppConfig, tmp_path: Path
) -> None:
    """Run 1 calibrates and saves; run 2 starts calibrated on its first block and arms
    when the hand is relaxed, without the rest/reference routine."""
    store = CalibrationStore(tmp_path / "cal.json")
    first = calibrated_run(config, store, 8.9)
    assert first.control.calibrated and (tmp_path / "cal.json").is_file()
    assert any(text.startswith("Calibration saved to") for text in messages(first))

    second = calibrated_run(config, store, 0.5)
    assert second.control.calibrated  # long before the 3-5 s reference window
    assert second.control.calibrator.error is None
    assert any(text.startswith("Saved calibration loaded") for text in messages(second))
    assert any(text.startswith("Using the saved calibration") for text in messages(second))
    assert second.control.controller.state == ControllerState.IDLE  # armed: the hand is at rest
    assert not any(text.startswith("Calibration saved") for text in messages(second))
    levels = second.control.calibrator.result
    assert levels is not None and levels.channel_names == config.channel_names


def test_use_saved_false_calibrates_again_and_replaces_the_file(
    config: AppConfig, tmp_path: Path
) -> None:
    """--calibrate: an existing file is ignored, and the new calibration is saved over it."""
    path = tmp_path / "cal.json"
    calibrated_run(config, CalibrationStore(path), 8.9)
    first_saved = json.loads(path.read_text(encoding="utf-8"))["saved_at"]
    core = calibrated_run(config, CalibrationStore(path, use_saved=False), 2.0)
    assert not core.control.calibrated  # the routine is running
    core = calibrated_run(config, CalibrationStore(path, use_saved=False), 8.9)
    assert core.control.calibrated
    assert any(text.startswith("Calibration saved to") for text in messages(core))
    assert json.loads(path.read_text(encoding="utf-8"))["saved_at"] >= first_saved


def test_recalibration_request_discards_the_loaded_calibration(
    config: AppConfig, tmp_path: Path
) -> None:
    """The 'c' key discards a loaded calibration: the controller disarms and the routine starts."""
    store = CalibrationStore(tmp_path / "cal.json")
    calibrated_run(config, store, 8.9)
    source = offline_synthetic(config)
    source.start()
    core = StreamCore(config, RATE, calibration_store=store)
    run_until(core, source, 0.5)
    assert core.control.calibrated
    core.control.request_recalibration()
    run_until(core, source, 1.0)
    assert not core.control.calibrated
    assert core.control.controller.state == ControllerState.DISARMED


def test_a_saved_calibration_that_does_not_fit_is_ignored(
    config: AppConfig, tmp_path: Path
) -> None:
    """Other channels, changed processing settings, a corrupt file or a weak reference level:
    the file is ignored with a message and the routine runs."""
    path = tmp_path / "cal.json"
    calibrated_run(config, CalibrationStore(path), 8.9)
    good = json.loads(path.read_text(encoding="utf-8"))

    def ignored_with(data: object, cfg: AppConfig = config) -> str:
        """Write `data` as the saved file, start a run, and return the 'ignored' message."""
        path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
        core = calibrated_run(cfg, CalibrationStore(path), 0.5)
        assert not core.control.calibrated  # the routine runs instead
        return next(text for text in messages(core) if text.startswith("Saved calibration ignored"))

    assert "channels" in ignored_with({**good, "channel_names": ["A", "B", "C", "D"]})
    changed = replace(config, processing=replace(config.processing, notch_hz=60.0))
    assert "processing settings changed" in ignored_with(good, changed)
    assert "Saved calibration ignored" in ignored_with("{not json")
    assert "not at least" in ignored_with({**good, "reference_slow": good["rest_slow"]})


def test_a_failed_save_is_reported_and_the_run_goes_on(
    config: AppConfig, tmp_path: Path
) -> None:
    """An unwritable path does not stop the controller from arming."""
    blocked = tmp_path / "a-file"
    blocked.write_text("x", encoding="utf-8")
    core = calibrated_run(config, CalibrationStore(blocked / "cal.json"), 8.9)
    assert core.control.calibrated
    assert any(text.startswith("Could not save the calibration") for text in messages(core))


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["--source", "trigno"], (DEFAULT_PATH, True)),
        (["--source", "trigno", "--calibrate"], (DEFAULT_PATH, False)),
        (["--source", "synthetic", "--calibration", "x.json"], (Path("x.json"), True)),
    ],
)
def test_run_emg_calibration_options(argv: list[str], expected: tuple[Path, bool]) -> None:
    """Trigno keeps a calibration by default, --calibrate ignores the saved one."""
    store = run_emg.calibration_store(run_emg.parse_args(argv))
    assert store is not None and (store.path, store.use_saved) == expected


def test_run_emg_synthetic_and_recorded_calibrate_every_run_by_default() -> None:
    """Only the live sensors keep a calibration unless a file is given."""
    assert run_emg.calibration_store(run_emg.parse_args(["--source", "synthetic"])) is None
    assert run_emg.calibration_store(run_emg.parse_args(["--source", "recorded"])) is None
