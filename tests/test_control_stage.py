"""The control stage inside StreamCore: calibration, controller states and the viewer output."""

from dataclasses import replace

import numpy as np
import pytest
from conftest import offline_synthetic

from source.acquisition import StreamCore
from source.config import AppConfig, ConfigError, ControllerConfig, validate_config
from source.controller import ControllerState
from source.sensor_mock import SyntheticSource
from source.viewer import ConsoleStatus, controller_lines, status_lines

RATE = 2000.0
BLOCK = 27


def run_until(core: StreamCore, source: SyntheticSource, seconds: float) -> None:
    """Feed blocks until stream time reaches `seconds`."""
    while True:
        block = source.read_block()
        assert block is not None
        core.handle_block(block)
        if (block.first_sample_index + block.n_samples) / RATE >= seconds:
            return


def test_synthetic_scenario_calibrates_arms_and_stiffens_with_tremor(config: AppConfig) -> None:
    """
    Synthetic timeline: rest 0-3 s, reference 3-5 s, rest 5-7 s, then rest 7-9 s,
    grasp 9-13 s, grasp + tremor 13-19 s, tremor subsides 19-21 s, rest 21-23 s.
    """
    source = offline_synthetic(config)
    source.start()
    core = StreamCore(config, RATE)

    run_until(core, source, 2.0)
    assert not core.control.calibrated
    assert core.control.calibrator.phase == "rest (baseline)"
    assert core.control.controller.state == ControllerState.DISARMED

    run_until(core, source, 8.9)
    assert core.control.calibrated and core.control.calibrator.error is None
    assert core.control.controller.state == ControllerState.IDLE
    assert source.segment_at(8.9).label == "rest"

    run_until(core, source, 12.9)
    assert source.segment_at(12.9).label == "grasp"
    assert core.control.controller.state == ControllerState.GRASP_ARMED
    assert core.control.controller.stiffness == 0.0

    run_until(core, source, 18.9)
    assert source.segment_at(18.9).label == "grasp + tremor"
    assert core.control.controller.state == ControllerState.STIFFENED
    assert core.control.controller.stiffness >= config.controller.engage_stiffness

    snapshot = core.snapshot(source, True, True)
    assert snapshot.state == "STIFFENED" and snapshot.armed
    assert snapshot.stiffness[-1] >= config.controller.engage_stiffness
    assert snapshot.grasp[-1] >= config.controller.grasp_on_threshold
    assert snapshot.tremor_control[-1] >= config.controller.tremor_on_threshold
    assert snapshot.control_time.shape == snapshot.stiffness.shape
    assert np.all(np.diff(snapshot.control_time) > 0)
    # The whole 5 s window of control history is kept, one entry per block
    assert snapshot.control_time[-1] - snapshot.control_time[0] >= 4.5

    run_until(core, source, 22.9)
    assert source.segment_at(22.9).label == "rest"
    assert core.control.controller.state == ControllerState.IDLE
    assert core.control.controller.stiffness == 0.0


def test_rest_tremor_without_grasp_does_not_stiffen(config: AppConfig) -> None:
    """Rest tremor without grasp (27-30 s of the synthetic scenario) leaves the stiffness at 0."""
    source = offline_synthetic(config)
    source.start()
    core = StreamCore(config, RATE)
    run_until(core, source, 25.0)
    stiffness = []
    while True:
        block = source.read_block()
        assert block is not None
        core.handle_block(block)
        stream_time = (block.first_sample_index + block.n_samples) / RATE
        if stream_time > 27.5:
            stiffness.append(core.control.controller.stiffness)
        if stream_time >= 29.9:
            break
    assert source.segment_at(29.9).label == "rest tremor, no grasp"
    assert core.control.controller.state in (ControllerState.IDLE, ControllerState.GRASP_ARMED)
    assert max(stiffness) == 0.0


def test_recalibration_disarms_and_calibrates_again(config: AppConfig) -> None:
    """A recalibration request takes effect on the next block; the controller re-arms after
    the hand relaxes."""
    source = offline_synthetic(config)
    source.start()
    core = StreamCore(config, RATE)
    run_until(core, source, 6.5)
    assert core.control.calibrated
    core.control.request_recalibration()
    run_until(core, source, 7.0)
    assert not core.control.calibrated
    assert core.control.controller.state == ControllerState.DISARMED
    assert core.snapshot(source, True, True).calibration_phase == "rest (baseline)"
    # From 6.5 s: the baseline window falls in rest (7-9 s) and the reference window
    # (9.8-11.2 s) in the synthetic grasp, which serves as the reference contraction.
    # The grasp goes on until 21 s, so the controller only arms after that.
    run_until(core, source, 11.5)
    assert core.control.calibrated
    assert core.control.controller.state == ControllerState.DISARMED
    assert core.control.phase == "relax to arm"
    run_until(core, source, 22.0)
    assert source.segment_at(22.0).label == "rest"
    assert core.control.controller.state != ControllerState.DISARMED
    assert core.control.phase == "done"


def test_controller_arms_only_after_the_reference_grip_is_released(config: AppConfig) -> None:
    """
    Calibration ends at 4.7 s while the synthetic reference grip lasts until 5 s: the
    controller stays disarmed until the hand has relaxed for grasp_off_time_s.
    """
    source = offline_synthetic(config)
    source.start()
    core = StreamCore(config, RATE)
    states = []
    while True:
        block = source.read_block()
        assert block is not None
        core.handle_block(block)
        stream_time = (block.first_sample_index + block.n_samples) / RATE
        if core.control.calibrated:
            states.append((stream_time, core.control.controller.state, core.control.phase))
        if stream_time >= 7.0:
            break

    calibrated_at = states[0][0]
    assert calibrated_at < 5.0 and source.segment_at(calibrated_at).label == "reference contraction"
    # Never in a grasp state, and disarmed ("relax to arm") until the grip is released
    assert all(state in (ControllerState.DISARMED, ControllerState.IDLE) for _, state, _ in states)
    armed_at = next(t for t, state, _ in states if state != ControllerState.DISARMED)
    assert armed_at >= 5.0 + config.controller.grasp_off_time_s
    assert all(phase == "relax to arm" for t, _, phase in states if t < armed_at)
    assert core.control.phase == "done"
    texts = [text for _, text in core.event_log()]
    assert "Calibration complete; relax the hand to arm the controller" in texts
    assert "Hand relaxed; controller armed" in texts


def test_failed_calibration_is_reported_once(config: AppConfig) -> None:
    """Calibrating on rest only (no reference contraction) fails once and stays disarmed."""
    failing = replace(
        config,
        calibration=replace(
            config.calibration, baseline_window_s=(0.2, 1.0), reference_window_s=(1.2, 2.0)
        ),
    )
    source = offline_synthetic(failing)
    source.start()
    core = StreamCore(failing, RATE)
    run_until(core, source, 3.0)
    assert not core.control.calibrated
    assert core.control.calibrator.error is not None
    assert core.control.controller.state == ControllerState.DISARMED
    messages = [text for _, text in core.event_log() if text.startswith("Calibration failed")]
    assert len(messages) == 1
    snapshot = core.snapshot(source, True, True)
    assert "FAILED" in controller_lines(snapshot)[-1]


def test_status_lines_show_the_controller(
    config: AppConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    """The side panel and the console line show state, stiffness and calibration."""
    source = offline_synthetic(config)
    source.start()
    core = StreamCore(config, RATE)
    run_until(core, source, 18.9)
    snapshot = core.snapshot(source, True, True)
    text = "\n".join(status_lines(snapshot))
    assert "Controller:  STIFFENED" in text and "stiffness 0." in text
    assert "Calibration: done" in text and "tremor depth" in text

    class FakeAcquisition:  # pylint: disable=too-few-public-methods
        """Stands in for Acquisition: only snapshot() is used."""

        def snapshot(self) -> object:
            """The snapshot above."""
            return snapshot

    ConsoleStatus(FakeAcquisition())()  # type: ignore[arg-type]
    assert "STIFFENED" in capsys.readouterr().out


def test_controller_config_is_validated(config: AppConfig) -> None:
    """Inconsistent thresholds and unknown tremor measures are rejected."""
    for bad in (
        ControllerConfig(grasp_off_threshold=0.3),
        ControllerConfig(tremor_off_threshold=0.2),
        ControllerConfig(tremor_measure="power"),
        ControllerConfig(max_stiffness=1.5),
    ):
        with pytest.raises(ConfigError):
            validate_config(replace(config, controller=bad))
    with pytest.raises(ConfigError):
        validate_config(
            replace(
                config,
                calibration=replace(
                    config.calibration, baseline_window_s=(0.0, 4.0), reference_window_s=(3.0, 6.0)
                ),
            )
        )
