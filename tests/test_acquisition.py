"""Live acquisition: buffering, gap and NaN handling, recording, threads, viewer and CLI."""

import csv
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from conftest import offline_synthetic

import applied_config
import run_emg
from source.acquisition import Acquisition, EMGRecorder, RingBuffer, StreamCore
from source.config import AppConfig, RecordedConfig
from source.import_data import load_columns
from source.mock_tcu import MockSensor, MockTCU, MockTCUSettings
from source.emg_source import EMGBlock
from source.sensor_mock import RecordedSource, SyntheticSource
from source.trigno import TrignoSource
from source.viewer import ConsoleStatus, LiveViewer, status_lines

RATE = 2000.0
BLOCK = 27


def blocks_from(source: SyntheticSource, n_blocks: int) -> list[EMGBlock]:
    """n_blocks consecutive blocks from a started offline source."""
    blocks = []
    for _ in range(n_blocks):
        block = source.read_block()
        assert block is not None
        blocks.append(block)
    return blocks


def noise_block(first_index: int, n_channels: int = 4, receive_time: float = 0.0) -> EMGBlock:
    """A block of small random EMG."""
    rng = np.random.default_rng(first_index)
    return EMGBlock(rng.standard_normal((n_channels, BLOCK)) * 0.01, first_index, receive_time)


def wait_for(condition: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Poll until condition() is true."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def fast_reconnect(config: AppConfig) -> AppConfig:
    """Shorter reconnection delay for tests."""
    return replace(config, runtime=replace(config.runtime, reconnect_delay_s=0.05))


# -- StreamCore (no threads) ------------------------------------------------------


def test_ring_buffer_keeps_latest() -> None:
    """The viewer buffers keep the newest values in order and never grow."""
    buffer = RingBuffer(1, 5)
    buffer.write(np.array([[1.0, 2.0, 3.0]]))
    buffer.write(np.array([[4.0, 5.0, 6.0, 7.0]]))
    np.testing.assert_array_equal(buffer.read(), [[3.0, 4.0, 5.0, 6.0, 7.0]])
    buffer.write(np.arange(12.0)[None, :])
    np.testing.assert_array_equal(buffer.read(), [[7.0, 8.0, 9.0, 10.0, 11.0]])


def test_core_buffers_raw_filtered_and_envelope(config: AppConfig) -> None:
    """The last monitor_seconds of raw, filtered and envelope signals are kept, in mV."""
    source = offline_synthetic(config)
    source.start()
    core = StreamCore(config, RATE)
    blocks = blocks_from(source, 600)  # 8.1 s, more than the 5 s window
    for block in blocks:
        core.handle_block(block)
    snapshot = core.snapshot(source, connected=True, running=True)

    capacity = int(config.runtime.monitor_seconds * RATE)
    for signal in (
        snapshot.filtered,
        snapshot.envelope,
        snapshot.fast_envelope,
        snapshot.tremor,
        snapshot.tremor_ratio,
        snapshot.tremor_depth,
    ):
        assert signal.shape == snapshot.raw.shape == (4, capacity)
        assert np.all(np.isfinite(signal))
    expected_raw = np.hstack([block.samples for block in blocks])[:, -capacity:]
    np.testing.assert_allclose(snapshot.raw, expected_raw)
    np.testing.assert_allclose(np.diff(snapshot.time), 1 / RATE)
    assert snapshot.time[-1] == pytest.approx((600 * BLOCK - 1) / RATE)
    assert np.all(snapshot.envelope >= 0) and np.all(snapshot.fast_envelope >= -1e-9)
    assert np.all(snapshot.tremor_ratio >= 0)
    assert snapshot.counters["blocks"] == 600


def test_dominant_frequency_matches_synthetic_tremor(config: AppConfig) -> None:
    """During synthetic grasp + tremor, every channel's fast envelope peaks at the tremor rate."""
    source = offline_synthetic(config)
    source.start()
    core = StreamCore(config, RATE)
    for block in blocks_from(source, int(18.0 * RATE / BLOCK)):  # grasp + tremor: 13-19 s
        core.handle_block(block)
    assert source.segment_at(17.9).label == "grasp + tremor"
    snapshot = core.snapshot(source, True, True)
    expected = config.synthetic.tremor_frequency_hz
    np.testing.assert_allclose(snapshot.dominant_hz, expected, atol=0.3)
    assert np.all(snapshot.tremor_depth[:, -1] > 0.2)


def test_gaps_and_index_resets_are_counted(config: AppConfig) -> None:
    """Missing samples and backwards jumps in the sample index are counted."""
    core = StreamCore(config, RATE)
    for first in (0, BLOCK, 3 * BLOCK + 10, 0):
        core.handle_block(noise_block(first))
    counters = core.snapshot(offline_synthetic(config), True, True).counters
    assert counters["missing samples"] == BLOCK + 10
    assert counters["gaps"] == 1
    assert counters["index resets"] == 1


def test_new_session_restarts_gap_detection(config: AppConfig) -> None:
    """After a reconnection the next index is not a gap or a reset."""
    core = StreamCore(config, RATE)
    core.handle_block(noise_block(5000))
    core.new_session()
    core.handle_block(noise_block(0))
    counters = core.snapshot(offline_synthetic(config), True, True).counters
    assert "gaps" not in counters and "index resets" not in counters


def test_invalid_samples_do_not_poison_filters(config: AppConfig) -> None:
    """A NaN block is counted and zeroed; filtering continues normally afterwards."""
    core = StreamCore(config, RATE)
    core.handle_block(noise_block(0))
    core.handle_block(EMGBlock(np.full((4, BLOCK), np.nan), BLOCK, 0.0))
    core.handle_block(noise_block(2 * BLOCK))
    snapshot = core.snapshot(offline_synthetic(config), True, True)
    assert snapshot.counters["invalid blocks"] == 1
    assert np.all(np.isfinite(snapshot.filtered)) and np.all(np.isfinite(snapshot.envelope))


def test_measured_rate_follows_arrival_times(config: AppConfig) -> None:
    """The measured rate is samples per second of receive time."""
    core = StreamCore(config, RATE)
    for i in range(100):
        core.handle_block(noise_block(i * BLOCK, receive_time=i * BLOCK / RATE))
    snapshot = core.snapshot(offline_synthetic(config), True, True)
    assert snapshot.measured_rate == pytest.approx(RATE)


def test_recorder_writes_raw_samples(config: AppConfig, tmp_path: Path) -> None:
    """The CSV has a header, a time column and the raw samples in mV, as received."""
    path = tmp_path / "sub" / "session.csv"
    recorder = EMGRecorder(path, config.channel_names, RATE)
    core = StreamCore(config, RATE, recorder)
    blocks = [noise_block(0), noise_block(BLOCK)]
    for block in blocks:
        core.handle_block(block)
    recorder.close()
    recorder.close()

    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    assert rows[0] == ["time_s", "FDS_mV", "EDC_mV", "FDI_mV", "APB_mV"]
    data = np.array(rows[1:], dtype=float)
    assert data.shape == (2 * BLOCK, 5) and recorder.rows == 2 * BLOCK
    np.testing.assert_allclose(data[:, 0], np.arange(2 * BLOCK) / RATE)
    np.testing.assert_allclose(data[:, 1:].T, np.hstack([b.samples for b in blocks]))
    # The header is skipped by the dataset loader, so recordings can be loaded like the dataset
    assert len(load_columns(path, [1])[0]) == 2 * BLOCK


# -- threaded acquisition ---------------------------------------------------------


def test_threaded_acquisition_streams_and_stops(config: AppConfig) -> None:
    """Real-time synthetic EMG flows at the source rate until stop()."""
    acquisition = Acquisition(config, SyntheticSource(config.synthetic, config.channels))
    acquisition.start()
    try:
        assert wait_for(lambda: acquisition.snapshot().counters.get("blocks", 0) > 100)
        snapshot = acquisition.snapshot()
        assert snapshot.connected and snapshot.running
        assert snapshot.measured_rate == pytest.approx(RATE, rel=0.2)
    finally:
        acquisition.stop()
    assert not acquisition.running and acquisition.error is None
    assert not acquisition.snapshot().connected


def test_disconnect_reconnects(config: AppConfig) -> None:
    """After a lost connection the source is restarted and data flows again."""
    source = SyntheticSource(config.synthetic, config.channels)
    acquisition = Acquisition(fast_reconnect(config), source)
    acquisition.start()
    try:
        assert wait_for(lambda: acquisition.snapshot().counters.get("blocks", 0) > 10)
        source.inject_disconnect()
        assert wait_for(lambda: acquisition.snapshot().counters.get("reconnections", 0) == 1)
        blocks = acquisition.snapshot().counters["blocks"]
        assert wait_for(lambda: acquisition.snapshot().counters["blocks"] > blocks + 10)
        assert acquisition.snapshot().connected
    finally:
        acquisition.stop()


def test_recording_end_stops_acquisition(config: AppConfig, tmp_path: Path) -> None:
    """A non-looping recording ends the acquisition thread without an error."""
    path = tmp_path / "short.csv"
    path.write_text("\n".join("0.001,0.002,0.003,0.004" for _ in range(400)), encoding="utf-8")
    source = RecordedSource(
        RecordedConfig(file=str(path), loop=False, realtime=False), config.channels
    )
    acquisition = Acquisition(config, source)
    acquisition.start()
    try:
        assert wait_for(lambda: not acquisition.running)
        snapshot = acquisition.snapshot()
        assert any("end of recording" in message for message in snapshot.messages)
        assert snapshot.raw.shape[1] == 400
    finally:
        acquisition.stop()
    assert acquisition.error is None


def test_trigno_through_mock_tcu(config: AppConfig) -> None:
    """The Trigno backend streams, survives a dropped data connection and quits cleanly."""
    generator = offline_synthetic(config)
    settings = MockTCUSettings(
        sensors=[
            MockSensor(slot=c.trigno_slot, position=c.trigno_slot - 1) for c in config.channels
        ]
    )
    with MockTCU(settings, lambda first, n: generator.generate(first, n) / 1000.0) as tcu:
        trigno = replace(
            config.trigno,
            command_port=tcu.command_port,
            emg_port=tcu.emg_port,
            command_timeout_s=1.0,
        )
        acquisition = Acquisition(fast_reconnect(config), TrignoSource(trigno, config.channels))
        acquisition.start()
        try:
            assert acquisition.core is not None and acquisition.core.sample_rate == RATE
            assert wait_for(lambda: acquisition.snapshot().counters.get("blocks", 0) > 50)
            tcu.drop_data_connection()
            assert wait_for(lambda: acquisition.snapshot().counters.get("reconnections", 0) == 1)
        finally:
            acquisition.stop()
        assert tcu.received_commands[-1] == "QUIT"


# -- viewer and CLI ---------------------------------------------------------------


def test_viewer_and_console_render(config: AppConfig, capsys: pytest.CaptureFixture[str]) -> None:
    """The live viewer draws filtered and raw frames; the console prints a status line."""
    acquisition = Acquisition(config, SyntheticSource(config.synthetic, config.channels))
    acquisition.start()
    try:
        assert wait_for(lambda: acquisition.snapshot().time.size > 200)
        viewer = LiveViewer(acquisition)
        viewer._draw(0)  # pylint: disable=protected-access
        assert viewer.axes[0].get_title().startswith("Filtered EMG")
        assert viewer.envelope_lines[0].get_visible()
        assert viewer.fast_envelope_lines[0].get_visible()
        assert np.asarray(viewer.tremor_lines[0].get_xdata()).size > 200
        viewer._toggle_raw()  # pylint: disable=protected-access
        viewer._draw(1)  # pylint: disable=protected-access
        assert viewer.axes[0].get_title() == "Raw EMG"
        assert not viewer.envelope_lines[0].get_visible()
        assert not viewer.fast_envelope_lines[0].get_visible()
        assert viewer.tremor_lines[0].get_visible()
        plt.close(viewer.figure)
        assert any(line.startswith("Sample rate") for line in status_lines(acquisition.snapshot()))
        header = status_lines(acquisition.snapshot())
        assert any("tremor uV" in line and "depth" in line and "Hz" in line for line in header)
        ConsoleStatus(acquisition)()
        output = capsys.readouterr().out
        assert "connected" in output and "depth" in output and "ratio" in output
    finally:
        acquisition.stop()


def test_cli_headless_saves_csv(tmp_path: Path) -> None:
    """A short headless run exits cleanly and records the raw EMG."""
    path = tmp_path / "session.csv"
    code = run_emg.main(
        ["--source", "synthetic", "--headless", "--duration", "1.0", "--save", str(path)]
    )
    assert code == 0
    rows = path.read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("time_s,FDS_mV") and len(rows) > 1000


def test_cli_reports_missing_tcu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a TCU the trigno source fails to start with exit code 3."""
    monkeypatch.setattr(
        applied_config,
        "TRIGNO",
        replace(applied_config.TRIGNO, command_port=1, connect_timeout_s=0.3),
    )
    assert run_emg.main(["--source", "trigno", "--headless"]) == run_emg.EXIT_SOURCE


def test_cli_rejects_inconsistent_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inconsistent settings (two channels on one TCU slot) give exit code 2."""
    duplicate = replace(applied_config.FOUR_SENSORS[1], trigno_slot=1)
    monkeypatch.setattr(applied_config, "CHANNELS", (applied_config.FOUR_SENSORS[0], duplicate))
    assert run_emg.main(["--headless"]) == run_emg.EXIT_CONFIG


def test_settings_are_valid() -> None:
    """Both channel sets in applied_config.py pass validation."""
    assert applied_config.build_config(applied_config.ONE_SENSOR).channel_names == ("FDS",)
    four = applied_config.build_config(applied_config.FOUR_SENSORS)
    assert four.channel_names == ("FDS", "EDC", "FDI", "APB")


def test_cli_rejects_unwritable_recording(tmp_path: Path) -> None:
    """A recording path that cannot be created gives exit code 2."""
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("", encoding="utf-8")
    code = run_emg.main(["--headless", "--save", str(blocker / "session.csv")])
    assert code == run_emg.EXIT_CONFIG
