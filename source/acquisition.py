"""
Live EMG acquisition:

  EMG source -> acquisition thread -> StreamingProcessor -> ring buffers -> viewer
                                   |                    \\-> ControlStage -> control ring buffer
                                   \\-> optional CSV recording of the raw samples

- One thread reads blocks from the source, filters them (causal, see
  streaming.py) and appends the raw EMG, filtered EMG, slow and fast envelopes,
  tremor amplitude, ratio and depth to ring buffers holding the last
  runtime.monitor_seconds.
- Every block then goes through the control stage (control_stage.py):
  calibration, normalized activation and the controller. Its grasp activation,
  tremor measure and stiffness command (the control signal) are buffered too.
- snapshot() also reports the dominant frequency of each fast envelope over
  the last processing.frequency_window_s.
- The viewer (viewer.py) copies those buffers with snapshot().
- A lost connection is retried runtime.max_reconnect_attempts times; the
  filters restart after a reconnection.

StreamCore holds the per-block logic without threads, so tests can drive it
deterministically.
"""

import csv
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .calibration_store import CalibrationStore
from .config import AppConfig
from .control_stage import ControlStage
from .controller import ControllerState
from .emg_source import (
    EMGBlock,
    EMGSource,
    SourceConnectionError,
    SourceError,
    SourceTimeoutError,
)
from .streaming import StreamingProcessor, dominant_frequency

FloatArray = NDArray[np.float64]
RATE_WINDOW_BLOCKS = 150  # blocks used for the measured sample rate (~2 s of 13.5 ms frames)
FLUSH_PERIOD_S = 1.0  # the recording is flushed to disk at least this often
MAX_EVENTS = 10000  # messages kept for the session log (the viewer shows the last 8)
CONTROL_BUFFER_DIVISOR = 4  # control history capacity: monitor_seconds * sample_rate / this


class RingBuffer:
    """Fixed-size history of (n_rows, n) data; oldest values are overwritten."""

    def __init__(self, n_rows: int, capacity: int) -> None:
        self.data = np.zeros((n_rows, capacity))
        self.capacity = capacity
        self._position = 0
        self._filled = 0

    def write(self, values: FloatArray) -> None:
        """Append columns."""
        values = values[:, -self.capacity :]
        n = values.shape[1]
        end = self._position + n
        if end <= self.capacity:
            self.data[:, self._position : end] = values
        else:
            first = self.capacity - self._position
            self.data[:, self._position :] = values[:, :first]
            self.data[:, : n - first] = values[:, first:]
        self._position = end % self.capacity
        self._filled = min(self.capacity, self._filled + n)

    def read(self) -> FloatArray:
        """All stored columns, oldest first."""
        if self._filled < self.capacity:
            return self.data[:, : self._filled].copy()
        return np.concatenate(
            [self.data[:, self._position :], self.data[:, : self._position]], axis=1
        )


class EMGRecorder:
    """Writes raw EMG to CSV: a header row, then time_s and one column per channel in mV."""

    def __init__(self, path: Path, channel_names: Sequence[str], sample_rate: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.sample_rate = sample_rate
        self.rows = 0
        self._last_flush = time.monotonic()
        self._file = open(  # pylint: disable=consider-using-with  # closed in close()
            path, "w", newline="", encoding="utf-8"
        )
        self._writer = csv.writer(self._file)
        self._writer.writerow(["time_s", *(f"{name}_mV" for name in channel_names)])

    def write(self, block: EMGBlock) -> None:
        """Append one block; time_s is the sample index divided by the sample rate."""
        if self._file.closed:
            return
        times = (block.first_sample_index + np.arange(block.n_samples)) / self.sample_rate
        self._writer.writerows(np.column_stack((times, block.samples.T)).tolist())
        self.rows += block.n_samples
        if time.monotonic() - self._last_flush >= FLUSH_PERIOD_S:
            # Keep the samples on disk if the program or the PC dies mid-session
            self._file.flush()
            self._last_flush = time.monotonic()

    def close(self) -> None:
        """Flush and close the file. Safe to call twice."""
        self._file.close()


@dataclass(frozen=True)
class ViewerSnapshot:  # pylint: disable=too-many-instance-attributes
    """
    Everything the viewer displays, copied under a lock. Signals are
    (n_channels, n), in mV except the dimensionless tremor_ratio and tremor_depth
    (see streaming.py):

    envelope       slow envelope: moving RMS over processing.slow_window_s (voluntary effort)
    fast_envelope  rectified EMG low-passed at processing.fast_cutoff_hz (follows tremor bursts)
    tremor         RMS of the fast envelope's processing.tremor_band_hz component
    tremor_ratio   tremor-band / rhythm-band fluctuation: ~1 rhythmic tremor, 0.2-0.6 otherwise
    tremor_depth   tremor / slow envelope: ~0.05 without tremor at any effort, higher with tremor
    dominant_hz    per channel: peak frequency of the fast envelope's fluctuation within
                   processing.rhythm_band_hz over the last frequency_window_s (NaN if unknown)

    The control history has one entry per block, (n,):
    control_time   stream time of the controller tick, s
    grasp          grasp activation (0 = rest, 1 = reference grip)
    tremor_control the tremor measure the controller thresholds apply to (tremor_measure)
    stiffness      the control signal: commanded stiffness 0 (released) to 1
    """

    channel_names: tuple[str, ...]
    sample_rate: float
    measured_rate: float
    time: FloatArray
    raw: FloatArray
    filtered: FloatArray
    envelope: FloatArray
    fast_envelope: FloatArray
    tremor: FloatArray
    tremor_ratio: FloatArray
    tremor_depth: FloatArray
    dominant_hz: FloatArray
    connected: bool
    running: bool
    counters: dict[str, int]
    source_info: dict[str, str]
    messages: tuple[str, ...]
    recording: str | None
    control_time: FloatArray = field(default_factory=lambda: np.zeros(0))
    grasp: FloatArray = field(default_factory=lambda: np.zeros(0))
    tremor_control: FloatArray = field(default_factory=lambda: np.zeros(0))
    stiffness: FloatArray = field(default_factory=lambda: np.zeros(0))
    state: str = ControllerState.DISARMED.value
    armed: bool = False
    last_transition: str = ""
    calibration_phase: str = ""
    calibration_seconds_left: float = 0.0
    calibration_error: str | None = None
    tremor_measure: str = "depth"
    thresholds: dict[str, float] = field(default_factory=dict)


class StreamCore:  # pylint: disable=too-many-instance-attributes
    """Filtering, buffering, gap detection, recording and control, one block at a time."""

    def __init__(
        self,
        config: AppConfig,
        sample_rate: float,
        recorder: EMGRecorder | None = None,
        calibration_store: CalibrationStore | None = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("The sample rate must be known before acquisition starts")
        self.config = config
        self.sample_rate = sample_rate
        self.recorder = recorder
        n_channels = len(config.channel_names)
        self.processor = StreamingProcessor(config.processing, n_channels, sample_rate)
        self.control = ControlStage(config, sample_rate, self.message, calibration_store)
        capacity = max(2, int(config.runtime.monitor_seconds * sample_rate))
        self._time = RingBuffer(1, capacity)
        self._raw = RingBuffer(n_channels, capacity)
        self._filtered = RingBuffer(n_channels, capacity)
        self._envelope = RingBuffer(n_channels, capacity)
        self._fast_envelope = RingBuffer(n_channels, capacity)
        self._tremor = RingBuffer(n_channels, capacity)
        self._tremor_ratio = RingBuffer(n_channels, capacity)
        self._tremor_depth = RingBuffer(n_channels, capacity)
        # time, grasp, tremor measure, stiffness; one entry per block
        self._control = RingBuffer(4, max(2, capacity // CONTROL_BUFFER_DIVISOR))
        self._arrivals: deque[tuple[float, int]] = deque(maxlen=RATE_WINDOW_BLOCKS)
        self._samples_received = 0
        self._expected_index: int | None = None
        self._counters: dict[str, int] = {}
        self._messages: deque[str] = deque(maxlen=8)
        self._events: deque[tuple[float, str]] = deque(maxlen=MAX_EVENTS)
        self._lock = threading.Lock()

    def message(self, text: str) -> None:
        """Add a line to the viewer's message log and to the event log."""
        with self._lock:
            self._messages.append(text)
            self._events.append((self._samples_received / self.sample_rate, text))

    def event_log(self) -> list[tuple[float, str]]:
        """All messages with the recording time (s of samples received) they happened at."""
        with self._lock:
            return list(self._events)

    def count(self, name: str, increment: int = 1) -> None:
        """Increase a counter."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + increment

    def new_session(self) -> None:
        """A reconnection: restart the filters and the gap detection."""
        self.processor.reset()
        self._expected_index = None
        with self._lock:
            self._arrivals.clear()
        self.message("New acquisition session: filters reset")

    def handle_block(self, block: EMGBlock) -> None:
        """Record, check, filter, buffer and control on one block from the source."""
        if self.recorder is not None:
            self.recorder.write(block)
        if not np.all(np.isfinite(block.samples)):
            # NaN/inf would poison the filter state for good
            self.count("invalid blocks")
            samples = np.nan_to_num(block.samples, nan=0.0, posinf=0.0, neginf=0.0)
            block = EMGBlock(samples, block.first_sample_index, block.receive_time)

        if self._expected_index is not None and block.first_sample_index != self._expected_index:
            missing = block.first_sample_index - self._expected_index
            if missing > 0:
                self.count("missing samples", missing)
                self.count("gaps")
            else:
                self.count("index resets")
        self._expected_index = block.first_sample_index + block.n_samples

        processed = self.processor.process(block)
        times = (block.first_sample_index + np.arange(block.n_samples)) / self.sample_rate
        with self._lock:
            self._time.write(times[None, :])
            self._raw.write(processed.raw)
            self._filtered.write(processed.filtered)
            self._envelope.write(processed.slow_envelope)
            self._fast_envelope.write(processed.fast_envelope)
            self._tremor.write(processed.tremor_amplitude)
            self._tremor_ratio.write(processed.tremor_ratio)
            self._tremor_depth.write(processed.tremor_depth)
            self._samples_received += block.n_samples
            self._arrivals.append((block.receive_time, self._samples_received))
            self._counters["blocks"] = self._counters.get("blocks", 0) + 1

        output = self.control.step(processed)
        activation = self.control.activation
        grasp = activation.grasp_activation if activation is not None else 0.0
        tremor = (
            activation.tremor_measure(self.control.tremor_measure)
            if activation is not None
            else 0.0
        )
        with self._lock:
            self._control.write(
                np.array([[float(times[-1])], [grasp], [tremor], [output.stiffness]])
            )

    def _measured_rate(self) -> float:
        """Samples per second of wall-clock time over the recent blocks (call under the lock)."""
        if len(self._arrivals) < 2:
            return 0.0
        (start_time, start_count), (end_time, end_count) = self._arrivals[0], self._arrivals[-1]
        if end_time <= start_time:
            return 0.0
        return (end_count - start_count) / (end_time - start_time)

    def snapshot(self, source: EMGSource, connected: bool, running: bool) -> ViewerSnapshot:
        """Copy everything the viewer shows."""
        source_info = source.info()
        control = self.control
        output = control.last_output
        with self._lock:
            fast_envelope = self._fast_envelope.read()
            history = self._control.read()
            snapshot = ViewerSnapshot(
                channel_names=self.config.channel_names,
                sample_rate=self.sample_rate,
                measured_rate=self._measured_rate(),
                time=self._time.read()[0],
                raw=self._raw.read(),
                filtered=self._filtered.read(),
                envelope=self._envelope.read(),
                fast_envelope=fast_envelope,
                tremor=self._tremor.read(),
                tremor_ratio=self._tremor_ratio.read(),
                tremor_depth=self._tremor_depth.read(),
                dominant_hz=np.full(len(self.config.channel_names), np.nan),  # set below
                connected=connected,
                running=running,
                counters=dict(self._counters),
                source_info=source_info,
                messages=tuple(self._messages),
                recording=str(self.recorder.path) if self.recorder is not None else None,
                control_time=history[0],
                grasp=history[1],
                tremor_control=history[2],
                stiffness=history[3],
                state=output.state.value if output else control.controller.state.value,
                armed=output.armed if output else False,
                last_transition=control.controller.last_transition,
                calibration_phase=control.phase,
                calibration_seconds_left=control.calibrator.seconds_left,
                calibration_error=control.calibrator.error,
                tremor_measure=control.tremor_measure,
                thresholds=control.thresholds,
            )
        # The spectrum runs outside the lock so acquisition is never held up by the viewer
        processing = self.config.processing
        window = int(processing.frequency_window_s * self.sample_rate)
        dominant_hz = dominant_frequency(
            fast_envelope[:, -window:], self.sample_rate, processing.rhythm_band_hz
        )
        return replace(snapshot, dominant_hz=dominant_hz)


class Acquisition:
    """Runs a StreamCore on a source in a background thread."""

    def __init__(
        self,
        config: AppConfig,
        source: EMGSource,
        save_path: Path | None = None,
        calibration_store: CalibrationStore | None = None,
    ) -> None:
        self.config = config
        self.source = source
        self.save_path = save_path
        self.calibration_store = calibration_store
        self.core: StreamCore | None = None
        self.error: str | None = None
        self._stop = threading.Event()
        self._connected = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """
        Start the source and the acquisition thread. Raises SourceError if the
        source cannot start, OSError if the recording file cannot be created.
        """
        self.source.start()
        self._connected = True
        recorder = None
        if self.save_path is not None:
            try:
                recorder = EMGRecorder(
                    self.save_path, self.config.channel_names, self.source.sample_rate
                )
            except OSError:
                self.source.stop()
                self._connected = False
                raise
        self.core = StreamCore(
            self.config, self.source.sample_rate, recorder, self.calibration_store
        )
        self.core.message(f"Source started: {self.source.kind} at {self.source.sample_rate:g} Hz")
        self._thread = threading.Thread(target=self._loop, name="acquisition", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread and the source, close the recording. Safe to call twice."""
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5.0)
        try:
            self.source.stop()
        except SourceError as error:
            self.error = self.error or f"Source stop failed: {error}"
        self._connected = False
        if self.core is not None and self.core.recorder is not None:
            self.core.recorder.close()

    @property
    def running(self) -> bool:
        """True while the acquisition thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def run_for(
        self, seconds: float, on_tick: Callable[[], None] | None = None, period_s: float = 1.0
    ) -> None:
        """Block for `seconds` (or until acquisition ends), calling on_tick periodically."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and self.running and not self._stop.is_set():
            time.sleep(min(period_s, max(0.0, deadline - time.monotonic())))
            if on_tick is not None:
                on_tick()

    def snapshot(self) -> ViewerSnapshot:
        """Viewer data."""
        assert self.core is not None
        return self.core.snapshot(self.source, self._connected, self.running)

    def recalibrate(self) -> None:
        """Restart the calibration from the next block (the controller disarms meanwhile)."""
        if self.core is not None:
            self.core.control.request_recalibration()

    def _loop(self) -> None:
        assert self.core is not None
        core = self.core
        try:
            while not self._stop.is_set():
                try:
                    block = self.source.read_block()
                except SourceTimeoutError as error:
                    core.count("timeouts")
                    core.message(f"Timeout: {error}")
                    continue
                except SourceConnectionError as error:
                    self._connected = False
                    core.message(f"Connection lost: {error}")
                    if not self._reconnect():
                        return
                    core.new_session()
                    continue
                if block is None:
                    core.message("Source finished (end of recording)")
                    return
                core.handle_block(block)
        except Exception as error:  # pylint: disable=broad-exception-caught
            # Report any unexpected error in the viewer instead of dying silently
            self.error = f"Acquisition crashed: {error!r}"
            core.message(self.error)

    def _reconnect(self) -> bool:
        assert self.core is not None
        runtime = self.config.runtime
        for attempt in range(1, runtime.max_reconnect_attempts + 1):
            if self._stop.wait(runtime.reconnect_delay_s):
                return False
            try:
                self.source.stop()
                rate = self.source.sample_rate
                self.source.start()
            except SourceError as error:
                self.core.message(f"Reconnect attempt {attempt} failed: {error}")
                continue
            if self.source.sample_rate != rate:
                self.core.message("Sample rate changed on reconnection; giving up")
                return False
            self._connected = True
            self.core.count("reconnections")
            self.core.message(f"Reconnected (attempt {attempt})")
            return True
        self.core.message("Reconnection failed; acquisition stopped")
        return False
