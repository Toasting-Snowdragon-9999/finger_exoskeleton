"""
The live pipeline: what happens to every block of EMG.

  Pipeline.step(block):
    1. filter it (SignalProcessor)
    2. keep the last few seconds for the viewer
    3. calibrate, or compute the activation with the calibration
    4. after calibration: arm the controller once the hand has relaxed
       (grasp <= GRASP_OFF for GRASP_OFF_TIME_S), so a grip still held at
       the end of the calibration does not count as a grasp
    5. update the stiffness controller

  Acquisition: a thread that reads blocks from a source, records them (with
  --out) and gives them to Pipeline.step until it is stopped.

The controller runs on stream time (sample number / sample rate), so a replay
gives the same result as the live run it was recorded from.
"""

import threading
from collections import deque
from pathlib import Path

import numpy as np

import settings
from settings import Channel
from source.calibration import Activation, Calibration, Calibrator, compute_activation
from source.calibration import load_calibration
from source.signal_processing import EMGBlock, SignalProcessor
from source.stiffness_controller import HoldTimer, State, StiffnessController


class Pipeline:  # pylint: disable=too-many-instance-attributes
    """Filtering, calibration and control of one stream of EMG blocks."""

    def __init__(self, channels: list[Channel], sample_rate: float,
                 calibration_file: Path | None = None, use_saved: bool = True) -> None:
        self.channels = channels
        self.names = [channel.name for channel in channels]
        self.sample_rate = sample_rate
        self.calibration_file = calibration_file
        self.processor = SignalProcessor(len(channels), sample_rate)
        self.calibrator = Calibrator(self.names, sample_rate)
        self.controller = StiffnessController()
        self.relax_timer = HoldTimer(settings.GRASP_OFF_TIME_S)

        flexors = np.array([channel.role == "flexor" for channel in channels])
        self.flexors = flexors if flexors.any() else np.ones(len(channels), dtype=bool)

        self.calibration: Calibration | None = None  # in use
        self.activation: Activation | None = None  # of the newest block
        self.waiting_to_arm = False  # calibrated, waiting for the hand to relax
        self.loaded_calibration = False
        self.reported_error: str | None = None
        self.recalibrate_requested = False
        self.samples_received = 0

        self.lock = threading.Lock()  # the viewer reads the history from another thread
        self.messages: list[tuple[float, str]] = []  # (seconds of EMG received, text)
        blocks = int(settings.MONITOR_SECONDS * sample_rate / settings.BLOCK_SIZE) * 2
        self.history: deque = deque(maxlen=blocks)  # recent ProcessedBlocks
        self.control_history: deque = deque(maxlen=blocks)  # (time, grasp, depth, stiffness)

        if calibration_file is not None and use_saved:
            saved, note = load_calibration(calibration_file, self.names)
            if note:
                self.message(note)
            if saved is not None:
                self.loaded_calibration = True
                self.calibrator.result = saved

    # --- what the viewer and the motor read ---------------------------------------------------

    @property
    def state(self) -> State:
        """The controller state."""
        return self.controller.state

    @property
    def phase(self) -> str:
        """What the user should be doing: the calibration phase, or "relax to arm"."""
        return "relax to arm" if self.waiting_to_arm else self.calibrator.phase

    def message(self, text: str) -> None:
        """Print a message and keep it for the viewer and the session file."""
        print(text, flush=True)
        with self.lock:
            self.messages.append((self.samples_received / self.sample_rate, text))

    def recent(self) -> dict:
        """Copies of the last MONITOR_SECONDS of signals, for the viewer."""
        with self.lock:
            blocks = list(self.history)
            control = np.array(list(self.control_history)).reshape(-1, 4).T
            messages = [text for _, text in self.messages[-8:]]
        n_keep = int(settings.MONITOR_SECONDS * self.sample_rate)
        n_channels = len(self.channels)

        def join(name: str) -> np.ndarray:
            if not blocks:
                return np.zeros((n_channels, 0))
            return np.concatenate([getattr(block, name) for block in blocks], axis=1)[:, -n_keep:]

        if blocks:
            first = blocks[0].first_index
            n_samples = sum(block.raw.shape[1] for block in blocks)
            times = (first + np.arange(n_samples)) / self.sample_rate
        else:
            times = np.zeros(0)
        return {
            "time": times[-n_keep:], "filtered": join("filtered"), "slow": join("slow"),
            "fast": join("fast"), "depth": join("depth"), "ratio": join("ratio"),
            "control_time": control[0], "grasp": control[1], "control_depth": control[2],
            "stiffness": control[3], "messages": messages,
        }

    def recalibrate(self) -> None:
        """Start a new calibration from the next block (called from the viewer)."""
        self.recalibrate_requested = True

    # --- the work for every block -------------------------------------------------------------

    def step(self, block: EMGBlock) -> None:
        """Process one block of raw EMG; see the top of this file."""
        # NaN or infinity would break the filters for good
        block.samples = np.nan_to_num(block.samples, nan=0.0, posinf=0.0, neginf=0.0)
        processed = self.processor.process(block)
        n = block.samples.shape[1]
        self.samples_received += n
        with self.lock:
            self.history.append(processed)

        if self.recalibrate_requested:
            self.recalibrate_requested = False
            self.restart_calibration()
        if self.calibration is None:
            self.calibrator.update(processed)
            self.check_calibration()

        now = (block.first_index + n) / self.sample_rate
        if self.calibration is not None:
            self.activation = compute_activation(processed, self.calibration, self.flexors)
            if self.waiting_to_arm:
                self.arm_when_relaxed(self.activation, now)
        self.controller.update(self.activation, now)

        grasp = self.activation.grasp if self.activation is not None else 0.0
        depth = self.activation.tremor_depth if self.activation is not None else 0.0
        with self.lock:
            self.control_history.append((now, grasp, depth, self.controller.stiffness))

    def check_calibration(self) -> None:
        """Start using a finished calibration, or report why it failed."""
        result = self.calibrator.result
        if result is not None:
            self.calibration = result
            self.waiting_to_arm = True
            self.relax_timer.reset()
            start = "Using the saved calibration" if self.loaded_calibration else "Calibration complete"
            self.message(f"{start}; relax the hand to arm the controller")
            levels = " ".join(f"{name} {rest * 1000:.0f}/{grip * 1000:.0f}" for name, rest, grip
                              in zip(self.names, result.rest_slow, result.reference_slow))
            self.message(f"  rest/reference uV: {levels}")
            if not self.loaded_calibration and self.calibration_file is not None:
                result.save(self.calibration_file)
                self.message(f"Calibration saved to {self.calibration_file}; later runs use it")
        elif self.calibrator.error is not None and self.calibrator.error != self.reported_error:
            self.reported_error = self.calibrator.error
            self.message(f"Calibration failed: {self.calibrator.error} Press 'c' to repeat it.")

    def arm_when_relaxed(self, activation: Activation, now: float) -> None:
        """Arm the controller once the grasp has stayed at rest for GRASP_OFF_TIME_S."""
        if self.relax_timer.update(activation.grasp <= settings.GRASP_OFF, now):
            self.waiting_to_arm = False
            self.controller.arm()
            self.message("Hand relaxed; controller armed")

    def restart_calibration(self) -> None:
        """Forget the calibration and run the calibration routine again."""
        self.controller.disarm("recalibration")
        self.calibration = None
        self.activation = None
        self.waiting_to_arm = False
        self.loaded_calibration = False
        self.reported_error = None
        self.calibrator.restart()
        rest, grip = settings.REST_WINDOW_S, settings.GRIP_WINDOW_S
        self.message(f"Calibration restarted: rest {rest[0]:g}-{rest[1]:g} s, "
                     f"reference grip {grip[0]:g}-{grip[1]:g} s")


class Acquisition:
    """Reads blocks from a source in a background thread and feeds them to the pipeline."""

    def __init__(self, source, pipeline: Pipeline, session=None) -> None:
        self.source = source
        self.pipeline = pipeline
        self.session = session  # a SessionWriter (source/recording.py) or None
        self.error: str | None = None
        self.stop_requested = False
        self.thread = threading.Thread(target=self.loop, name="acquisition", daemon=True)

    @property
    def running(self) -> bool:
        """True while the thread is reading blocks."""
        return self.thread.is_alive()

    def start(self) -> None:
        """Start reading blocks."""
        self.thread.start()

    def stop(self) -> None:
        """Stop the thread and the source."""
        self.stop_requested = True
        if self.thread.is_alive():
            self.thread.join(timeout=5.0)
        self.source.stop()

    def loop(self) -> None:
        """The thread: read, record and process blocks until stopped."""
        try:
            while not self.stop_requested:
                try:
                    block = self.source.read_block()
                except TimeoutError:
                    self.pipeline.message("No EMG data from the source; still waiting")
                    continue
                if block is None:
                    self.pipeline.message("Source finished (end of recording)")
                    return
                if self.session is not None:
                    self.session.write(block)
                self.pipeline.step(block)
        except Exception as error:  # pylint: disable=broad-exception-caught
            # Keep the error for the viewer and the session file instead of losing it in the thread
            self.error = f"Acquisition stopped: {error!r}"
            self.pipeline.message(self.error)
