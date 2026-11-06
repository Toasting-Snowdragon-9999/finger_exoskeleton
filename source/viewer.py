"""
Live EMG viewer for a running Acquisition.

LiveViewer: a matplotlib window, refreshed about ten times per second. Per channel:
  - EMG panel: filtered (or raw) EMG with the slow envelope (moving RMS, voluntary
    effort) and the fast envelope (follows tremor bursts).
  - Tremor panel: tremor depth (4-6 Hz fluctuation of the fast envelope relative
    to the slow envelope, so independent of effort) and the tremor ratio
    (rhythmicity: ~1 for rhythmic tremor, ~0.5 for broadband).
  Below the channels, the controller panel: grasp activation, the tremor measure
  the controller uses, and the stiffness command (the control signal, 0 = released,
  1 = maximum), with the on-thresholds as dashed lines. Its title shows the
  controller state and the last transition.
  The status panel shows connection, sample rate, current values, the controller
  state and stiffness, the calibration phase and counters.
  Keys: r = raw / filtered, c = recalibrate, space = pause, q = quit.

ConsoleStatus: one status line per call, for headless runs.
"""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event, KeyEvent
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from .acquisition import Acquisition, ViewerSnapshot

# Fixed y range of the EMG panels in mV, so levels can be compared between runs (None: autoscale).
# 0 to 0.15 shows the envelopes; the oscillating EMG trace is cut off below 0 and above 0.15.
EMG_Y_LIMITS: tuple[float, float] | None = (0.0, 0.15)

SIGNAL_COLOR = "#1f77b4"
ENVELOPE_COLOR = "#d62728"
FAST_ENVELOPE_COLOR = "#ff7f0e"
TREMOR_COLOR = "#7b1fa2"
RATIO_COLOR = "#757575"
STIFFNESS_COLOR = "#000000"
STATE_COLORS = {
    "DISARMED": "#757575",
    "IDLE": "#2e7d32",
    "GRASP_ARMED": "#f9a825",
    "STIFFENED": "#c62828",
    "FAULT": "#c62828",
}


def _latest(values: NDArray[np.float64], scale: float = 1.0) -> list[float]:
    """Latest value per channel of an (n_channels, n) history, times scale."""
    if values.size == 0:
        return [0.0] * values.shape[0]
    return [float(value) * scale for value in values[:, -1]]


def _last(values: NDArray[np.float64]) -> float:
    """Last value of a (n,) history, 0 when empty."""
    return float(values[-1]) if values.size else 0.0


def _hz(value: float) -> str:
    """A frequency for display; '-' when unknown."""
    return "-" if np.isnan(value) else f"{value:.1f}"


def _channel_values(snapshot: ViewerSnapshot) -> list[tuple[str, float, float, float, float, str]]:
    """Per channel: name, RMS uV, tremor uV, depth, ratio, dominant frequency."""
    return list(
        zip(
            snapshot.channel_names,
            _latest(snapshot.envelope, 1000.0),
            _latest(snapshot.tremor, 1000.0),
            _latest(snapshot.tremor_depth),
            _latest(snapshot.tremor_ratio),
            [_hz(float(value)) for value in snapshot.dominant_hz],
        )
    )


def calibration_text(snapshot: ViewerSnapshot) -> str:
    """One line on the calibration: what to do now, or the error."""
    if snapshot.calibration_error:
        return f"FAILED: {snapshot.calibration_error}"
    if snapshot.calibration_phase in ("done", "relax to arm"):
        return snapshot.calibration_phase
    return f"{snapshot.calibration_phase} ({snapshot.calibration_seconds_left:.0f} s left)"


def controller_lines(snapshot: ViewerSnapshot) -> list[str]:
    """Text status of the controller: state, stiffness, inputs and thresholds."""
    thresholds = snapshot.thresholds
    return [
        f"Controller:  {snapshot.state}   stiffness {_last(snapshot.stiffness):.2f}",
        f"  grasp {_last(snapshot.grasp):.2f} (on {thresholds.get('grasp on', 0):.2f} / "
        f"off {thresholds.get('grasp off', 0):.2f})",
        f"  tremor {snapshot.tremor_measure} {_last(snapshot.tremor_control):.2f} "
        f"(on {thresholds.get('tremor on', 0):.2f} / off {thresholds.get('tremor off', 0):.2f}, "
        f"ratio >= {thresholds.get('ratio gate', 0):.2f})",
        f"  last: {snapshot.last_transition}",
        f"Calibration: {calibration_text(snapshot)}",
    ]


def status_lines(snapshot: ViewerSnapshot) -> list[str]:
    """Text status for the viewer's side panel."""
    if not snapshot.running:
        connection = "STOPPED"
    else:
        connection = "connected" if snapshot.connected else "DISCONNECTED"
    lines = [
        f"Connection:  {connection}",
        f"Sample rate: {snapshot.sample_rate:.3f} Hz (measured {snapshot.measured_rate:.1f} Hz)",
        "",
        f"  {'now':6s} {'RMS uV':>7s} {'tremor uV':>9s} {'depth':>5s} {'ratio':>5s} {'Hz':>4s}",
    ]
    lines += [
        f"  {name:6s} {rms:7.1f} {tremor:9.2f} {depth:5.2f} {ratio:5.2f} {hz:>4s}"
        for name, rms, tremor, depth, ratio, hz in _channel_values(snapshot)
    ]
    lines += [""] + controller_lines(snapshot)
    counters = ", ".join(f"{name} {value}" for name, value in sorted(snapshot.counters.items()))
    lines += ["", f"Counters: {counters or '-'}"]
    if snapshot.recording:
        lines.append(f"Recording:   {snapshot.recording}")
    lines.append("")
    lines += [f"{key}: {value}" for key, value in snapshot.source_info.items()]
    lines += [""] + list(snapshot.messages)
    return lines


class LiveViewer:  # pylint: disable=too-many-instance-attributes
    """matplotlib live view; run() blocks until the window is closed or q is pressed."""

    def __init__(self, acquisition: Acquisition) -> None:
        self.acquisition = acquisition
        self.show_raw = False
        self.paused = False
        names = acquisition.config.channel_names
        self.figure = plt.figure(figsize=(14, max(6.5, 3.0 * len(names) + 2.5)))
        self.figure.canvas.manager.set_window_title("Live EMG")  # type: ignore[union-attr]
        grid = self.figure.add_gridspec(
            2 * len(names) + 1,
            2,
            width_ratios=(3.0, 1.0),
            height_ratios=(2.0, 1.0) * len(names) + (1.6,),
            hspace=0.4,
        )

        self.axes: list[Axes] = []  # EMG panels
        self.tremor_axes: list[Axes] = []
        self.ratio_axes: list[Axes] = []
        self.signal_lines: list[Line2D] = []
        self.envelope_lines: list[Line2D] = []
        self.fast_envelope_lines: list[Line2D] = []
        self.tremor_lines: list[Line2D] = []
        self.ratio_lines: list[Line2D] = []
        for row, name in enumerate(names):
            self._add_channel(grid, row, name)
        self.axes[0].set_xlim(-acquisition.config.runtime.monitor_seconds, 0.0)
        self.axes[0].legend(loc="upper left", fontsize=7, ncol=3)
        self.tremor_axes[0].legend(
            handles=[self.tremor_lines[0], self.ratio_lines[0]],
            loc="upper left",
            fontsize=7,
            ncol=2,
        )
        self.control_ax, self.control_lines = self._add_controller(grid, 2 * len(names))

        self.ax_text = self.figure.add_subplot(grid[:, 1])
        self.ax_text.axis("off")
        self._status = self.ax_text.text(
            0.0,
            1.0,
            "",
            fontsize=8,
            family="monospace",
            va="top",
            transform=self.ax_text.transAxes,
        )
        self._animation: FuncAnimation | None = None
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)

    def _add_channel(self, grid: GridSpec, row: int, name: str) -> None:
        """EMG panel and tremor panel for one channel."""
        processing = self.acquisition.config.processing
        shared = self.axes[0] if self.axes else None
        ax = self.figure.add_subplot(grid[2 * row, 0], sharex=shared)
        (signal,) = ax.plot([], [], linewidth=0.5, color=SIGNAL_COLOR, label="EMG")
        (envelope,) = ax.plot(
            [],
            [],
            linewidth=1.6,
            color=ENVELOPE_COLOR,
            label=f"slow envelope (RMS {processing.slow_window_s * 1000:g} ms)",
        )
        (fast,) = ax.plot(
            [],
            [],
            linewidth=0.9,
            color=FAST_ENVELOPE_COLOR,
            label=f"fast envelope ({processing.fast_cutoff_hz:g} Hz)",
        )
        ax.set_ylabel(f"{name}\nmV")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelbottom=False)

        low, high = processing.tremor_band_hz
        tremor_ax = self.figure.add_subplot(grid[2 * row + 1, 0], sharex=shared or ax)
        (tremor,) = tremor_ax.plot(
            [], [], linewidth=1.4, color=TREMOR_COLOR, label=f"tremor depth ({low:g}-{high:g} Hz)"
        )
        tremor_ax.set_ylabel("depth", color=TREMOR_COLOR)
        tremor_ax.grid(True, alpha=0.3)
        tremor_ax.tick_params(labelbottom=False)
        ratio_ax = tremor_ax.twinx()
        (ratio,) = ratio_ax.plot(
            [], [], linewidth=0.9, color=RATIO_COLOR, label="ratio (1 = rhythmic)"
        )
        # ~1 for rhythmic tremor, 0.2-0.6 otherwise; brief spikes at transitions are clipped
        ratio_ax.set_ylim(0.0, 1.5)
        ratio_ax.set_ylabel("ratio", color=RATIO_COLOR)

        self.axes.append(ax)
        self.tremor_axes.append(tremor_ax)
        self.ratio_axes.append(ratio_ax)
        self.signal_lines.append(signal)
        self.envelope_lines.append(envelope)
        self.fast_envelope_lines.append(fast)
        self.tremor_lines.append(tremor)
        self.ratio_lines.append(ratio)

    def _add_controller(self, grid: GridSpec, row: int) -> tuple[Axes, dict[str, Line2D]]:
        """The controller panel: grasp activation, tremor measure and stiffness command."""
        controller = self.acquisition.config.controller
        ax = self.figure.add_subplot(grid[row, 0], sharex=self.axes[0])
        lines = {}
        (lines["grasp"],) = ax.plot(
            [], [], linewidth=1.2, color=ENVELOPE_COLOR, label="grasp activation"
        )
        (lines["tremor"],) = ax.plot(
            [], [], linewidth=1.2, color=TREMOR_COLOR, label=f"tremor {controller.tremor_measure}"
        )
        (lines["stiffness"],) = ax.plot(
            [], [], linewidth=2.2, color=STIFFNESS_COLOR, label="stiffness (control signal)"
        )
        ax.axhline(
            controller.grasp_on_threshold, color=ENVELOPE_COLOR, linestyle="--", linewidth=0.8
        )
        ax.axhline(
            controller.tremor_on_threshold, color=TREMOR_COLOR, linestyle="--", linewidth=0.8
        )
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("controller\n0-1")
        ax.set_xlabel("Seconds before now")
        ax.grid(True, alpha=0.3)
        # Below the panel, so the legend never covers the traces
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), fontsize=7, ncol=3)
        self.figure.subplots_adjust(bottom=0.1)
        return ax, lines

    @staticmethod
    def help_text() -> str:
        """Key bindings."""
        return "Keys: r raw/filtered | c recalibrate | space pause | q quit"

    def _title(self) -> str:
        processing = self.acquisition.config.processing
        if self.show_raw:
            return "Raw EMG"
        notch = f" + {processing.notch_hz:g} Hz notch" if processing.notch_enabled else ""
        return (
            f"Filtered EMG ({processing.bandpass_low_hz:g}-{processing.bandpass_high_hz:g} Hz"
            f"{notch})"
        )

    def _on_key(self, event: Event) -> None:
        if not isinstance(event, KeyEvent) or event.key is None:
            return
        actions: dict[str, Callable[[], None]] = {
            "r": self._toggle_raw,
            "c": self.acquisition.recalibrate,
            " ": self._toggle_pause,
            "q": lambda: plt.close(self.figure),
        }
        action = actions.get(event.key.lower())
        if action is not None:
            action()

    def _toggle_raw(self) -> None:
        self.show_raw = not self.show_raw

    def _toggle_pause(self) -> None:
        self.paused = not self.paused

    def _update_traces(self, snapshot: ViewerSnapshot) -> None:
        latest = snapshot.time[-1] if snapshot.time.size else 0.0
        seconds_ago = snapshot.time - latest
        signals = snapshot.raw if self.show_raw else snapshot.filtered
        for row, ax in enumerate(self.axes):
            self.signal_lines[row].set_data(seconds_ago, signals[row])
            # The envelopes belong to the filtered signal; raw EMG may carry a DC offset
            for line, values in (
                (self.envelope_lines[row], snapshot.envelope[row]),
                (self.fast_envelope_lines[row], snapshot.fast_envelope[row]),
            ):
                line.set_data(seconds_ago, values)
                line.set_visible(not self.show_raw)
            depth = snapshot.tremor_depth[row]
            self.tremor_lines[row].set_data(seconds_ago, depth)
            self.ratio_lines[row].set_data(seconds_ago, snapshot.tremor_ratio[row])
            if signals.shape[1] == 0:
                continue
            if EMG_Y_LIMITS is not None:
                ax.set_ylim(*EMG_Y_LIMITS)
            else:
                low, high = (float(value) for value in np.percentile(signals[row], [0.1, 99.9]))
                span = max(high - low, 1e-3)
                ax.set_ylim(low - 0.15 * span, high + 0.15 * span)
            # ~0.05 without tremor, ~0.3 or more with it; keep that range visible
            self.tremor_axes[row].set_ylim(0.0, max(float(np.max(depth)) * 1.2, 0.5))
        self.axes[0].set_title(self._title(), fontsize=9)
        self._update_controller(snapshot, latest)

    def _update_controller(self, snapshot: ViewerSnapshot, latest: float) -> None:
        control_ago = snapshot.control_time - latest
        for key, values in (
            ("grasp", snapshot.grasp),
            ("tremor", snapshot.tremor_control),
            ("stiffness", snapshot.stiffness),
        ):
            self.control_lines[key].set_data(control_ago, values)
        # Scale to the visible window only; the buffer may hold more history than shown
        visible = control_ago >= -self.acquisition.config.runtime.monitor_seconds
        peak = max(
            (float(np.max(values[visible])) for values in (snapshot.grasp, snapshot.tremor_control)
             if np.any(visible)),
            default=0.0,
        )
        self.control_ax.set_ylim(0.0, max(1.05, peak * 1.1))
        self.control_ax.set_title(
            f"Controller {snapshot.state}: stiffness {_last(snapshot.stiffness):.2f}   "
            f"[{snapshot.last_transition}]   calibration: {calibration_text(snapshot)}",
            fontsize=8,
            color=STATE_COLORS.get(snapshot.state, STIFFNESS_COLOR),
        )

    def _draw(self, _frame: int) -> None:
        snapshot = self.acquisition.snapshot()
        if not self.paused:
            self._update_traces(snapshot)
        extra = ["", self.help_text()] + (["", "PAUSED (space to resume)"] if self.paused else [])
        self._status.set_text("\n".join(status_lines(snapshot) + extra))
        if not snapshot.running:
            self.figure.suptitle(
                "Acquisition stopped: " + (self.acquisition.error or "see messages"),
                color="#e53935",
            )

    def run(self) -> None:
        """Show the window and refresh until it is closed."""
        self._animation = FuncAnimation(
            self.figure, self._draw, interval=100, cache_frame_data=False
        )
        plt.show()


class ConsoleStatus:  # pylint: disable=too-few-public-methods
    """Prints one compact status line per call."""

    def __init__(self, acquisition: Acquisition) -> None:
        self.acquisition = acquisition

    def __call__(self) -> None:
        snapshot = self.acquisition.snapshot()
        channels = "  ".join(
            f"{name} {rms:7.1f} uV tremor {tremor:5.2f} uV depth {depth:4.2f} "
            f"ratio {ratio:4.2f} {hz:>4s} Hz"
            for name, rms, tremor, depth, ratio, hz in _channel_values(snapshot)
        )
        scenario = snapshot.source_info.get("scenario", "")
        print(
            f"{'connected' if snapshot.connected else 'DISCONNECTED':12s} "
            f"| {snapshot.measured_rate:7.1f} Hz | {channels} "
            f"| {snapshot.state:11s} stiffness {_last(snapshot.stiffness):.2f} "
            f"grasp {_last(snapshot.grasp):.2f} "
            f"| cal {calibration_text(snapshot)} "
            f"| missing {snapshot.counters.get('missing samples', 0)} | {scenario}",
            flush=True,
        )
