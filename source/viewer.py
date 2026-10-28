"""
Live EMG viewer for a running Acquisition.

LiveViewer: a matplotlib window, refreshed about ten times per second. Per channel:
  - EMG panel: filtered (or raw) EMG with the slow envelope (moving RMS, voluntary
    effort) and the fast envelope (follows tremor bursts).
  - Tremor panel: tremor depth (4-6 Hz fluctuation of the fast envelope relative
    to the slow envelope, so independent of effort) and the tremor ratio
    (rhythmicity: ~1 for rhythmic tremor, ~0.5 for broadband).
  The status panel adds the tremor amplitude in uV and the dominant frequency of
  each fast envelope over the last processing.frequency_window_s.
  A status panel shows connection, sample rate, current values and counters.
  Keys: r = raw / filtered, space = pause, q = quit.

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

SIGNAL_COLOR = "#1f77b4"
ENVELOPE_COLOR = "#d62728"
FAST_ENVELOPE_COLOR = "#ff7f0e"
TREMOR_COLOR = "#7b1fa2"
RATIO_COLOR = "#757575"


def _latest(values: NDArray[np.float64], scale: float = 1.0) -> list[float]:
    """Latest value per channel of an (n_channels, n) history, times scale."""
    if values.size == 0:
        return [0.0] * values.shape[0]
    return [float(value) * scale for value in values[:, -1]]


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
        self.figure = plt.figure(figsize=(14, max(5.5, 3.0 * len(names) + 1.0)))
        self.figure.canvas.manager.set_window_title("Live EMG")  # type: ignore[union-attr]
        grid = self.figure.add_gridspec(
            2 * len(names),
            2,
            width_ratios=(3.0, 1.0),
            height_ratios=(2.0, 1.0) * len(names),
            hspace=0.35,
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
        self.tremor_axes[-1].tick_params(labelbottom=True)
        self.tremor_axes[-1].set_xlabel("Seconds before now")

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

    @staticmethod
    def help_text() -> str:
        """Key bindings."""
        return "Keys: r raw/filtered | space pause | q quit"

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
            low, high = (float(value) for value in np.percentile(signals[row], [0.1, 99.9]))
            span = max(high - low, 1e-3)
            ax.set_ylim(low - 0.15 * span, high + 0.15 * span)
            # ~0.05 without tremor, ~0.3 or more with it; keep that range visible
            self.tremor_axes[row].set_ylim(0.0, max(float(np.max(depth)) * 1.2, 0.5))
        self.axes[0].set_title(self._title(), fontsize=9)

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
            f"| missing {snapshot.counters.get('missing samples', 0)} | {scenario}",
            flush=True,
        )
