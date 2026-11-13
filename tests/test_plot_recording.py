"""plot_recording.py: a recording drawn with its labels as colours."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw without opening a window

# pylint: disable=wrong-import-position
import plot_recording  # noqa: E402
import run_emg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "measurements" / "20261105-144738_trigno_to-lange-greb"


def test_cue_texts_decide_the_colour() -> None:
    """HVIL is rest, GRIB a grip, KLEM a squeeze; calibration and waiting are "other"."""
    def row(cue: str, intention: str = "") -> dict:
        return {"cue": cue, "intention": intention, "tremor": ""}

    assert plot_recording.kind_of(row("HVIL: slip og slap af", "0")) == "rest"
    assert plot_recording.kind_of(row("GRIB objektet med normalt greb og hold", "1")) == "grip"
    assert plot_recording.kind_of(row("KLEM HÅRDT om objektet og hold", "1")) == "squeeze"
    assert plot_recording.kind_of(row("KALIBRERING: hvil hånden")) == "other"
    assert plot_recording.kind_of(row("VENT")) == "other"


def test_labels_without_cue_texts_are_coloured_by_intention() -> None:
    """A hand-written labels.csv has no cue column: 0 is rest, above 0 a grip, empty unknown."""
    assert plot_recording.kind_of({"intention": "0", "tremor": "0"}) == "rest"
    assert plot_recording.kind_of({"intention": "0.5", "tremor": "0"}) == "grip"
    assert plot_recording.kind_of({"intention": "", "tremor": "1"}) == "other"


def test_a_recorded_session_is_drawn_with_one_plot_per_channel(tmp_path: Path) -> None:
    """A session made with --cues gives a figure with two channels, and --save writes a PNG."""
    code = run_emg.main(["--source", "recorded", "--file", str(BENCH), "--headless",
                         "--no-loop", "--duration", "3", "--out", str(tmp_path),
                         "--cues", str(ROOT / "protocols" / "greb_1.csv")])
    assert code == 0
    (folder,) = tmp_path.iterdir()
    figure = plot_recording.plot_recording(folder)
    assert len(figure.axes) == 2
    assert figure.axes[0].get_ylabel().startswith("FDS (flexor)")
    assert len(figure.axes[0].patches) == 15  # one coloured span per row of the protocol

    assert plot_recording.main([str(folder), "--save"]) == 0
    assert (folder / "labels.png").stat().st_size > 10000


def test_tremor_labels_get_a_bar() -> None:
    """The committed tremor recording: 7 label rows plus a bar for each of its 3 tremor spans."""
    figure = plot_recording.plot_recording(Path(str(BENCH) + "_tremor"))
    assert len(figure.axes[0].patches) == 7 + 3
