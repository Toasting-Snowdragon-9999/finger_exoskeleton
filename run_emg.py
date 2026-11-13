"""
Read EMG, show it live and (with --motor) drive the tendon motor. Settings: settings.py.

Examples (see kommandoer.txt for the bench commands):
  python run_emg.py --source trigno                               live Delsys Trigno
  python run_emg.py --source trigno --out measurements --label grip   ...and record it
  python run_emg.py --source trigno --calibrate                   calibrate again and save
  python run_emg.py --source trigno --motor --motor-port COM5     the grasp moves the motor
  python run_emg.py --source trigno --mock-tcu                    a fake TCU, no hardware
  python run_emg.py --source recorded --file measurements/<session folder>   replay
  python run_emg.py --source trigno --method gru --model models/gru.pt   the GRU instead of DSP
  add --headless --duration 60 for status lines instead of a window

--method dsp (the default) uses the signal processing; --method gru asks a trained
GRU model (see GRU_README.md) and falls back to the signal processing when the
model is missing or its estimates are not valid.
"""

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

import settings
from source.gru.inference import GRUEstimator, load_model
from source.mock_tcu import MockTCU
from source.motor import GraspSignal, MotorThread, connect_motor
from source.pipeline import Acquisition, Pipeline
from source.recording import Recording, RecordedSource, SessionWriter, new_session_folder
from source.trigno import TrignoSource

# What the fake TCU streams: a recording of two long grips
MOCK_RECORDING = Path(__file__).parent / "measurements" / "20261105-144738_trigno_to-lange-greb"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command-line options."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["trigno", "recorded"], default="trigno")
    parser.add_argument("--file", type=Path, help="with --source recorded: a session folder")
    parser.add_argument("--no-loop", action="store_true",
                        help="with --source recorded: stop at the end instead of starting over")
    parser.add_argument("--host", help="the PC running the Trigno Control Utility")
    parser.add_argument("--mock-tcu", action="store_true",
                        help="with --source trigno: use a fake TCU that plays a recording")
    parser.add_argument("--headless", action="store_true", help="status lines instead of a window")
    parser.add_argument("--duration", type=float, help="with --headless: seconds to run")
    parser.add_argument("--out", type=Path, help="record the run in a new session folder in here")
    parser.add_argument("--label", help="with --out: added to the session folder name")
    parser.add_argument("--motor", action="store_true", help="drive the tendon motor")
    parser.add_argument("--motor-port", default=settings.MOTOR_PORT,
                        help=f"serial port of the motor (default {settings.MOTOR_PORT})")
    parser.add_argument("--calibration", type=Path,
                        help="calibration file: used if it fits, written after a calibration "
                             f"(default for live Trigno: {settings.CALIBRATION_FILE})")
    parser.add_argument("--calibrate", action="store_true",
                        help="calibrate now even if a saved calibration exists")
    parser.add_argument("--method", choices=["dsp", "gru"], default="dsp",
                        help="dsp: signal processing (default); gru: a trained GRU model")
    parser.add_argument("--model", type=Path, default=Path(settings.GRU_MODEL_FILE),
                        help=f"with --method gru: the model (default {settings.GRU_MODEL_FILE})")
    return parser.parse_args(argv)


def start_mock_tcu() -> MockTCU:
    """Start a fake TCU that plays MOCK_RECORDING."""
    recording = Recording(MOCK_RECORDING)
    slots = [channel.slot for channel in settings.CHANNELS]
    mock = MockTCU(slots, recording.emg[:len(slots)] / 1000.0)  # mV -> V
    mock.start()
    print(f"Fake TCU on ports {mock.command_port}/{mock.emg_port}, playing {MOCK_RECORDING.name}")
    return mock


def load_gru_model(args: argparse.Namespace) -> Any:
    """The GRU model for --method gru, or None: then the signal processing is used."""
    if args.method != "gru":
        return None
    model, note = load_model(args.model)
    if model is None:
        print(f"GRU not available: {note}. Using the signal processing (DSP).")
    elif args.motor and model.config.data_kind == "synthetic":
        print("This GRU model was trained on synthetic data (a software test), so it is not "
              "used with --motor. Using the signal processing (DSP).")
        model = None
    else:
        print(note)
    return model


def calibration_file(args: argparse.Namespace) -> Path | None:
    """The calibration file to use: --calibration, else the default for live Trigno runs."""
    if args.calibration is not None:
        return args.calibration
    if args.source == "trigno" and not args.mock_tcu:
        return Path(settings.CALIBRATION_FILE)
    return None  # replays and the fake TCU calibrate every run


def start_source(args: argparse.Namespace) -> tuple[RecordedSource | TrignoSource,
                                                     MockTCU | None]:
    """Start the EMG source (and the fake TCU with --mock-tcu)."""
    mock = None
    source: RecordedSource | TrignoSource
    if args.source == "recorded":
        source = RecordedSource(args.file, loop=not args.no_loop)
        print(f"Replaying {args.file} at {source.sample_rate:g} Hz")
    else:
        host = args.host or settings.TRIGNO_HOST
        ports = (settings.TRIGNO_COMMAND_PORT, settings.TRIGNO_EMG_PORT)
        if args.mock_tcu:
            mock = start_mock_tcu()
            host, ports = "127.0.0.1", (mock.command_port, mock.emg_port)
        source = TrignoSource(settings.CHANNELS, host, *ports)
    source.start()
    print(f"Streaming {source.name} EMG at {source.sample_rate:g} Hz")
    return source, mock


def show_until_done(args: argparse.Namespace, pipeline: Pipeline, acquisition: Acquisition) -> None:
    """The live window, or a status line every second with --headless."""
    if not args.headless:
        from source.viewer import LiveViewer  # pylint: disable=import-outside-toplevel
        LiveViewer(pipeline, acquisition).run()
        return
    from source.viewer import status_line  # pylint: disable=import-outside-toplevel
    end = time.monotonic() + (args.duration if args.duration is not None else np.inf)
    while acquisition.running and time.monotonic() < end:
        time.sleep(min(1.0, max(0.0, end - time.monotonic())))
        print(status_line(pipeline), flush=True)


def main(argv: list[str] | None = None) -> int:
    """Run until the window is closed, --duration has passed or Ctrl+C; returns the exit code."""
    args = parse_args(argv)
    model = load_gru_model(args)  # before the source starts: loading PyTorch takes a moment

    source, mock = start_source(args)

    estimator = None
    if model is not None:
        estimator = GRUEstimator(model, source.channels, source.sample_rate)
    pipeline = Pipeline(source.channels, source.sample_rate, calibration_file(args),
                        use_saved=not args.calibrate, estimator=estimator)
    if pipeline.loaded_calibration:
        print("Press 'c' in the window (or run with --calibrate) to calibrate again.")
    else:
        rest, grip = settings.REST_WINDOW_S, settings.GRIP_WINDOW_S
        print(f"Calibration: rest from {rest[0]:g} to {rest[1]:g} s, then hold a firm grip from "
              f"{grip[0]:g} to {grip[1]:g} s. Then relax: the controller arms.")

    session = None
    if args.out is not None:
        folder = new_session_folder(args.out, source.name, args.label)
        session = SessionWriter(folder, source.channels, source.sample_rate, source.name,
                                source.info(), args.label)
        print(f"Recording to {folder}")

    acquisition = Acquisition(source, pipeline, session)
    acquisition.start()
    motors = motor_thread = None
    try:
        if args.motor:
            motors, controller = connect_motor(args.motor_port)
            motor_thread = MotorThread(controller, GraspSignal(pipeline), pipeline.message)
            motor_thread.start()
        show_until_done(args, pipeline, acquisition)
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        # The finger opens first, then everything else stops
        if motor_thread is not None:
            motor_thread.stop()
        if motors is not None:
            motors.close()
        acquisition.stop()
        if mock is not None:
            mock.stop()
        if session is not None:
            calibration = pipeline.calibration.to_dict() if pipeline.calibration else None
            session.finish(pipeline.messages, acquisition.error, calibration, args.method)
            print(f"Session saved to {session.folder}")
    return 1 if acquisition.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
