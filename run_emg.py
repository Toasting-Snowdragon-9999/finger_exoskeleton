"""
Acquire EMG and show it live. The settings are in applied_config.py.

Examples:
  python run_emg.py --source trigno                          (Delsys Trigno through the TCU)
  python run_emg.py --source trigno --out measurements       (each run in a new session folder)
  python run_emg.py --source trigno --out measurements --label grip --note "subject 1"
  python run_emg.py --source trigno --save logs/session.csv  (record the raw EMG to one CSV)
  python run_emg.py --source trigno --mock-tcu               (protocol demo, no hardware)
  python run_emg.py --source synthetic
  python run_emg.py --source recorded --file 10_raw.csv
  python run_emg.py --source recorded --file measurements/20261023-140512_trigno_grip  (replay)
  python run_emg.py --source trigno --headless --duration 30 (status lines, no window)
  python run_emg.py --source trigno --motor                  (grasp activation drives the motor)
  python run_emg.py --source trigno --calibrate              (calibrate now, replace the saved one)

Every mode runs the controller (calibration -> activation -> state machine) and
shows its stiffness command, the control signal. Nothing physical is driven
unless --motor is given: then the grasp activation moves the tendon motor
between open and closed (applied_config.MOTOR, extended position mode).
"""

import argparse
import contextlib
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import applied_config
from source.acquisition import Acquisition
from source.calibration_store import DEFAULT_PATH as DEFAULT_CALIBRATION_PATH
from source.calibration_store import CalibrationStore
from source.control_stage import ControlStage
from source.config import (
    SOURCE_KINDS,
    AppConfig,
    ConfigError,
    MotorConfig,
    validate_config,
    validate_motor_config,
)
from source.mock_tcu import MockSensor, MockTCU, MockTCUSettings
from source.motor_stage import MotorStage, connect_controller, grasp_signal
from source.emg_source import EMGSource, SourceError
from source.sensor_mock import RecordedSource, SyntheticSource
from source.session import RecordingError, SessionLog, create_session_dir, find_recording
from source.trigno import TrignoSource

EXIT_CONFIG, EXIT_SOURCE, EXIT_RUNTIME, EXIT_MOTOR = 2, 3, 4, 5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        choices=SOURCE_KINDS,
        default="synthetic",
        help="EMG source (default: synthetic)",
    )
    parser.add_argument(
        "--file",
        help="with --source recorded: a session folder saved with --out, a CSV saved with "
        "--save, or a dataset CSV (name in data/sEMG_online or a path)",
    )
    parser.add_argument(
        "--no-loop", action="store_true", help="with --source recorded: stop at the end"
    )
    parser.add_argument("--host", help="host running the Trigno Control Utility")
    parser.add_argument(
        "--mock-tcu",
        action="store_true",
        help="with --source trigno: start a local mock TCU serving synthetic EMG",
    )
    parser.add_argument("--headless", action="store_true", help="no window; print status lines")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="with --headless: seconds to run (default: until Ctrl+C)",
    )
    recording = parser.add_mutually_exclusive_group()
    recording.add_argument(
        "--out",
        type=Path,
        default=None,
        help="measurement folder: each run is saved in a new timestamped session folder in it "
        "(emg.csv + session.json)",
    )
    recording.add_argument(
        "--save", type=Path, default=None, help="record the raw EMG to this CSV file"
    )
    parser.add_argument("--label", help="with --out: short name added to the session folder")
    parser.add_argument("--note", help="with --out: free-text note stored in session.json")
    parser.add_argument(
        "--motor",
        action="store_true",
        help="drive the tendon motor from the grasp activation (settings: applied_config.MOTOR)",
    )
    parser.add_argument("--motor-port", help="with --motor: serial port of the motor")
    parser.add_argument(
        "--calibration",
        type=Path,
        help="calibration file: loaded if it fits, saved after a successful calibration "
        f"(default with --source trigno: {DEFAULT_CALIBRATION_PATH}; other sources: none)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="calibrate now even if a saved calibration exists, and save the result",
    )
    return parser.parse_args(argv)


def calibration_store(args: argparse.Namespace) -> CalibrationStore | None:
    """Where the calibration is kept between runs, or None (calibrate every run)."""
    path = args.calibration or (DEFAULT_CALIBRATION_PATH if args.source == "trigno" else None)
    return None if path is None else CalibrationStore(path, use_saved=not args.calibrate)


def build_source(args: argparse.Namespace, config: AppConfig) -> tuple[EMGSource, MockTCU | None]:
    """Create the requested source (and a mock TCU if asked for)."""
    if args.source == "synthetic":
        return SyntheticSource(config.synthetic, config.channels), None
    if args.source == "recorded":
        return RecordedSource(config.recorded, config.channels), None

    trigno = config.trigno
    if args.host:
        trigno = replace(trigno, host=args.host)
    mock = None
    if args.mock_tcu:
        generator = SyntheticSource(replace(config.synthetic, realtime=False), config.channels)
        settings = MockTCUSettings(
            sensors=[
                MockSensor(slot=channel.trigno_slot, position=channel.trigno_slot - 1)
                for channel in config.channels
            ],
            start_index_base=trigno.start_index_base,
        )
        mock = MockTCU(settings, signal=lambda first, n: generator.generate(first, n) / 1000.0)
        mock.start()
        trigno = replace(
            trigno, host="127.0.0.1", command_port=mock.command_port, emg_port=mock.emg_port
        )
        print(
            f"Mock TCU listening on 127.0.0.1:{mock.command_port} / {mock.emg_port} "
            "(documented protocol subset, NOT hardware)"
        )
    return TrignoSource(trigno, config.channels), mock


def with_recording(args: argparse.Namespace, config: AppConfig) -> AppConfig:
    """
    The configuration for --source recorded --file: for our own recordings the
    channels and sample rate they were recorded with. Raises RecordingError.
    """
    if args.source != "recorded":
        return config
    recorded = replace(config.recorded, loop=config.recorded.loop and not args.no_loop)
    if not args.file:
        return replace(config, recorded=recorded)
    recording = find_recording(Path(args.file), config.channels)
    if recording is None:  # a dataset CSV
        return replace(config, recorded=replace(recorded, file=args.file))
    print(f"Replaying {recording.emg_path} at {recording.sample_rate_hz:g} Hz")
    config = replace(
        config,
        channels=recording.channels,
        recorded=replace(
            recorded,
            file=str(recording.emg_path.resolve()),
            sample_rate_hz=recording.sample_rate_hz,
        ),
        # Our sessions start with the live calibration routine, not the dataset timeline
        calibration=applied_config.CALIBRATION["trigno"],
    )
    validate_config(config)
    return config


def calibration_instructions(config: AppConfig) -> str:
    """What the user must do at the start for the controller to calibrate."""
    rest, grip = config.calibration.baseline_window_s, config.calibration.reference_window_s
    return (
        f"Calibration: rest from {rest[0]:g} to {rest[1]:g} s, then hold a firm grip from "
        f"{grip[0]:g} to {grip[1]:g} s. The controller arms when it succeeds; press 'c' to repeat."
    )


def open_session(args: argparse.Namespace, config: AppConfig) -> SessionLog | None:
    """A new session folder in --out, or None without --out. Raises OSError."""
    if args.out is None:
        return None
    directory = create_session_dir(args.out, args.source, args.label)
    return SessionLog(directory, config, args.label, args.note)


def start(acquisition: Acquisition, session: SessionLog | None, mock: MockTCU | None) -> None:
    """Start the acquisition and the session log. On failure clean up and re-raise."""
    try:
        acquisition.start()
        if session is not None:
            session.started(acquisition.source)
    except (SourceError, OSError):
        acquisition.stop()
        if mock is not None:
            mock.stop()
        if session is not None:
            with contextlib.suppress(OSError):  # only removed while still empty
                session.directory.rmdir()
        raise


def show(args: argparse.Namespace, acquisition: Acquisition) -> None:
    """Show the running acquisition live, or print status lines when headless."""
    if args.headless:
        from source.viewer import ConsoleStatus  # pylint: disable=import-outside-toplevel

        duration = args.duration if args.duration is not None else float("inf")
        acquisition.run_for(duration, on_tick=ConsoleStatus(acquisition))
    else:
        from source.viewer import LiveViewer  # pylint: disable=import-outside-toplevel

        viewer = LiveViewer(acquisition)
        print(viewer.help_text())
        viewer.run()


def finish_session(session: SessionLog, acquisition: Acquisition) -> None:
    """Write the end of a stopped acquisition to session.json."""
    if acquisition.core is None:
        return
    snapshot = acquisition.snapshot()
    recorder = acquisition.core.recorder
    session.finished(
        recorder.rows if recorder is not None else 0,
        snapshot.counters,
        acquisition.core.event_log(),
        acquisition.error,
    )
    print(f"Session saved to {session.directory}")


def motor_config(args: argparse.Namespace) -> MotorConfig | None:
    """The validated motor settings with --motor, else None. Raises ConfigError."""
    if not args.motor:
        return None
    config = applied_config.MOTOR
    if args.motor_port:
        config = replace(config, port=args.motor_port)
    validate_motor_config(config)
    return config


def start_motor(config: MotorConfig, acquisition: Acquisition) -> tuple[MotorStage, Any]:
    """Connect the motor and start the motor thread. Raises on connection errors."""
    motors, controller = connect_controller(config)

    def control() -> ControlStage | None:
        core = acquisition.core
        return None if core is None else core.control

    def message(text: str) -> None:
        if acquisition.core is not None:
            acquisition.core.message(text)
        else:
            print(text)

    signal = grasp_signal(control, config.stale_s, hold_window_s=config.hold_window_s)
    stage = MotorStage(controller, signal, message)
    stage.start()
    print(f"Motor control on {config.port}: {config.controller} controller, "
          f"open {config.open_pos if config.open_pos is not None else 'at start'}, "
          f"closed {config.closed_pos}")
    return stage, motors


def show_with_motor(
    args: argparse.Namespace, acquisition: Acquisition, motor: MotorConfig | None
) -> bool:
    """
    show() with the motor driven meanwhile (with --motor). Returns False if the
    motor could not be started; then nothing is shown.
    """
    if motor is None:
        show(args, acquisition)
        return True
    try:
        stage, motors = start_motor(motor, acquisition)
    except Exception as error:  # pylint: disable=broad-exception-caught
        # Motors raises bare Exception for port and motor errors
        print(f"Cannot start motor control on {motor.port}: {error}", file=sys.stderr)
        return False
    try:
        show(args, acquisition)
    finally:
        stage.stop()  # the finger returns to open before the acquisition stops
        motors.close()
    return True


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = parse_args(argv)
    try:
        config = with_recording(args, applied_config.build_config(source=args.source))
        motor = motor_config(args)
    except (ConfigError, RecordingError) as error:
        replay = isinstance(error, RecordingError)
        print(f"{'Cannot replay' if replay else 'Configuration error'}: {error}", file=sys.stderr)
        return EXIT_SOURCE if replay else EXIT_CONFIG

    try:
        session = open_session(args, config)
    except OSError as error:
        print(f"Cannot create a session folder in {args.out}: {error}", file=sys.stderr)
        return EXIT_CONFIG
    save_path = session.emg_path if session is not None else args.save

    store = calibration_store(args)
    source, mock = build_source(args, config)
    acquisition = Acquisition(config, source, save_path, store)
    try:
        start(acquisition, session, mock)
    except SourceError as error:
        print(f"Cannot start the {args.source} source: {error}", file=sys.stderr)
        return EXIT_SOURCE
    except OSError as error:
        print(f"Cannot create the recording {save_path}: {error}", file=sys.stderr)
        return EXIT_CONFIG
    print(f"Streaming {source.kind} EMG at {source.sample_rate:g} Hz")
    saved, note = (None, "") if store is None else store.load(config)
    if saved is not None:
        print(f"{note}. Press 'c' in the window (or run with --calibrate) to calibrate again.")
    else:
        if note:
            print(note)
        print(calibration_instructions(config))
    if session is not None:
        print(f"Recording session {session.directory}")

    exit_code = 0
    try:
        if not show_with_motor(args, acquisition, motor):
            exit_code = EXIT_MOTOR
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        acquisition.stop()
        if mock is not None:
            mock.stop()

    if session is not None:
        finish_session(session, acquisition)
    elif args.save:
        print(f"Raw EMG saved to {args.save}")
    if acquisition.error:
        print(f"Acquisition error: {acquisition.error}", file=sys.stderr)
        exit_code = EXIT_RUNTIME
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
