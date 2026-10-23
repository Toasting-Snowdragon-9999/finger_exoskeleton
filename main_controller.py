"""Command-line entry point for direct Dynamixel motor control."""

import argparse
import time
from collections.abc import Sequence

from source.DynamixelHardwareInterface import Motors
from source.controllers import TendonTensionController


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Connect to the Dynamixel motor and optionally send one command."
    )
    parser.add_argument("--port", default="COM4", help="Serial port (default: COM4)")
    parser.add_argument(
        "--baudrate", type=int, default=3000000, help="Bus baudrate (default: 3000000)"
    )
    parser.add_argument(
        "--mode",
        choices=("pos", "vel", "cur"),
        default="pos",
        help="Control mode: position, velocity, or current (default: pos)",
    )
    parser.add_argument(
        "--motor-id",
        type=int,
        help="Motor ID to command (default: the first detected motor)",
    )
    parser.add_argument(
        "--command",
        type=int,
        help="Raw Dynamixel command in the selected control mode",
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=0.0,
        help="Keep torque enabled for this many seconds after sending a command",
    )
    parser.add_argument(
        "--tension",
        type=float,
        help="Run the tendon take-up controller with this peak pull current in mA "
        "(keep it LOW: just enough to reel in slack). Forces current mode; "
        "runs for --hold seconds, or until Ctrl+C",
    )
    parser.add_argument(
        "--assist",
        type=float,
        default=0.0,
        help="Optional pay-out assist in mA while the user pulls the tendon out "
        "(cancels gearbox friction; default: 0 = off)",
    )
    parser.add_argument(
        "--probe-period",
        type=float,
        default=1.0,
        help="Seconds between slack checks while idle; 0 = never re-check (default: 1.0)",
    )
    parser.add_argument(
        "--wind-direction",
        type=int,
        choices=(1, -1),
        default=1,
        help="Sign of current that reels the tendon in (default: 1)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Connect to the motor, show feedback, and optionally send a command."""
    args = build_parser().parse_args(argv)
    motors = Motors(port=args.port, baudrate=args.baudrate)

    try:
        motor_id = args.motor_id
        if motor_id is None:
            motor_id = motors.motor_ids[0]
        if motor_id not in motors.motor_ids:
            raise ValueError(f"Motor ID {motor_id} was not detected.")

        # The Dynamixel control table requires torque to be disabled while
        # changing the operating mode.
        motors.disable_torque()
        motors.set_cont_mode("cur" if args.tension is not None else args.mode)
        motors.enable_torque()

        print(f"Motor {motor_id} position: {motors.get_position(motor_id)}")
        if args.tension is not None:
            controller = TendonTensionController(
                motors,
                motor_id,
                tension_ma=args.tension,
                wind_direction=args.wind_direction,
                assist_ma=args.assist,
                probe_period_s=args.probe_period if args.probe_period > 0 else None,
                rate_hz=20,
            )
            print(f"Taking up tendon slack (peak {args.tension} mA). Ctrl+C to stop.")
            controller.run(duration=args.hold if args.hold > 0 else None)
            return 0
        if args.command is not None:
            motors.sendMotorCommand(motor_id, args.command)
            print(f"Sent {args.mode} command {args.command} to motor {motor_id}.")
            if args.hold > 0:
                time.sleep(args.hold)
    finally:
        motors.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())