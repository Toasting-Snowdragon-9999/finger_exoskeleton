"""Command-line entry point for direct Dynamixel motor control and EMG control."""

import argparse
import time
from collections.abc import Sequence

from source.DynamixelHardwareInterface import Motors
from source.controllers import ProportionalEMGController, ThresholdEMGController


def simulated_activation(t: float, period_s: float = 8.0) -> float:
    """Bench-test stand-in for the EMG activation (0..1): rest, ramp up, hold, ramp down."""
    p = (t % period_s) / period_s * 8.0
    if p < 2:
        return 0.0
    if p < 3:
        return p - 2
    if p < 5:
        return 1.0
    if p < 6:
        return 6 - p
    return 0.0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Connect to the Dynamixel motor; send one command or run an EMG controller."
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
        help="Keep torque enabled for this many seconds after sending a command "
        "(with --controller: run time in seconds, 0 = until Ctrl+C)",
    )

    emg = parser.add_argument_group("EMG controllers (forces position mode)")
    emg.add_argument(
        "--controller",
        choices=("proportional", "threshold"),
        help="Run an EMG controller instead of sending a single command",
    )
    emg.add_argument("--closed-pos", type=int, help="Motor position (ticks) with the grasp closed")
    emg.add_argument(
        "--open-pos",
        type=int,
        help="Motor position (ticks) with the finger open (default: position at start)",
    )
    emg.add_argument("--max-speed", type=float, default=600.0, help="Max ticks/s (default: 600)")
    emg.add_argument("--rate", type=float, default=50.0, help="Control loop rate in Hz (default: 50)")
    emg.add_argument(
        "--emg-sim",
        action="store_true",
        help="Drive the controller with a simulated activation signal (bench test)",
    )
    emg.add_argument("--deadband", type=float, default=0.1, help="[proportional] rest deadband 0..1")
    emg.add_argument("--gamma", type=float, default=1.0, help="[proportional] response curve exponent")
    emg.add_argument("--on-threshold", type=float, default=0.5, help="[threshold] close above this")
    emg.add_argument("--off-threshold", type=float, default=0.3, help="[threshold] open below this")
    emg.add_argument(
        "--toggle", action="store_true", help="[threshold] each contraction toggles open/closed"
    )
    return parser


def run_emg_controller(args, motors, motor_id) -> None:
    if args.closed_pos is None:
        raise SystemExit("--closed-pos is required with --controller")
    if not args.emg_sim:
        raise SystemExit(
            "No EMG source connected here: the activation signal comes from the EMG "
            "pipeline. Use --emg-sim for a bench test, or call controller.step(activation) / "
            "controller.run(get_activation) from the EMG code."
        )
    open_pos = args.open_pos if args.open_pos is not None else motors.get_position(motor_id)

    common = dict(open_pos=open_pos, closed_pos=args.closed_pos,
                  max_speed=args.max_speed, rate_hz=args.rate)
    if args.controller == "proportional":
        controller = ProportionalEMGController(
            motors, motor_id, deadband=args.deadband, gamma=args.gamma, **common)
    else:
        controller = ThresholdEMGController(
            motors, motor_id,
            on_threshold=args.on_threshold, off_threshold=args.off_threshold,
            mode="toggle" if args.toggle else "hold", **common)

    t0 = time.perf_counter()
    print(f"{args.controller} controller: open={open_pos}, closed={args.closed_pos}. "
          "Ctrl+C to stop (the finger returns to open).")
    controller.run(lambda: simulated_activation(time.perf_counter() - t0),
                   duration=args.hold if args.hold > 0 else None)


def main(argv: Sequence[str] | None = None) -> int:
    """Connect to the motor, show feedback, and optionally send a command / run EMG control."""
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
        motors.set_cont_mode("pos" if args.controller else args.mode)
        motors.enable_torque()

        print(f"Motor {motor_id} position: {motors.get_position(motor_id)}")
        if args.controller:
            run_emg_controller(args, motors, motor_id)
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