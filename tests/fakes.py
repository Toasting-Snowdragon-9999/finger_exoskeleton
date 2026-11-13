"""Test helpers: made-up EMG and a fake Dynamixel bus."""

import numpy as np

import settings
from settings import Channel
from source.gru.config import GRUConfig

RATE = 2000.0
# Two sensors for the tests, whatever settings.CHANNELS is set to on the bench
TWO_SENSORS = [Channel("FDS", "flexor", 15), Channel("EDC", "extensor", 14)]


def make_emg(segments: list[tuple[float, float]], n_channels: int = 2, tremor_hz: float = 0.0,
             seed: int = 0) -> np.ndarray:
    """
    Noise-like EMG in mV, (n_channels, n). segments: (seconds, RMS level in mV) one after the
    other. With tremor_hz the level bursts on and off at that frequency.
    """
    levels = np.concatenate([np.full(round(seconds * RATE), level) for seconds, level in segments])
    if tremor_hz:
        times = np.arange(len(levels)) / RATE
        levels = levels * (1 + np.sin(2 * np.pi * tremor_hz * times))
    noise = np.random.default_rng(seed).standard_normal((n_channels, len(levels)))
    return noise * levels


class FakeMotors:
    """The part of Motors (DynamixelHardwareInterface.py) that the motor code uses."""

    POS_CONTROL = 3
    EXT_POS_CONTROL = 4
    MODES = {"pos": 3, "ext": 4, "vel": 1, "cur": 0}

    def __init__(self, port: str = "COM0", baudrate: int = 0, init_mode: str | None = None,
                 position: int = 4600) -> None:
        self.port = port
        self.baudrate = baudrate
        self.init_mode = init_mode
        self.motor_ids = [1, settings.MOTOR_ID]
        self.control_mode = [1, 1]  # velocity mode, as found on the bench
        self.torque = [False, False]
        self.position = position
        self.sent: list[int] = []
        self.closed = False

    def disable_torque(self, motor_ids: list[int]) -> None:
        """Torque off."""
        for motor_id in motor_ids:
            self.torque[self.motor_ids.index(motor_id)] = False

    def enable_torque(self, motor_ids: list[int]) -> None:
        """Torque on."""
        for motor_id in motor_ids:
            self.torque[self.motor_ids.index(motor_id)] = True

    def set_cont_mode(self, mode: str, motor_ids: list[int]) -> None:
        """Operating mode; like the real motor only with torque off."""
        for motor_id in motor_ids:
            index = self.motor_ids.index(motor_id)
            assert not self.torque[index], "mode change with torque on"
            self.control_mode[index] = self.MODES[mode]

    def get_position(self, motor_id: int) -> int:
        """Present position in ticks."""
        assert motor_id == settings.MOTOR_ID
        return self.position

    # pylint: disable-next=invalid-name  # the name Motors uses
    def sendMotorCommand(self, motor_id: int, command: int) -> None:
        """Goal position; the fake motor is there at once."""
        assert motor_id == settings.MOTOR_ID
        self.sent.append(command)
        self.position = command

    def close(self) -> None:
        """Close the port."""
        self.closed = True


class FakeModel:
    """A GRU model stand-in without PyTorch: always answers the same two numbers."""

    def __init__(self, roles: list[str] | None = None, intention: float = 0.5,
                 probability: float = 0.1) -> None:
        self.config = GRUConfig(channel_roles=roles or ["flexor"])
        self.intention = intention
        self.probability = probability
        self.frames_seen = 0
        self.hidden_seen: list = []  # the hidden state passed in at every call

    def step(self, frames: np.ndarray, hidden: object) -> tuple[np.ndarray, np.ndarray, object]:
        """Like IntentionTremorGRU.step: per-frame outputs and the new hidden state."""
        self.hidden_seen.append(hidden)
        self.frames_seen += len(frames)
        return (np.full(len(frames), self.intention), np.full(len(frames), self.probability),
                self.frames_seen)
