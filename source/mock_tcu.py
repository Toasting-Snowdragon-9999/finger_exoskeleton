"""
A fake Trigno Control Utility, for tests and for trying run_emg.py without the hardware.

It answers the commands source/trigno.py sends, the way the real TCU 3.6.0 did
on the bench, and streams a recording (in volts) on the EMG port in 13.5 ms
frames of 27 samples, starting over at the end. It does not model the real
TCU's timing or errors.
"""

import socket
import threading
import time

import numpy as np

FRAME_SAMPLES = 27
FRAME_SECONDS = 0.0135  # 27 samples per 13.5 ms = 2000 Hz
COLUMNS = 16


class MockTCU:
    """Serves the command port and the EMG port on two free local ports."""

    def __init__(self, slots: list[int], signal_volts: np.ndarray) -> None:
        self.slots = slots  # the paired slots; slot n is in column n - 1 of the EMG port
        self.signal = signal_volts  # (len(slots), n) in volts
        self.commands: list[str] = []  # every command received, for tests
        self.streaming = False
        self.running = True
        self.command_server = self.listen()
        self.data_server = self.listen()
        self.command_port = self.command_server.getsockname()[1]
        self.emg_port = self.data_server.getsockname()[1]
        self.threads = [threading.Thread(target=self.serve_commands, daemon=True),
                        threading.Thread(target=self.serve_data, daemon=True)]

    @staticmethod
    def listen() -> socket.socket:
        """A server socket on a free local port."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(0.1)
        return server

    def start(self) -> None:
        """Start serving."""
        for thread in self.threads:
            thread.start()

    def stop(self) -> None:
        """Stop serving and close the ports."""
        self.running = False
        for thread in self.threads:
            thread.join(timeout=2.0)
        self.command_server.close()
        self.data_server.close()

    def reply(self, command: str) -> str:  # pylint: disable=too-many-return-statements
        """The TCU's answer to one command."""
        command = command.upper()
        words = command.split()
        if command == "START":
            self.streaming = True
            return "OK"
        if command in ("STOP", "QUIT"):
            self.streaming = False
            return "OK"
        fixed = {"BACKWARDS COMPATIBILITY?": "YES", "UPSAMPLING?": "UPSAMPLING ON",
                 "FRAME INTERVAL?": "0.0135", "MAX SAMPLES EMG?": "17", "ENDIANNESS?": "LITTLE"}
        if command in fixed:
            return fixed[command]
        if len(words) >= 3 and words[0] == "SENSOR":
            slot, question = int(words[1]), " ".join(words[2:])
            if question == "PAIRED?":
                return "YES" if slot in self.slots else "NO"
            answers = {"EMGCHANNELCOUNT?": "1", "STARTINDEX?": str(slot), "CHANNEL 1 UNITS?": "V"}
            return answers.get(question, "INVALID COMMAND")
        return "INVALID COMMAND"

    def serve_commands(self) -> None:
        """Answer commands from one client at a time."""
        while self.running:
            try:
                client, _ = self.command_server.accept()
            except (TimeoutError, OSError):
                continue
            with client:
                client.settimeout(0.1)
                client.sendall(b"Mock Trigno Control Utility\r\n\r\n")
                buffer = b""
                while self.running:
                    try:
                        chunk = client.recv(4096)
                    except TimeoutError:
                        continue
                    if not chunk:
                        break
                    buffer += chunk
                    while b"\r\n\r\n" in buffer:
                        packet, buffer = buffer.split(b"\r\n\r\n", 1)
                        command = packet.decode("ascii").strip()
                        self.commands.append(command)
                        client.sendall(f"{self.reply(command)}\r\n\r\n".encode("ascii"))

    def serve_data(self) -> None:
        """Stream the signal to the EMG port client while streaming is on."""
        client = None
        position = 0
        next_time = time.monotonic()
        while self.running:
            if client is None:
                try:
                    client, _ = self.data_server.accept()
                except (TimeoutError, OSError):
                    continue
            if not self.streaming:
                time.sleep(0.005)
                next_time = time.monotonic()
                continue
            frame = np.zeros((FRAME_SAMPLES, COLUMNS), dtype="<f4")
            columns = np.arange(position, position + FRAME_SAMPLES) % self.signal.shape[1]
            for row, slot in enumerate(self.slots):
                frame[:, slot - 1] = self.signal[row, columns]
            position += FRAME_SAMPLES
            try:
                client.sendall(frame.tobytes())
            except OSError:
                client = None
                continue
            next_time += FRAME_SECONDS
            time.sleep(max(0.0, next_time - time.monotonic()))
        if client is not None:
            client.close()
