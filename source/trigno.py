"""
Live EMG from the Delsys Trigno system, through the Trigno Control Utility (TCU).

The TCU runs on a Windows PC and is a TCP server (Trigno SDK guide MAN-025-3-5):

  command port 50040   text commands ending in two CR LF; each gets a one-line reply
  EMG port 50043       16 channels, 4-byte little-endian floats in volts, one row per sample

Start-up asks the TCU for its settings and, per configured slot, whether a
sensor is paired, where its EMG is in the 16 columns (STARTINDEX - 1, checked on
the hardware) and its units. Then it sends START. The sensors and their pairing
are set up in the TCU itself; this client only asks and starts/stops.
"""

import socket
import time

import numpy as np

import settings
from settings import Channel
from source.signal_processing import EMGBlock


class TrignoSource:
    """EMG blocks from the TCU's EMG port."""

    name = "trigno"

    def __init__(self, channels: list[Channel], host: str = settings.TRIGNO_HOST,
                 command_port: int = settings.TRIGNO_COMMAND_PORT,
                 emg_port: int = settings.TRIGNO_EMG_PORT) -> None:
        self.channels = channels
        self.host = host
        self.command_port = command_port
        self.emg_port = emg_port
        self.sample_rate = 0.0  # known after start()
        self.banner = ""
        self.columns: list[int] = []  # column of each channel in the 16 EMG columns
        self.commands: socket.socket | None = None
        self.data: socket.socket | None = None
        self.command_buffer = b""
        self.data_buffer = b""
        self.samples_read = 0
        self.streaming = False

    # --- command port ---------------------------------------------------------------------------

    def read_line(self) -> str:
        """The next non-empty line from the command port. A STOPPED notice is skipped."""
        assert self.commands is not None
        while True:
            while b"\r\n" not in self.command_buffer:
                chunk = self.commands.recv(4096)
                if not chunk:
                    raise ConnectionError("The TCU closed the command connection")
                self.command_buffer += chunk
            line, self.command_buffer = self.command_buffer.split(b"\r\n", 1)
            text = line.decode("ascii", errors="replace").strip()
            if text and text != "STOPPED":
                return text

    def send(self, command: str) -> str:
        """Send a command and return the TCU's reply."""
        assert self.commands is not None
        self.commands.sendall(f"{command}\r\n\r\n".encode("ascii"))
        return self.read_line()

    def ask(self, question: str) -> str:
        """The last word of the reply to a question, e.g. "YES" or "2000"."""
        return self.send(question).split()[-1].upper()

    # --- start and stop -------------------------------------------------------------------------

    def start(self) -> None:
        """Connect, read the TCU's settings and start streaming."""
        address = (self.host, self.command_port)
        self.commands = socket.create_connection(address, timeout=settings.TRIGNO_CONNECT_TIMEOUT_S)
        self.commands.settimeout(settings.TRIGNO_COMMAND_TIMEOUT_S)
        self.banner = self.read_line()
        self.commands.settimeout(0.2)  # the banner may have more lines; read them all
        try:
            while True:
                self.banner += " " + self.read_line()
        except TimeoutError:
            pass
        self.commands.settimeout(settings.TRIGNO_COMMAND_TIMEOUT_S)

        backwards_compatible = self.ask("BACKWARDS COMPATIBILITY?") in ("YES", "ON")
        upsampling = self.ask("UPSAMPLING?") in ("YES", "ON")
        frame_interval = float(self.ask("FRAME INTERVAL?"))
        samples_per_frame = float(self.ask("MAX SAMPLES EMG?"))
        if self.ask("ENDIANNESS?") != "LITTLE":
            raise ValueError("The TCU sends big-endian data; this client reads little-endian")
        if backwards_compatible:
            # SDK 6.1.2: fixed rates on the EMG port. On the bench the TCU reports 17 samples per
            # 13.5 ms frame (1259 Hz) but really streams 2000 Hz with upsampling on.
            self.sample_rate = 2000.0 if upsampling else 15 / 0.0135
        else:
            self.sample_rate = samples_per_frame / frame_interval

        self.columns = []
        for channel in self.channels:
            slot = channel.slot
            if self.ask(f"SENSOR {slot} PAIRED?") != "YES":
                raise ValueError(f"No sensor is paired in TCU slot {slot} ({channel.name}). "
                                 f"Pair it in the Trigno Control Utility.")
            self.ask(f"SENSOR {slot} EMGCHANNELCOUNT?")
            # STARTINDEX counts from 1 (checked on the hardware with TCU 3.6.0)
            self.columns.append(int(self.ask(f"SENSOR {slot} STARTINDEX?")) - 1)
            units = self.ask(f"SENSOR {slot} CHANNEL 1 UNITS?")
            if units not in ("V", "VOLTS"):
                raise ValueError(f"Slot {slot} sends EMG in {units}, expected volts")

        self.data = socket.create_connection((self.host, self.emg_port),
                                             timeout=settings.TRIGNO_CONNECT_TIMEOUT_S)
        self.data.settimeout(settings.TRIGNO_DATA_TIMEOUT_S)
        reply = self.send("START")
        if reply != "OK":
            raise ValueError(f"The TCU replied {reply!r} to START")
        self.streaming = True

    def stop(self) -> None:
        """Stop streaming and close the connections."""
        if self.commands is not None:
            commands = ["STOP", "QUIT"] if self.streaming else ["QUIT"]
            try:
                for command in commands:
                    self.send(command)
            except OSError:  # the TCU may already be gone
                pass
            self.commands.close()
            self.commands = None
        if self.data is not None:
            self.data.close()
            self.data = None
        self.streaming = False

    # --- data -----------------------------------------------------------------------------------

    def read_block(self) -> EMGBlock:
        """
        All complete samples that have arrived (at least one), in mV. Raises
        TimeoutError if no data came for TRIGNO_DATA_TIMEOUT_S.
        """
        assert self.data is not None
        row_bytes = settings.TRIGNO_CHANNELS_ON_PORT * 4
        receive_time = time.monotonic()
        while len(self.data_buffer) < row_bytes:
            chunk = self.data.recv(65536)
            if not chunk:
                raise ConnectionError("The TCU closed the EMG data connection")
            self.data_buffer += chunk
            receive_time = time.monotonic()
        n_rows = len(self.data_buffer) // row_bytes
        values = np.frombuffer(self.data_buffer[:n_rows * row_bytes], dtype="<f4")
        self.data_buffer = self.data_buffer[n_rows * row_bytes:]
        rows = values.reshape(n_rows, settings.TRIGNO_CHANNELS_ON_PORT)
        samples = rows[:, self.columns].T.astype(np.float64) * 1000.0  # volts -> mV
        first = self.samples_read
        self.samples_read += n_rows
        return EMGBlock(samples, first, receive_time)

    def info(self) -> dict:
        """What the TCU said, for session.json."""
        slots = ", ".join(f"{channel.name}=slot {channel.slot}@{column}"
                          for channel, column in zip(self.channels, self.columns))
        return {"host": f"{self.host}:{self.command_port}/{self.emg_port}",
                "server": self.banner, "sample rate": f"{self.sample_rate:g} Hz", "slots": slots}
