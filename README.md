# finger_exoskeleton

An adaptive hand exoskeleton project for Parkinsonian tremor suppression, using surface EMG to detect grasp intent and tremor-related muscle activity. The system focuses on a 2-DOF thumb–index design and aims to apply adaptive stiffness or damping only when tremor is present, reducing involuntary motion while preserving natural voluntary movement.

## Goals

- Detect grasp intent and tremor-related muscle activity from surface EMG.
- Actuate a 2-DOF thumb–index mechanism.
- Apply adaptive stiffness or damping only while tremor is present.
- Preserve natural voluntary movement.

## Requirements

- Python 3.12 or newer

## How to run

Install the dependencies once (the test scripts also do this). On Windows:

```bat
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

On Linux / macOS, use `python3 -m venv .venv` and `.venv/bin/python` instead.

The project currently does **EMG acquisition and live viewing only**; there is no
controller or actuator code. All settings (channels and muscles, Trigno, filters, view
window) are in [applied_config.py](applied_config.py); their types, defaults and validation
(the blueprint) are in [source/config.py](source/config.py). `CHANNELS`
selects `ONE_SENSOR` (FDS in TCU slot 15, the default) or `FOUR_SENSORS` (FDS, EDC, FDI,
APB in slots 1-4).

| Mode | Command |
|---|---|
| Live Delsys Trigno | `.venv\Scripts\python.exe run_emg.py --source trigno` |
| Trigno backend against a mock TCU (no hardware) | `.venv\Scripts\python.exe run_emg.py --source trigno --mock-tcu` |
| Synthetic EMG | `.venv\Scripts\python.exe run_emg.py --source synthetic` |
| Recorded EMG (dataset) | `.venv\Scripts\python.exe run_emg.py --source recorded --file 10_raw.csv` |
| **Replay one of your sessions** (no sensors needed) | `.venv\Scripts\python.exe run_emg.py --source recorded --file D:\EMG\measurements\20261023-140512_trigno_grip` |
| Play a recording once instead of looping | add `--no-loop` |
| **Save a measurement** (new session folder per run) | add `--out <folder>` (optional `--label grip --note "subject 1"`) |
| Record the raw EMG to one CSV file | add `--save logs/session.csv` |
| Headless (status lines, no window) | add `--headless --duration 30` |
| Offline analysis plots (original scripts) | `.venv\Scripts\python.exe main_sensor.py` |

The live window shows one plot per channel: the filtered EMG (20-450 Hz band-pass + 50 Hz
notch) with its 200 ms RMS envelope, and a status panel with the connection, the
reported and measured sample rate, the current RMS per channel and counters (blocks,
missing samples, reconnections). Keys: `r` raw/filtered, `space` pause, `q` quit.

`--save` writes a CSV with a header row, then `time_s` and one column per channel in mV
(the raw samples, exactly as received).

### Saving measurements

Give a measurement folder with `--out`; every run gets its own sub-folder there, so
nothing is ever overwritten:

```bat
.venv\Scripts\python.exe run_emg.py --source trigno --out D:\EMG\measurements --label grip --note "subject 1, right hand"
```

```
D:\EMG\measurements\
  20261023-140512_trigno_grip\
    emg.csv        time_s and one column per channel in mV (raw, as received)
    session.json   start/end time, status, source and TCU details, sample rate,
                   channels and slots, label, note, counters (gaps, reconnections),
                   events (reconnects, timeouts, ... with their time in the
                   recording), errors and the full configuration in use
```

The folder name is `<date>-<time>_<source>[_<label>]`, so a synthetic or mock run cannot
be mistaken for a real one. The CSV is flushed to disk every second, and `session.json` is
written when streaming starts (`"status": "recording"`) and again when the run ends
(`"complete"` or `"error"`), so a crash keeps everything recorded up to that point. If the
source cannot start, no folder is left behind. Stop a recording by closing the window,
pressing `q`, or Ctrl+C.

`time_s` is the sample index divided by the sample rate, so lost samples show up as a jump
in `time_s` (and in the `gaps` / `missing samples` counters). After a reconnection the
count continues, so the pause itself is not visible in `time_s`; see `events`.

### Using saved measurements without sensors

Replay a session through the live pipeline (filters, envelopes, viewer) exactly as it was
recorded: `--source recorded --file <session folder>`. The channels and the sample rate
come from `session.json`, whatever `CHANNELS` is set to. A plain `--save` CSV works too
(the sample rate then follows from `time_s`). Replaying with `--out` saves an identical copy.

For analysis in Python:

```python
from pathlib import Path
from source.session import find_recording

recording = find_recording(Path(r"D:\EMG\measurements\20261023-140512_trigno_grip"))
times, emg = recording.load()          # (n,) in s, (n_channels, n) in mV
names = [channel.name for channel in recording.channels]
rate = recording.sample_rate_hz
note = recording.metadata["note"]      # everything in session.json
```

### First connection to the Delsys Trigno (DS-T03 / SP-W02 / SP-W06)

1. Install the Delsys Trigno Control Utility (TCU, from the free "Trigno SDK Server"
   installer) on a Windows PC. Connect the base station by USB and start the TCU.
2. In the TCU, pair the sensors to slots 1-4, the slots in `CHANNELS` in `applied_config.py`.
   Put each sensor on the muscle named there: FDS, EDC, FDI, APB. Close EMGworks and any
   other software that might be using the base station.
3. Run `.venv\Scripts\python.exe run_emg.py --source trigno`. On start-up the
   program queries the TCU and refuses to run if a slot is unpaired or the units are not
   volts. It reports the sample rate it derived. If the TCU runs on another PC, add
   `--host <ip>`.
4. Check the live view:
   - Raw EMG should be near zero at rest and rise with contraction.
   - The sample rate should be 2000 Hz with Backwards Compatibility and Upsample on, or
     1111.111 Hz on port 50043 with Upsample off.
   - Look for warnings in the status panel.

**Verified on hardware** (TCU / Trigno SDK Server 3.6.0, one sensor):
- **STARTINDEX base is 1:** the sensor in slot 4 reports `STARTINDEX 4` and its data is
  in 0-based column 3 of the EMG port, so `start_index_base=1` in `TRIGNO`.
- **Units reply is `V`** (the SDK guide's example shows `Volts`).
- **Sample rate:** 2000 Hz on port 50043 with Backwards Compatibility and Upsample on,
  although `MAX SAMPLES EMG?` reports 17 (the sensor's native ~1259 Hz).
- **STOP takes ~2.6 s** to reply `OK`, hence `command_timeout_s=5.0`.

If the TCU reports "drivers are missing" and the base station shows Code 28 in Device
Manager, install the driver from an administrator terminal:
`pnputil /add-driver "C:\Program Files (x86)\Delsys, Inc\Trigno SDK\Drivers\Trigno\SiUSBXp.inf" /install`,
then reconnect the base station.

**Not yet verified on hardware:**
- **Reply format:** the exact replies (`UPSAMPLING?`, `FRAME INTERVAL?`, ...) and the banner.
- **Pause before data:** the delay between START and the first data.
- **Latency:** the real end-to-end delay.

## Testing

The test scripts create a `.venv`, install the dev dependencies from `requirements-dev.txt`, and then run flake8, pylint, mypy and pytest.

Windows:

```bat
test.bat
```

Linux / macOS:

```bash
./test.sh
```

Set the `PYTHON` environment variable to pick the interpreter used to create the virtualenv (default: `py -3` on Windows, `python3` on Linux).

## Project structure

```
run_emg.py                   Entry point: acquire EMG and show it live
applied_config.py            The settings in use (channels, Trigno, filters, view window)
main_sensor.py               Offline analysis plots of the recorded dataset
source/
  config.py                  Configuration blueprint: setting types, defaults, validation
  emg_source.py              EMGSource interface, EMGBlock, source errors (all sources)
  sensor_mock.py             SyntheticSource, RecordedSource (no hardware)
  trigno.py                  TrignoSource: Delsys Trigno Control Utility TCP/IP client
  mock_tcu.py                Mock TCU server for tests and demos (not hardware)
  streaming.py               Causal, stateful multi-channel filtering and envelopes
  acquisition.py             Acquisition thread, ring buffers, CSV recording
  session.py                 Measurement session folders (emg.csv + session.json)
  calibration.py             Amplitude calibration (placeholder, not used yet)
  viewer.py                  Live matplotlib viewer and console status
  import_data.py             Dataset loading (offline)
  process_signal.py          Offline (zero-phase) analysis and filter replication
  visualize_data.py          Offline plots
tests/                       pytest suite (sources, mock TCU, filters, acquisition, CLI)
```

Data flow: EMG source → acquisition thread (filtering, gap detection, optional CSV
recording) → ring buffers holding the last 5 s → live viewer.

## How the project works
_To be written._

# Interpreting EMG signals
_To be written._

# Control loop
_To be written._

# Actuation
_To be written._
