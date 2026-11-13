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

All settings (sensors and slots, Trigno, filters, calibration, controller thresholds, motor
positions) are plain constants in [settings.py](settings.py). The commands used on the
bench are in [kommandoer.txt](kommandoer.txt).

| What | Command |
|---|---|
| Live Delsys Trigno | `.venv\Scripts\python.exe run_emg.py --source trigno` |
| A fake TCU playing a recording (no hardware) | `.venv\Scripts\python.exe run_emg.py --source trigno --mock-tcu` |
| Replay a recorded session | `.venv\Scripts\python.exe run_emg.py --source recorded --file measurements\<session folder>` |
| Play a recording once instead of looping | add `--no-loop` |
| Record the run in a new session folder | add `--out measurements` (optional `--label grip`) |
| Status lines instead of a window | add `--headless --duration 30` |
| Drive the tendon motor | add `--motor` (optional `--motor-port COM5`) |
| Use or write another calibration file | add `--calibration <file>` |
| Calibrate again | add `--calibrate`, or press `c` in the window |
| Change the 20 s wait before calibration and cues | add `--start-delay 5` |
| Show cues and save them as labels | add `--out measurements --cues protocols\greb_1.csv` |
| Use a trained GRU model instead of the signal processing | add `--method gru --model models\gru.pt` |
| Look at a recording with its labels as colours | `.venv\Scripts\python.exe plot_recording.py measurements\<session folder>` |

The window shows the filtered EMG of every channel with its slow envelope (effort) and fast
envelope (tremor bursts), a controller panel (grasp activation, tremor depth and stiffness,
with the on-thresholds dashed) and a text panel with the latest values, the controller
state, the calibration and messages. Keys: `c` calibrate again, `q` quit.

The signal processing (DSP) is the default method. A small GRU neural network can be trained
as an alternative and compared with it on the same controller: see
[GRU_README.md](GRU_README.md) (in Danish).

### Calibration

The controller needs to know the rest and grip levels of the muscles. A live run first waits
`START_DELAY_S` (20 s, in `settings.py`) so the window has time to open; the status shows the
countdown. Then: rest from 0.5 to 3 s, and hold a firm grip somewhere between 3.5 and 10 s. When the calibration succeeds the
controller arms as soon as the hand has relaxed (the window shows "relax to arm"); if it
fails, it says why and stays `DISARMED`.

With `--source trigno` a successful calibration is saved to `calibration.json` and used by
later runs, so the routine is only done when you ask for it (`--calibrate` or the `c` key).
A saved file is ignored, with a message, if it is for other channels, was made with other
filter settings, or its grip is not at least 3x the rest level. It belongs to one sensor
placement, so calibrate again after moving the sensors. Replays and the fake TCU calibrate
every run unless you give `--calibration <file>`.

### The controller

`source/stiffness_controller.py` decides from the grasp activation (0 = rest, 1 = the
calibration grip) and the tremor depth and ratio:

```
DISARMED -> (calibrated, hand relaxed) -> IDLE -> (grip) -> GRASP_ARMED -> (rhythmic tremor) -> STIFFENED
```

and back when the grip is released or the tremor stops. In `STIFFENED` it computes a
stiffness 0-1 that grows with the tremor depth. The stiffness is only shown; the motor
uses the state (see below).

### Recording and replaying

With `--out <folder>` every run gets its own sub-folder, so nothing is overwritten:

```
measurements\
  20261105-144738_trigno_grip\
    emg.csv        time_s and one column per channel in mV (the raw samples)
    session.json   start/end time, source and TCU details, sample rate, channels,
                   filter settings, messages with their time, and any error
```

`emg.csv` is flushed every second and `session.json` is written at the start and the end,
so a crash keeps what was recorded. Replay a folder with
`--source recorded --file <folder>`: the channels and sample rate come from `session.json`.

For analysis in Python:

```python
from pathlib import Path
from source.recording import Recording

recording = Recording(Path("measurements/20261105-144738_trigno_to-lange-greb"))
recording.times          # (n,) in s
recording.emg            # (n_channels, n) in mV
recording.channels       # names, roles and slots
recording.metadata       # everything in session.json
```

### First connection to the Delsys Trigno

1. Install the Delsys Trigno Control Utility (TCU, from the free "Trigno SDK Server"
   installer) on a Windows PC. Connect the base station by USB and start the TCU.
2. In the TCU, pair the sensors to the slots in `CHANNELS` in `settings.py` (FDS in slot 15,
   EDC in slot 14). Close EMGworks and any other software using the base station.
3. Run `.venv\Scripts\python.exe run_emg.py --source trigno`. It stops with a message if a
   slot has no paired sensor. If the TCU runs on another PC, add `--host <ip>` and open the
   ports 50040-50044 in that PC's firewall.

Checked on the hardware (TCU 3.6.0): `STARTINDEX` counts from 1; the units reply is `V`;
the EMG port streams 2000 Hz with Backwards Compatibility and Upsample on, although
`MAX SAMPLES EMG?` reports 17 (about 1259 Hz); `STOP` takes about 2.6 s to answer.

If the TCU reports "drivers are missing" and the base station shows Code 28 in Device
Manager, install the driver from an administrator terminal:
`pnputil /add-driver "C:\Program Files (x86)\Delsys, Inc\Trigno SDK\Drivers\Trigno\SiUSBXp.inf" /install`,
then reconnect the base station.

## Driving the motor

With `--motor`, Emil's `ProportionalEMGController` (`source/controllers.py`) moves the
finger between `OPEN_POS` and `CLOSED_POS` (in `settings.py`) in Dynamixel extended position
mode. `source/motor.py` runs it in its own thread and decides what it gets:

- `DISARMED` (not calibrated yet): 0, the finger stays open.
- `IDLE` (no grasp): 0, so rest activity such as rest tremor does not move the finger.
- `GRASP_ARMED`: the grasp activation, so the finger follows the grip.
- `STIFFENED` (tremor during a grasp): the mean grasp of the second before the tremor, so
  the finger holds its position instead of following the tremor.
- No fresh EMG for 0.3 s: no signal; after 0.5 s more the finger opens.

On exit (window closed, `q`, Ctrl+C, end of `--duration`) the finger opens before anything
else stops. A motor error stops only the motor; the EMG goes on. Only motor `MOTOR_ID` is
touched. The start is refused, with torque off, if the motor is far outside the
open-closed range.

## Testing

The test scripts create a `.venv`, install the dev dependencies from
`requirements-dev.txt`, and run flake8, pylint, mypy and pytest. Emil's motor code
(`source/controllers.py`, `source/DynamixelHardwareInterface.py`) is used as it is and is
not linted.

```bat
test.bat
```

```bash
./test.sh
```

Set the `PYTHON` environment variable to pick the interpreter used to create the
virtualenv (default: `py -3` on Windows, `python3` on Linux).

## Project structure

```
run_emg.py                   Command line: source, pipeline, window or status lines, motor
train_gru.py                 Train a GRU model (needs PyTorch, requirements-ml.txt)
evaluate_gru.py              Test a GRU model and the DSP on recordings with labels
plot_recording.py            Draw a recording with its labels as colours (check before training)
settings.py                  All settings as plain constants
main.py, main_controller.py  Emil's motor tools (one command; bench test of the controllers)
source/
  trigno.py                  TrignoSource: talks to the Trigno Control Utility over TCP
  mock_tcu.py                MockTCU: a fake TCU that plays a recording (tests, demos)
  recording.py               Session folders: SessionWriter, Recording, RecordedSource
  signal_processing.py       Filters and envelopes, block by block
  calibration.py             Calibration (rest + grip), the calibration file, activation
  stiffness_controller.py    State machine DISARMED/IDLE/GRASP_ARMED/STIFFENED -> stiffness
  pipeline.py                Pipeline (what happens to every block) and the Acquisition thread
  motor.py                   GraspSignal, MotorThread, connect_motor
  viewer.py                  The live window and the status line
  cues.py                    Cue protocol: what to do when, saved as labels.csv
  gru/                       The GRU method: features, model, training, evaluation,
                             real-time interface (see GRU_README.md)
  controllers.py             Emil's EMG position controllers
  DynamixelHardwareInterface.py  Emil's Dynamixel interface
protocols/                   Cue protocols for recordings with labels (greb_1-3.csv)
measurements/                The recordings (kept in git: they are the training data)
tests/                       pytest suite
```

Data flow:

```
TrignoSource / RecordedSource -> Acquisition thread -> Pipeline.step(block)
    -> SignalProcessor -> Calibrator / compute_activation -> StiffnessController
    -> (viewer reads Pipeline.recent())    (motor thread reads Pipeline.state and .activation)
```

## How the project works
_To be written._

# Interpreting EMG signals
_To be written._

# Control loop
_To be written._

# Actuation
_To be written._
