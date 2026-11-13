"""
Made-up data. NOT evidence that anything works on real muscles or real tremor.

  add_tremor(recording)        lays synthetic tremor over a recording, so the tremor
                               label is known exactly. Used to get tremor examples
                               on top of our own (tremor-free) recordings.
  synthetic_recording(seed)    EMG-like noise with made-up intention labels, for the
                               software smoke test of the training pipeline.

The synthetic tremor imitates Parkinsonian tremor in the simplest way: the
EMG is switched on and off in bursts 4-7 times per second, and the flexor and
the extensor burst in turn (in antiphase). Real tremor is less regular.
"""

from dataclasses import replace

import numpy as np

from source.gru.labels import LabelledRecording, rest_level_of

RAMP_S = 0.3  # the tremor fades in and out over this long


def add_tremor(recording: LabelledRecording, seed: int) -> LabelledRecording:
    """A copy with synthetic tremor in random spans; its tremor label is 1 inside them."""
    rng = np.random.default_rng(seed)
    n = recording.emg.shape[1]
    times = np.arange(n) / recording.sample_rate

    # Random spans: 2-6 s without tremor, then 2-5 s with tremor, and so on
    weight = np.zeros(n)  # 0 = no tremor, 1 = full tremor, in between = fading
    start = rng.uniform(2.0, 6.0)
    while start < times[-1]:
        end = start + rng.uniform(2.0, 5.0)
        fade = np.minimum((times - start) / RAMP_S, (end - times) / RAMP_S)
        weight = np.maximum(weight, np.clip(fade, 0.0, 1.0))
        start = end + rng.uniform(2.0, 6.0)

    frequency = rng.uniform(4.0, 7.0)
    depth = rng.uniform(0.5, 1.0) * weight
    # Extensors burst half a period after the flexors
    phase = np.array([np.pi if role == "extensor" else 0.0 for role in recording.roles])
    burst = 0.5 * (1 + np.sin(2 * np.pi * frequency * times + phase[:, None]))  # 0 ... 1
    # Where the muscle is active the EMG is switched on and off (the mean power stays the same);
    # at rest small bursts of noise are added, as in rest tremor
    emg = recording.emg * ((1 - depth) + depth * 2 * burst)
    emg = emg + rng.standard_normal(emg.shape) * 2 * recording.rest_level[:, None] * depth * burst

    tremor = recording.tremor.copy()
    tremor[weight > 0.5] = 1.0
    tremor[(weight > 0) & (weight <= 0.5)] = np.nan  # while fading the label is not clear
    return replace(recording, name=f"{recording.name}+tremor{seed}", emg=emg, tremor=tremor,
                   synthetic=True)


def synthetic_recording(seed: int, roles: list[str], seconds: float = 60.0) -> LabelledRecording:
    """
    SOFTWARE TEST ONLY. EMG-like noise at 2000 Hz: first the calibration (rest, then
    a full grip), then random stretches of rest, half and full closing (and
    opening, with an extensor channel). The closing level sets the flexor
    amplitude, the opening level the extensor amplitude. Tremor label 0 everywhere.
    """
    rng = np.random.default_rng(seed)
    rate = 2000.0
    levels = [0.0, 0.0, 0.5, 1.0] + ([-1.0] if "extensor" in roles else [])
    parts = [(3.5, 0.0), (6.5, 1.0), (2.0, 0.0)]  # the calibration routine
    while sum(duration for duration, _ in parts) < seconds:
        parts.append((float(rng.uniform(2.0, 4.0)), float(rng.choice(levels))))
    intention = np.concatenate([np.full(round(duration * rate), level)
                                for duration, level in parts])

    rest_mv, grip_mv = 0.004, 0.08
    amplitude = []
    for role in roles:
        effort = np.maximum(-intention, 0) if role == "extensor" else np.maximum(intention, 0)
        amplitude.append(rest_mv + grip_mv * effort)
    emg = rng.standard_normal((len(roles), len(intention))) * np.array(amplitude)
    return LabelledRecording(
        name=f"synthetic-{seed}", group=f"synthetic-{seed}", emg=emg, sample_rate=rate,
        roles=list(roles), rest_level=rest_level_of(emg, rate), intention=intention,
        tremor=np.zeros(len(intention)), synthetic=True)
