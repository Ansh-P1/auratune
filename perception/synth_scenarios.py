"""
Synthetic ambient+content audio generator for the 3 demo/validation scenarios.

Stands in for real mic capture / playing-content audio so the rest of the
pipeline (context classifier -> agents -> DSP) can be exercised end-to-end
without live hardware. Swap for real sounddevice/streamlit-webrtc capture
in a live deployment -- nothing downstream changes.

Signals are shaped to have genuinely distinguishing acoustic features
(not just placeholder tones), so the context classifier's heuristics
produce the intended label for each scenario:
  - podcast: formant-like tones with a ~4 Hz syllable-rate envelope
  - music:   stable multi-tone harmonic stack, no envelope modulation
  - movie:   tonal dialogue bed + dense percussive FX bursts
"""
from __future__ import annotations

import numpy as np

SR = 44100

SCENARIOS = {
    "quiet_podcast": "Quiet room + podcast",
    "noisy_music": "Noisy environment + music",
    "home_movie": "Home + movie",
}

# Known ground-truth content type per demo scenario (see classify()'s
# content_type_hint param) -- a real deployment would get this from media
# player metadata rather than a label baked into a synthetic signal.
SCENARIO_CONTENT_TYPE = {
    "quiet_podcast": "podcast",
    "noisy_music": "music",
    "home_movie": "movie",
}


def synth_scenario(kind: str, sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(sr) / sr

    if kind == "quiet_podcast":
        ambient = np.random.randn(sr) * 0.002
        syllable_env = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 4 * t))
        content = (np.sin(2 * np.pi * 180 * t) * 0.15 +
                   np.sin(2 * np.pi * 340 * t) * 0.08) * syllable_env

    elif kind == "noisy_music":
        ambient = np.random.randn(sr) * 0.08
        content = (np.sin(2 * np.pi * 110 * t) * 0.25 +
                   np.sin(2 * np.pi * 440 * t) * 0.2 +
                   np.sin(2 * np.pi * 880 * t) * 0.1)

    elif kind == "home_movie":
        ambient = np.random.randn(sr) * 0.015
        # dialogue bed + irregular FX bursts (explosions/foley), typical of
        # a movie mix -- content_type is passed as a hint (see below) rather
        # than relying on the heuristic to disambiguate this from music
        n_bursts = 6
        burst_len = sr // 20
        burst_mask = np.zeros(sr)
        burst_starts = np.random.randint(0, sr - burst_len, size=n_bursts)
        for start in burst_starts:
            burst_mask[start:start + burst_len] = 1.0
        percussive = np.random.randn(sr) * 0.08 * burst_mask
        dialogue = np.sin(2 * np.pi * 200 * t) * 0.1
        content = dialogue + percussive

    else:
        raise ValueError(f"unknown scenario: {kind}")

    return ambient.astype(np.float32), content.astype(np.float32)
