"""
Live audio capture.

perception/synth_scenarios.py generates fake ambient + content audio to
stand in for a real microphone + media player in the demo. This module is
the real thing: capture a short buffer from the system's default
microphone using `sounddevice`, for a live deployment.

It's a clean drop-in swap -- everything downstream (context_classifier,
the agent pipeline, the DSP engine) takes a numpy array + sample rate and
doesn't care where it came from, so nothing else needs to change once this
works.

TODO(you): implement `record_ambient` below. See the inline TODOs.
Content audio (what's actually playing) is intentionally out of scope here
-- capturing "currently playing" audio needs OS-level loopback/virtual
audio cable setup that's much more platform-specific; ambient mic capture
alone is a genuinely useful, self-contained slice.
"""
from __future__ import annotations

import numpy as np

try:
    import sounddevice as sd
    _HAS_SOUNDDEVICE = True
except ImportError:
    _HAS_SOUNDDEVICE = False


class MicUnavailableError(RuntimeError):
    """Raised when sounddevice isn't installed or no input device exists."""


def record_ambient(duration_sec: float = 2.0, sr: int = 44100) -> np.ndarray:
    """Record `duration_sec` seconds of mono audio from the default input
    device and return it as a 1-D float32 numpy array in [-1, 1], matching
    the shape/dtype that perception/synth_scenarios.py already produces
    (so it's a true drop-in replacement -- check that file to confirm the
    exact shape your callers expect).

    TODO:
    1. If sounddevice isn't installed (_HAS_SOUNDDEVICE is False), raise
       MicUnavailableError with a helpful message ("pip install
       sounddevice" etc.) instead of crashing with an ImportError deeper
       in the call.
    2. Use sd.rec(...) to record `duration_sec` seconds at `sr`, mono
       (channels=1), blocking until done (sd.wait()).
    3. Return it as a flat (1-D) float32 array. sd.rec gives you a 2-D
       array shaped (frames, channels) -- squeeze/flatten it.
    4. Wrap the actual sd.rec call in a try/except and re-raise as
       MicUnavailableError with the original exception's message if it
       fails (e.g. no input device present, permission denied) -- this
       function should never let a raw sounddevice exception escape,
       since the Streamlit app needs to show a friendly message and fall
       back to synth_scenarios instead of crashing.
    """
    if not _HAS_SOUNDDEVICE:
        raise MicUnavailableError(
            "sounddevice is not installed. Install it with "
            "`pip install sounddevice` to enable live microphone capture."
        )

    num_frames = int(round(duration_sec * sr))

    try:
        recording = sd.rec(num_frames, samplerate=sr, channels=1, dtype="float32")
        sd.wait()
    except Exception as exc:  # pragma: no cover - depends on hardware/env
        raise MicUnavailableError(f"Microphone recording failed: {exc}") from exc

    audio = recording.reshape(-1).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def is_available() -> bool:
    """True if sounddevice is installed AND at least one input device is
    present. Use this to decide whether to show a "record from mic" option
    in the UI at all, vs. only offering synthetic scenarios.

    TODO: implement using sd.query_devices() -- return False on any
    exception (e.g. no audio subsystem at all in a headless environment),
    never raise from here.
    """
    if not _HAS_SOUNDDEVICE:
        return False

    try:
        devices = sd.query_devices()
        return any(d.get("max_input_channels", 0) > 0 for d in devices)
    except Exception:
        return False
