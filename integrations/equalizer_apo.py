"""
Equalizer APO integration -- apply an AuraTune curve to real audio, live.

Equalizer APO (https://equalizerapo.com, free/open source, Windows) is a
system-wide parametric EQ that hooks an output device and reads a plain-text
config file. It watches that file itself and reloads within a few hundred
milliseconds of it changing, so "applying" an EQ here is literally writing
the file -- no API, no service to poke.

    Preamp: -3.0 dB
    Filter 1: ON LSC Fc 100 Hz Gain 2.5 dB Q 0.707
    Filter 2: ON PK Fc 3000 Hz Gain 3.5 dB Q 1.0

Because the file is re-read as it changes, writes MUST be atomic -- a
half-written config is a config Equalizer APO will happily load. Every write
here goes to a temp file in the same directory and is then os.replace()d
over the target, which is atomic on both Windows and POSIX.

Graceful fallback, matching perception/live_capture.py: if Equalizer APO
isn't installed, is_available() is False and write_config() raises
EqualizerAPOUnavailable with an install pointer, rather than crashing the
caller.

Non-Windows machines can still exercise everything except the audible part:
point AURATUNE_APO_CONFIG at any writable path and the renderer/writer work
normally (that's how the tests run on macOS/Linux CI).
"""
from __future__ import annotations

import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from dsp.parametric_eq import EQBand, TargetCurve

# Equalizer APO's default install writes its config here. The installer lets
# you choose another directory, so AURATUNE_APO_CONFIG overrides it (and is
# what the tests and non-Windows dev machines use).
DEFAULT_CONFIG_PATH = Path(r"C:\Program Files\EqualizerAPO\config\config.txt")
CONFIG_PATH_ENV = "AURATUNE_APO_CONFIG"

# Our three named bands map onto Equalizer APO filter types. LSC/HSC are the
# shelving filters that take a Q (the plain LS/HS variants take a slope
# instead), which is what dsp.parametric_eq's biquads model.
_FILTER_TYPES = {
    "low_shelf": "LSC",
    "high_shelf": "HSC",
    "peaking": "PK",
}

_HEADER = (
    "# Written by AuraTune -- do not edit by hand while Auto mode is on.\n"
    "# Equalizer APO reloads this file automatically when it changes.\n"
)


class EqualizerAPOUnavailable(RuntimeError):
    """Raised when Equalizer APO's config file can't be found or written."""


@dataclass
class LiveApplyResult:
    """What a live write actually did -- surfaced in the UI/agent trace so an
    automatic EQ change is never invisible."""
    path: Path
    filters: int
    preamp_db: float
    bytes_written: int

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "filters": self.filters,
            "preamp_db": self.preamp_db,
            "bytes_written": self.bytes_written,
        }


def config_path(path: Optional[os.PathLike | str] = None) -> Path:
    """Where this integration reads/writes Equalizer APO's config."""
    if path is not None:
        return Path(path)
    from_env = os.environ.get(CONFIG_PATH_ENV)
    return Path(from_env) if from_env else DEFAULT_CONFIG_PATH


def is_available(path: Optional[os.PathLike | str] = None) -> bool:
    """True when we can plausibly write a config Equalizer APO will read.

    Deliberately checks the *directory*, not the file: a fresh install may
    not have a config.txt until something writes one.
    """
    return config_path(path).parent.is_dir()


def unavailable_reason(path: Optional[os.PathLike | str] = None) -> str:
    """One line explaining why live apply is off, for the UI to show."""
    target = config_path(path)
    if target.parent.is_dir():
        return ""
    if platform.system() != "Windows":
        return (f"Equalizer APO is Windows-only, and {target.parent} doesn't "
                f"exist on this {platform.system()} machine. Set "
                f"{CONFIG_PATH_ENV} to a writable path to test the file "
                f"writer without it.")
    return (f"Equalizer APO doesn't look installed -- {target.parent} is "
            f"missing. Install it from https://equalizerapo.com and attach "
            f"it to your output device, or set {CONFIG_PATH_ENV} if you "
            f"installed it elsewhere.")


def _fmt(value: float) -> str:
    """Trim trailing zeros -- APO accepts either, but configs get read by
    humans while debugging a demo."""
    return f"{value:.2f}".rstrip("0").rstrip(".") or "0"


def render_config(bands: Iterable[EQBand], preamp_db: float = 0.0) -> str:
    """Render bands as an Equalizer APO config file body.

    Bands whose gain rounds to 0.00 dB are written as OFF rather than
    dropped, so the filter numbering stays stable between writes -- easier to
    eyeball in a diff during a demo.
    """
    lines: List[str] = [_HEADER.rstrip("\n"), f"Preamp: {_fmt(preamp_db)} dB"]
    for i, band in enumerate(bands, start=1):
        kind = _FILTER_TYPES.get(band.kind)
        if kind is None:
            raise ValueError(f"no Equalizer APO filter type for band kind {band.kind!r}")
        state = "OFF" if abs(round(band.gain_db, 2)) < 0.005 else "ON"
        lines.append(
            f"Filter {i}: {state} {kind} Fc {_fmt(band.freq_hz)} Hz "
            f"Gain {_fmt(band.gain_db)} dB Q {_fmt(band.q)}"
        )
    return "\n".join(lines) + "\n"


def write_config(bands: Iterable[EQBand], preamp_db: float = 0.0,
                 path: Optional[os.PathLike | str] = None) -> LiveApplyResult:
    """Atomically write an Equalizer APO config. Returns what was written.

    Atomic because Equalizer APO watches the file: a plain open("w") would
    briefly expose an empty or partial config, which it would load.
    """
    target = config_path(path)
    if not target.parent.is_dir():
        raise EqualizerAPOUnavailable(unavailable_reason(path))

    bands = list(bands)
    text = render_config(bands, preamp_db)
    try:
        # Same directory as the target: os.replace() is only atomic within a
        # single filesystem.
        fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".auratune-",
                                        suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, target)
        except Exception:
            Path(tmp_name).unlink(missing_ok=True)
            raise
    except OSError as exc:
        raise EqualizerAPOUnavailable(
            f"Couldn't write {target}: {exc}. Equalizer APO's config "
            f"directory usually needs the app to run as administrator, or "
            f"set {CONFIG_PATH_ENV} to a writable path."
        ) from exc

    return LiveApplyResult(path=target, filters=len(bands),
                           preamp_db=round(preamp_db, 2),
                           bytes_written=len(text.encode("utf-8")))


def apply_curve(curve: TargetCurve,
                path: Optional[os.PathLike | str] = None) -> LiveApplyResult:
    """Apply a TargetCurve straight to live audio.

    The curve's overall level becomes APO's preamp; its three named bands
    become the filters. This is the entry point Part 2's live loop calls once
    per smoothing step, so a ramp lands as a series of small file writes.
    """
    return write_config(curve.to_bands(), preamp_db=curve.volume_db, path=path)


# Q for a projected graphic-EQ band. The projection step hands us gains on a
# fixed grid of centre frequencies with no Q of their own; ~1.41 is the usual
# choice for octave-spaced bands -- wide enough that neighbouring filters sum
# into a smooth curve rather than a row of narrow spikes.
PROJECTED_BAND_Q = 1.41


def write_projection(projected, path: Optional[os.PathLike | str] = None) -> LiveApplyResult:
    """Apply a dsp.eq_projection.ProjectedEQ to live audio.

    Takes the *projected* bands rather than the raw curve on purpose: those
    are the exact values the dashboard shows, so what a user sees on screen
    and what they hear are the same thing.
    """
    bands = [EQBand("peaking", b.freq_hz, b.set_gain_db, PROJECTED_BAND_Q)
             for b in projected.bands]
    return write_config(bands, preamp_db=projected.preamp_db, path=path)
