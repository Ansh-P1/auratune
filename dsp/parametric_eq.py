"""
Parametric EQ engine.

Implements a chain of biquad filters (low shelf, peaking bands, high shelf)
driven by a small set of high-level parameters that the agent pipeline
outputs (bass_gain_db, presence_gain_db, treble_gain_db, volume_db, etc).

This is pure DSP -- no ML, no network calls -- so it runs anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple
import numpy as np
from scipy import signal


@dataclass
class EQBand:
    kind: str          # "low_shelf" | "high_shelf" | "peaking"
    freq_hz: float
    gain_db: float
    q: float = 0.707

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TargetCurve:
    """A named, human-readable EQ target (what the agents reason about)."""
    name: str
    volume_db: float = 0.0
    bass_gain_db: float = 0.0        # low shelf ~100 Hz
    presence_gain_db: float = 0.0    # peaking ~2.5-3.5 kHz (vocal clarity)
    treble_gain_db: float = 0.0      # high shelf ~8 kHz
    bands: List[EQBand] = field(default_factory=list)

    def to_bands(self) -> List[EQBand]:
        bands = [
            EQBand("low_shelf", 100.0, self.bass_gain_db, 0.707),
            EQBand("peaking", 3000.0, self.presence_gain_db, 1.0),
            EQBand("high_shelf", 8000.0, self.treble_gain_db, 0.707),
        ]
        bands.extend(self.bands)
        return bands

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


def _biquad_coeffs(band: EQBand, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (b, a) biquad coefficients for one band (Audio-EQ-cookbook)."""
    A = 10 ** (band.gain_db / 40.0)
    w0 = 2 * np.pi * band.freq_hz / sr
    alpha = np.sin(w0) / (2 * band.q)
    cos_w0 = np.cos(w0)

    if band.kind == "peaking":
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A
    elif band.kind == "low_shelf":
        sqrtA = np.sqrt(A)
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrtA * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrtA * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrtA * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrtA * alpha
    elif band.kind == "high_shelf":
        sqrtA = np.sqrt(A)
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrtA * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrtA * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrtA * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrtA * alpha
    else:
        raise ValueError(f"unknown band kind: {band.kind}")

    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    return b, a


class ParametricEQ:
    """Applies a TargetCurve to audio, and can report its frequency response."""

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    def apply(self, audio: np.ndarray, curve: TargetCurve) -> np.ndarray:
        out = audio.astype(np.float64).copy()
        for band in curve.to_bands():
            if abs(band.gain_db) < 1e-6:
                continue
            b, a = _biquad_coeffs(band, self.sr)
            out = signal.lfilter(b, a, out)
        # apply overall volume trim/boost
        out *= 10 ** (curve.volume_db / 20.0)
        # safety clip to avoid blowing out speakers on large boosts
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    def frequency_response(self, curve: TargetCurve, n_points: int = 512):
        """Return (freqs, magnitude_db) for plotting the live curve."""
        freqs = np.logspace(np.log10(20), np.log10(self.sr / 2), n_points)
        w = 2 * np.pi * freqs / self.sr
        total_h = np.ones_like(w, dtype=np.complex128)
        for band in curve.to_bands():
            if abs(band.gain_db) < 1e-6:
                continue
            b, a = _biquad_coeffs(band, self.sr)
            _, h = signal.freqz(b, a, worN=w)
            total_h *= h
        mag_db = 20 * np.log10(np.maximum(np.abs(total_h), 1e-9)) + curve.volume_db
        return freqs, mag_db

    def delta(self, before: TargetCurve, after: TargetCurve) -> Dict[str, float]:
        """Human-readable deltas the Explainer agent turns into a sentence."""
        return {
            "volume_db": round(after.volume_db - before.volume_db, 2),
            "bass_gain_db": round(after.bass_gain_db - before.bass_gain_db, 2),
            "presence_gain_db": round(after.presence_gain_db - before.presence_gain_db, 2),
            "treble_gain_db": round(after.treble_gain_db - before.treble_gain_db, 2),
        }
