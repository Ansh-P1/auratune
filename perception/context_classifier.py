"""
Context classifier.

Takes two audio buffers -- the ambient mic feed and the currently playing
content -- and produces a Context object describing:
  - noise_level: "quiet" | "moderate" | "noisy"
  - content_type: "podcast" | "music" | "movie"

This is intentionally rule-based (RMS + spectral features via librosa)
rather than a trained classifier, matching the "Phase 1" scope in the
project's own roadmap slide (mock context first, model swap later).
It's a clean seam: swap `classify()` for a trained model without touching
anything downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False


@dataclass
class Context:
    noise_level: str      # quiet | moderate | noisy
    content_type: str     # podcast | music | movie
    ambient_rms_db: float
    content_features: dict


def _rms_db(audio: np.ndarray) -> float:
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def _classify_noise(ambient: np.ndarray) -> str:
    db = _rms_db(ambient)
    if db < -45:
        return "quiet"
    if db < -25:
        return "moderate"
    return "noisy"


def _envelope_modulation_score(y: np.ndarray, sr: int) -> float:
    """Strength of amplitude-envelope modulation in the 2-8 Hz syllabic-rate
    band, relative to total envelope energy. Speech has a strong, narrow
    peak here (~4 Hz syllable rate); sustained music tones and dense movie
    mixes generally don't. This is the main speech/non-speech discriminator.
    """
    envelope = np.abs(librosa.util.normalize(y))
    # smooth to ~50 Hz frame rate before taking the modulation spectrum
    hop = max(1, sr // 100)
    frames = envelope[: (len(envelope) // hop) * hop].reshape(-1, hop).mean(axis=1)
    if len(frames) < 8:
        return 0.0
    frames = frames - frames.mean()
    frame_rate = sr / hop
    spec = np.abs(np.fft.rfft(frames))
    freqs = np.fft.rfftfreq(len(frames), d=1 / frame_rate)
    total = float(np.sum(spec) + 1e-9)
    band = (freqs >= 2.0) & (freqs <= 8.0)
    return float(np.sum(spec[band]) / total)


def _classify_content(content: np.ndarray, sr: int) -> tuple[str, dict]:
    """Heuristic speech vs. music vs. mixed(movie) classification.

    Uses: syllable-rate envelope modulation (speech discriminator),
    spectral flatness (tonal vs. noisy/percussive content), and
    harmonic-percussive energy ratio (movies mix dense percussive/FX
    energy with dialogue bands).
    """
    if not _HAS_LIBROSA or len(content) < sr // 2:
        # not enough signal or no librosa -> default guess
        return "podcast", {"reason": "insufficient_signal_or_no_librosa"}

    y = content.astype(np.float32)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    envelope_mod = _envelope_modulation_score(y, sr)
    harmonic, percussive = librosa.effects.hpss(y)
    h_energy = float(np.sum(harmonic ** 2))
    p_energy = float(np.sum(percussive ** 2))
    hp_ratio = h_energy / (p_energy + 1e-9)

    feats = {
        "zcr": round(zcr, 4),
        "spectral_flatness": round(flatness, 4),
        "spectral_centroid_hz": round(centroid, 1),
        "harmonic_percussive_ratio": round(hp_ratio, 3),
        "envelope_modulation_2_8hz": round(envelope_mod, 3),
    }

    # Strong syllable-rate envelope modulation + non-trivial spectral flatness
    # (speech's envelope switching spreads energy) -> speech/podcast, checked
    # first. Flatness guard keeps brief incidental FX bursts (which can also
    # nudge the modulation spectrum) from being mistaken for speech.
    if envelope_mod > 0.35 and flatness > 0.15:
        return "podcast", feats
    # Dense percussive energy mixed with tonal content -> movie (FX + dialogue + score)
    if p_energy > 0 and hp_ratio < 1.2 and flatness > 0.08:
        return "movie", feats
    # High harmonic ratio + low flatness, stable tonal content -> music
    if hp_ratio >= 1.2 and flatness < 0.12:
        return "music", feats
    # Otherwise default to podcast (narrowband/ambiguous)
    return "podcast", feats


def classify(ambient: np.ndarray, content: np.ndarray, sr: int = 44100,
             content_type_hint: str | None = None) -> Context:
    """Classify ambient noise level (always signal-derived) and content type.

    content_type_hint lets a caller that already knows the content type
    (e.g. it came from the media player / app metadata, or a labeled demo
    scenario) skip the heuristic and use ground truth directly -- exactly
    how a real deployment would prefer an authoritative source over a
    guess when one's available. Falls back to the heuristic classifier
    when no hint is given.
    """
    noise_level = _classify_noise(ambient)
    if content_type_hint in ("podcast", "music", "movie"):
        content_type, feats = content_type_hint, {"source": "hint"}
    else:
        content_type, feats = _classify_content(content, sr)
    return Context(
        noise_level=noise_level,
        content_type=content_type,
        ambient_rms_db=round(_rms_db(ambient), 1),
        content_features=feats,
    )
