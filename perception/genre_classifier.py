"""
Genre/mood classification from real audio, using the models trained in
ml/train.py on the Spotify Tracks Dataset (114k real tracks -- see
ml/README.md).

The trained models expect 13 numeric inputs -- Spotify's original 12
audio features (danceability, energy, loudness, mode, speechiness,
acousticness, instrumentalness, liveness, valence, tempo, time_signature),
minus "key" (a categorical 0-11 pitch class, cyclically encoded instead as
key_sin/key_cos -- see ml/train.py's key_to_sin_cos()). Spotify's features
come from its own internal audio-analysis pipeline, which isn't public. So
at inference time, this module computes signal-derived *proxies* for each
one from the actual audio buffer via librosa, using techniques already
used elsewhere in this codebase (perception/context_classifier.py's
envelope-modulation speech detector, the harmonic/percussive split used
for stem separation). This is the same "real feature, honestly-documented
approximation" pattern the project already uses for Demucs -> HPSS and the
vision-based EQ reader -- see the per-feature notes below for exactly how
good/weak each proxy is.

Never raises: any failure (missing model artifacts, librosa error, too
little audio) returns a GenreResult with ok=False and a reason, so a
missing `ml/models/` directory (e.g. before anyone's run `python
ml/train.py`) degrades to "no genre signal" rather than crashing the
pipeline -- same fallback philosophy as the rest of perception/.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

_ML_DIR = Path(__file__).parent.parent / "ml"
_MODELS_DIR = _ML_DIR / "models"

# Kept as plain local constants (not imported from ml/model_def.py) on
# purpose: model_def.py imports torch at module level, and this module
# must stay importable -- with genre classification simply unavailable --
# even in an environment with no torch/sklearn installed at all, exactly
# like the rest of perception/ degrades gracefully without Demucs, an
# LLM key, etc. Every trained-model load is lazy, inside _ModelBundle.load()
# below. Keep these two lists in sync with ml/model_def.py by hand.
FEATURE_COLUMNS = [
    "danceability", "energy", "key_sin", "key_cos", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "time_signature",
]
GENRE_BUCKETS = [
    "electronic_dance", "rock_metal", "hiphop_rnb", "pop", "acoustic_folk",
    "classical_jazz", "chill_ambient", "world_latin",
]


def _key_to_sin_cos(key: int) -> tuple[float, float]:
    """Must match ml/train.py's key_to_sin_cos() exactly -- duplicated
    (not imported) for the same reason FEATURE_COLUMNS is duplicated above."""
    if key < 0:
        return 0.0, 0.0
    angle = 2 * np.pi * key / 12.0
    return float(np.sin(angle)), float(np.cos(angle))

# liveness and time_signature are the two features this module can't
# meaningfully estimate from a short buffer (liveness needs audience-noise
# modeling; time signature needs several bars of stable beat tracking).
# Rather than guess, they're pinned to the training set's typical values so
# they contribute ~no signal and don't mislead the model -- the other 10
# real, signal-derived features do the actual work.
_LIVENESS_DEFAULT = 0.2
_TIME_SIGNATURE_DEFAULT = 4.0


@dataclass
class GenreResult:
    ok: bool
    bucket: Optional[str] = None
    confidence: float = 0.0
    all_probabilities: Dict[str, float] = field(default_factory=dict)
    proxy_features: Dict[str, float] = field(default_factory=dict)
    model_used: str = ""
    reason: str = ""


def _extract_proxy_features(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """Signal-derived stand-ins for Spotify's 12 audio features.

    See the module docstring for the general approach. Per-feature notes:
      - loudness, tempo: directly measured, same units as Spotify (dB, BPM).
      - speechiness: reuses context_classifier's syllable-rate envelope
        modulation score directly -- it's a real speech/non-speech signal.
      - acousticness: harmonic energy share from an HPSS split (tonal
        acoustic content vs. percussive/synthetic).
      - instrumentalness: 1 - speechiness proxy (vocals dominate the
        speech-band envelope modulation; an imperfect but directional proxy).
      - danceability: rhythmic regularity, via the strongest periodicity
        peak in the onset-strength tempogram.
      - key_sin/key_cos, mode: chroma-based -- key from the dominant pitch
        class (then cyclically encoded, see _key_to_sin_cos above), mode
        from a simplified major/minor triad energy comparison.
      - valence: blends normalized spectral brightness with the mode
        estimate (brighter + major-leaning ~ more "positive"). The
        weakest proxy here -- valence is a genuinely hard signal to infer
        from audio alone, even for Spotify's own model.
      - energy: normalized RMS.
      - liveness, time_signature: not estimated -- see the defaults above.
    """
    y = audio.astype(np.float32)
    if len(y) < sr // 2:
        raise ValueError("need at least 0.5s of audio")

    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2) + 1e-12))
    loudness_db = float(np.clip(20 * np.log10(rms + 1e-12), -60.0, 0.0))
    energy = float(np.clip((loudness_db + 60.0) / 60.0, 0.0, 1.0))

    harmonic, percussive = librosa.effects.hpss(y)
    h_energy = float(np.sum(harmonic.astype(np.float64) ** 2))
    p_energy = float(np.sum(percussive.astype(np.float64) ** 2))
    acousticness = float(np.clip(h_energy / (h_energy + p_energy + 1e-9), 0.0, 1.0))

    envelope = np.abs(librosa.util.normalize(y))
    hop = max(1, sr // 100)
    frames = envelope[: (len(envelope) // hop) * hop].reshape(-1, hop).mean(axis=1)
    speechiness = 0.0
    if len(frames) >= 8:
        frames = frames - frames.mean()
        spec = np.abs(np.fft.rfft(frames))
        freqs = np.fft.rfftfreq(len(frames), d=hop / sr)
        total = float(np.sum(spec) + 1e-9)
        band = (freqs >= 2.0) & (freqs <= 8.0)
        speechiness = float(np.clip(np.sum(spec[band]) / total, 0.0, 1.0))
    instrumentalness = float(np.clip(1.0 - speechiness, 0.0, 1.0))

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0]) if np.ndim(tempo) else float(tempo)
    if not np.isfinite(tempo) or tempo <= 0:
        # No detectable beat (e.g. a sustained tone/drone with no rhythmic
        # onsets) -- fall back to a neutral, dataset-typical BPM rather than
        # feeding the model a value real training data never had (Spotify's
        # own tempo feature is never 0 for an actual track).
        tempo = 120.0
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    danceability = 0.5
    if len(onset_env) > 16:
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        periodicity_strength = np.max(np.mean(tempogram, axis=1))
        danceability = float(np.clip(periodicity_strength / (np.mean(tempogram) + 1e-6) / 10.0, 0.0, 1.0))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    key = int(np.argmax(chroma_mean))
    key_sin, key_cos = _key_to_sin_cos(key)
    major_third = chroma_mean[(key + 4) % 12]
    minor_third = chroma_mean[(key + 3) % 12]
    mode = 1 if major_third >= minor_third else 0

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    brightness = float(np.clip(centroid / (sr / 4), 0.0, 1.0))
    valence = float(np.clip(0.6 * brightness + 0.4 * mode, 0.0, 1.0))

    return {
        "danceability": danceability, "energy": energy,
        "key_sin": key_sin, "key_cos": key_cos,
        "loudness": loudness_db, "mode": float(mode), "speechiness": speechiness,
        "acousticness": acousticness, "instrumentalness": instrumentalness,
        "liveness": _LIVENESS_DEFAULT, "valence": valence, "tempo": tempo,
        "time_signature": _TIME_SIGNATURE_DEFAULT,
    }


class _ModelBundle:
    """Lazily loads + caches the scaler, label encoder, and every trained
    model, so repeated calls in one process (e.g. across Streamlit reruns)
    don't touch disk / re-init torch each time."""

    def __init__(self):
        self.scaler = None
        self.label_encoder = None
        self.sklearn_models: Dict[str, object] = {}
        self.torch_model = None
        self.metrics: dict = {}
        self.loaded = False
        self.load_error = ""

    def load(self):
        if self.loaded or self.load_error:
            return
        try:
            import joblib
            self.scaler = joblib.load(_MODELS_DIR / "scaler.joblib")
            self.label_encoder = joblib.load(_MODELS_DIR / "label_encoder.joblib")
            for name in ("logistic_regression", "gradient_boosting"):
                path = _MODELS_DIR / f"{name}.joblib"
                if path.exists():
                    self.sklearn_models[name] = joblib.load(path)

            nn_path = _MODELS_DIR / "neural_net.pt"
            if nn_path.exists():
                import torch
                import sys
                sys.path.insert(0, str(_MODELS_DIR.parent))
                from model_def import GenreMLP
                model = GenreMLP()
                model.load_state_dict(torch.load(nn_path, map_location="cpu", weights_only=True))
                model.eval()
                self.torch_model = model

            metrics_path = _MODELS_DIR / "metrics.json"
            if metrics_path.exists():
                self.metrics = json.loads(metrics_path.read_text())

            self.loaded = True
        except Exception as exc:
            self.load_error = str(exc)


_bundle = _ModelBundle()


def available() -> bool:
    _bundle.load()
    return _bundle.loaded and bool(_bundle.sklearn_models or _bundle.torch_model is not None)


def default_model_name() -> str:
    _bundle.load()
    return _bundle.metrics.get("best_model", "gradient_boosting")


def list_models() -> list[str]:
    _bundle.load()
    names = list(_bundle.sklearn_models.keys())
    if _bundle.torch_model is not None:
        names.append("neural_net")
    return names


def classify_genre(audio: np.ndarray, sr: int, model_name: str = "auto") -> GenreResult:
    """Predict an EQ-relevant genre bucket from a content audio buffer.

    model_name: "auto" uses the best-performing model per ml/models/
    metrics.json (falls back to any available model); or pass one of
    "logistic_regression", "gradient_boosting", "neural_net" explicitly.
    """
    if not _HAS_LIBROSA:
        return GenreResult(ok=False, reason="librosa not installed")

    _bundle.load()
    if not _bundle.loaded:
        return GenreResult(ok=False, reason=(
            f"No trained model found in {_MODELS_DIR} -- run `python ml/train.py` "
            f"once to generate it. ({_bundle.load_error})" if _bundle.load_error else
            f"No trained model found in {_MODELS_DIR} -- run `python ml/train.py` once."
        ))

    try:
        proxy = _extract_proxy_features(audio, sr)
    except Exception as exc:
        return GenreResult(ok=False, reason=f"feature extraction failed: {exc}")

    x = np.array([[proxy[c] for c in FEATURE_COLUMNS]], dtype=np.float32)
    x_scaled = _bundle.scaler.transform(x)

    name = model_name
    if name == "auto":
        name = default_model_name()
        if name not in _bundle.sklearn_models and not (name == "neural_net" and _bundle.torch_model is not None):
            available_names = list_models()
            if not available_names:
                return GenreResult(ok=False, reason="no trained models available")
            name = available_names[0]

    try:
        if name == "neural_net":
            if _bundle.torch_model is None:
                return GenreResult(ok=False, reason="neural_net model not available")
            import torch
            with torch.no_grad():
                logits = _bundle.torch_model(torch.tensor(x_scaled, dtype=torch.float32))
                probs = torch.softmax(logits, dim=1).numpy()[0]
        else:
            model = _bundle.sklearn_models.get(name)
            if model is None:
                return GenreResult(ok=False, reason=f"model '{name}' not available")
            probs = model.predict_proba(x_scaled)[0]
    except Exception as exc:
        return GenreResult(ok=False, reason=f"inference failed: {exc}")

    classes = _bundle.label_encoder.classes_
    top_idx = int(np.argmax(probs))
    return GenreResult(
        ok=True,
        bucket=str(classes[top_idx]),
        confidence=float(probs[top_idx]),
        all_probabilities={str(c): float(p) for c, p in zip(classes, probs)},
        proxy_features=proxy,
        model_used=name,
    )
