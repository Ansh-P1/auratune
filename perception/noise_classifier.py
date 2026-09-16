"""
Ambient-noise-type classification from real audio, using the models
trained in ml/train_noise.py on ESC-50 (2,000 real labeled environmental
sound clips -- see ml/README.md).

Unlike the genre classifier, there's no proxy-feature problem here:
extract_noise_features() below is the ONE place these 13 features are
computed, imported directly by ml/train_noise.py for training and by
classify_noise() here for inference on a live/synth ambient buffer.
Training and inference always see the literal same function -- not an
approximation of some other system's private features.

Never raises: any failure (missing model artifacts, librosa error, too
little audio) returns a NoiseResult with ok=False and a reason, so a
missing `ml/models/` directory (e.g. before anyone's run
`python ml/train_noise.py`) degrades to "no noise-type signal" rather
than crashing the pipeline -- same fallback philosophy as the rest of
perception/.
"""
from __future__ import annotations

import json
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

# Kept as plain local constants (not imported from ml/noise_model_def.py)
# for the same reason genre_classifier.py does this: this module must
# stay importable -- with noise-type classification simply unavailable --
# even with no sklearn/torch installed. Keep in sync with
# ml/noise_model_def.py by hand.
NOISE_BUCKETS = [
    "calm_nature", "domestic_ambient", "human_activity",
    "impulsive_transient", "mechanical_drone", "traffic_urban",
]
NOISE_FEATURE_COLUMNS = [
    "rms_db", "spectral_centroid", "spectral_bandwidth", "spectral_flatness",
    "spectral_rolloff", "zero_crossing_rate", "harmonic_ratio", "onset_rate",
    "mfcc1", "mfcc2", "mfcc3", "mfcc4", "mfcc5",
]


@dataclass
class NoiseResult:
    ok: bool
    bucket: Optional[str] = None
    confidence: float = 0.0
    all_probabilities: Dict[str, float] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)
    model_used: str = ""
    reason: str = ""


def extract_noise_features(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """The single source of truth for what a "noise fingerprint" is --
    used identically at training time (on ESC-50 clips) and inference
    time (on a live/synth ambient buffer). All 13 features are genuine
    signal measurements, no proxying involved:
      - rms_db: overall loudness.
      - spectral_centroid/bandwidth/rolloff: where the energy sits in the
        spectrum and how spread out it is -- separates a low rumbling
        drone from a bright, sharp transient.
      - spectral_flatness: tonal/periodic vs. noise-like.
      - zero_crossing_rate: coarse noisiness/high-frequency content proxy.
      - harmonic_ratio: tonal (voices, engines with a pitch) vs.
        percussive/broadband (impacts, wind, static).
      - onset_rate: onsets per second -- separates a steady drone
        (vacuum, engine) from a bursty/eventful scene (footsteps,
        clapping, door knocks).
      - mfcc1-5: coarse timbral texture, the same features speech/audio
        classifiers have used for decades.
    """
    y = audio.astype(np.float32)
    if len(y) < sr // 2:
        raise ValueError("need at least 0.5s of audio")
    duration_sec = len(y) / sr

    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2) + 1e-12))
    rms_db = float(np.clip(20 * np.log10(rms + 1e-12), -80.0, 0.0))

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    harmonic, percussive = librosa.effects.hpss(y)
    h_energy = float(np.sum(harmonic.astype(np.float64) ** 2))
    p_energy = float(np.sum(percussive.astype(np.float64) ** 2))
    harmonic_ratio = float(np.clip(h_energy / (h_energy + p_energy + 1e-9), 0.0, 1.0))

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_rate = float(len(onset_frames) / duration_sec) if duration_sec > 0 else 0.0

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
    mfcc_means = mfcc.mean(axis=1)

    return {
        "rms_db": rms_db, "spectral_centroid": centroid, "spectral_bandwidth": bandwidth,
        "spectral_flatness": flatness, "spectral_rolloff": rolloff,
        "zero_crossing_rate": zcr, "harmonic_ratio": harmonic_ratio,
        "onset_rate": onset_rate,
        "mfcc1": float(mfcc_means[0]), "mfcc2": float(mfcc_means[1]),
        "mfcc3": float(mfcc_means[2]), "mfcc4": float(mfcc_means[3]),
        "mfcc5": float(mfcc_means[4]),
    }


class _ModelBundle:
    """Lazily loads + caches the scaler, label encoder, and every trained
    noise model, so repeated calls in one process (e.g. across Streamlit
    reruns) don't touch disk each time."""

    def __init__(self):
        self.scaler = None
        self.label_encoder = None
        self.sklearn_models: Dict[str, object] = {}
        self.metrics: dict = {}
        self.loaded = False
        self.load_error = ""

    def load(self):
        if self.loaded or self.load_error:
            return
        try:
            import joblib
            self.scaler = joblib.load(_MODELS_DIR / "noise_scaler.joblib")
            self.label_encoder = joblib.load(_MODELS_DIR / "noise_label_encoder.joblib")
            for name in ("logistic_regression", "random_forest", "gradient_boosting"):
                path = _MODELS_DIR / f"noise_{name}.joblib"
                if path.exists():
                    self.sklearn_models[name] = joblib.load(path)

            metrics_path = _MODELS_DIR / "noise_metrics.json"
            if metrics_path.exists():
                self.metrics = json.loads(metrics_path.read_text())

            self.loaded = True
        except Exception as exc:
            self.load_error = str(exc)


_bundle = _ModelBundle()


def available() -> bool:
    _bundle.load()
    return _bundle.loaded and bool(_bundle.sklearn_models)


def default_model_name() -> str:
    _bundle.load()
    return _bundle.metrics.get("best_model", "random_forest")


def list_models() -> list[str]:
    _bundle.load()
    return list(_bundle.sklearn_models.keys())


def classify_noise(audio: np.ndarray, sr: int, model_name: str = "auto") -> NoiseResult:
    """Predict an EQ-relevant ambient-noise bucket from an ambient/mic buffer.

    model_name: "auto" uses the best-performing model per
    ml/models/noise_metrics.json; or pass one of "logistic_regression",
    "random_forest", "gradient_boosting" explicitly.
    """
    if not _HAS_LIBROSA:
        return NoiseResult(ok=False, reason="librosa not installed")

    _bundle.load()
    if not _bundle.loaded:
        return NoiseResult(ok=False, reason=(
            f"No trained noise model found in {_MODELS_DIR} -- run "
            f"`python ml/train_noise.py` once to generate it. ({_bundle.load_error})"
            if _bundle.load_error else
            f"No trained noise model found in {_MODELS_DIR} -- run `python ml/train_noise.py` once."
        ))

    try:
        feats = extract_noise_features(audio, sr)
    except Exception as exc:
        return NoiseResult(ok=False, reason=f"feature extraction failed: {exc}")

    x = np.array([[feats[c] for c in NOISE_FEATURE_COLUMNS]], dtype=np.float32)
    x_scaled = _bundle.scaler.transform(x)

    name = model_name
    if name == "auto":
        name = default_model_name()
        if name not in _bundle.sklearn_models:
            available_names = list_models()
            if not available_names:
                return NoiseResult(ok=False, reason="no trained noise models available")
            name = available_names[0]

    model = _bundle.sklearn_models.get(name)
    if model is None:
        return NoiseResult(ok=False, reason=f"model '{name}' not available")

    try:
        probs = model.predict_proba(x_scaled)[0]
    except Exception as exc:
        return NoiseResult(ok=False, reason=f"inference failed: {exc}")

    classes = _bundle.label_encoder.classes_
    top_idx = int(np.argmax(probs))
    return NoiseResult(
        ok=True,
        bucket=str(classes[top_idx]),
        confidence=float(probs[top_idx]),
        all_probabilities={str(c): float(p) for c, p in zip(classes, probs)},
        features=feats,
        model_used=name,
    )
