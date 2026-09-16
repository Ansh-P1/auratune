"""
Tests for perception/noise_classifier.py.

Needs librosa (like test_classifier.py) for the feature-extraction tests.
The "no trained model" fallback test needs neither librosa nor a trained
model, so it always runs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from perception import noise_classifier as nc


def test_classify_noise_never_raises_on_silence():
    tiny = np.zeros(100, dtype=np.float32)
    result = nc.classify_noise(tiny, sr=44100)
    assert result.ok is False
    assert result.reason


def test_classify_noise_reports_missing_model_or_extracts_features():
    """Whatever this environment's state is (no librosa, no trained model
    yet, or a fully working setup), classify_noise must return a valid
    NoiseResult and never raise."""
    sr = 44100
    t = np.arange(sr * 2) / sr
    fake_noise = (np.random.randn(len(t)) * 0.05).astype(np.float32)
    result = nc.classify_noise(fake_noise, sr=sr)
    assert isinstance(result.ok, bool)
    if result.ok:
        assert result.bucket in nc.NOISE_BUCKETS
        assert 0.0 <= result.confidence <= 1.0
        assert abs(sum(result.all_probabilities.values()) - 1.0) < 1e-3
        assert set(result.features.keys()) == set(nc.NOISE_FEATURE_COLUMNS)
    else:
        assert result.reason


def test_extract_noise_features_are_in_expected_ranges():
    if not nc._HAS_LIBROSA:
        print("  (skipped: librosa not installed)")
        return
    sr = 44100
    t = np.arange(sr * 2) / sr
    fake_noise = (np.sin(2 * np.pi * 220 * t) * 0.3 +
                  np.random.randn(len(t)) * 0.02).astype(np.float32)
    feats = nc.extract_noise_features(fake_noise, sr)
    assert set(feats.keys()) == set(nc.NOISE_FEATURE_COLUMNS)
    assert -80.0 <= feats["rms_db"] <= 0.0
    assert feats["spectral_centroid"] > 0
    assert feats["spectral_bandwidth"] > 0
    assert 0.0 <= feats["spectral_flatness"] <= 1.0
    assert feats["spectral_rolloff"] > 0
    assert 0.0 <= feats["zero_crossing_rate"] <= 1.0
    assert 0.0 <= feats["harmonic_ratio"] <= 1.0
    assert feats["onset_rate"] >= 0.0


def test_noise_model_predicts_when_available():
    """Only runs the real prediction path if librosa + a trained noise
    model are both available -- otherwise a no-op (see the always-run
    graceful-failure test above)."""
    if not nc._HAS_LIBROSA:
        print("  (skipped: librosa not installed)")
        return
    if not nc.available():
        print("  (skipped: no trained noise model in ml/models/ -- run `python ml/train_noise.py`)")
        return
    sr = 44100
    t = np.arange(sr * 3) / sr
    fake_traffic = (np.random.randn(len(t)) * 0.1 + np.sin(2 * np.pi * 90 * t) * 0.1).astype(np.float32)
    result = nc.classify_noise(fake_traffic, sr=sr)
    assert result.ok, result.reason
    assert result.bucket in nc.NOISE_BUCKETS
    assert result.model_used


if __name__ == "__main__":
    test_classify_noise_never_raises_on_silence()
    test_classify_noise_reports_missing_model_or_extracts_features()
    test_extract_noise_features_are_in_expected_ranges()
    test_noise_model_predicts_when_available()
    print("All noise classifier tests passed.")
