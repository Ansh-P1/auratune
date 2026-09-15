"""
Tests for perception/genre_classifier.py.

Needs librosa (like test_classifier.py) for the feature-extraction tests.
The "no trained model" fallback test needs neither librosa nor a trained
model, so it always runs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from perception import genre_classifier as gc


def test_classify_genre_never_raises_on_silence():
    # Too little audio (< 0.5s) should fail gracefully, not raise.
    tiny = np.zeros(100, dtype=np.float32)
    result = gc.classify_genre(tiny, sr=44100)
    assert result.ok is False
    assert result.reason  # some explanation string, whatever the cause


def test_classify_genre_reports_missing_model_or_extracts_features():
    """Whatever this environment's state is (no librosa, no trained model
    yet, or a fully working setup), classify_genre must return a valid
    GenreResult and never raise."""
    sr = 44100
    t = np.arange(sr * 2) / sr
    fake_music = (np.sin(2 * np.pi * 220 * t) * 0.3 +
                  np.sin(2 * np.pi * 440 * t) * 0.15).astype(np.float32)
    result = gc.classify_genre(fake_music, sr=sr)
    assert isinstance(result.ok, bool)
    if result.ok:
        assert result.bucket in gc.GENRE_BUCKETS
        assert 0.0 <= result.confidence <= 1.0
        assert abs(sum(result.all_probabilities.values()) - 1.0) < 1e-3
        assert set(result.proxy_features.keys()) == set(gc.FEATURE_COLUMNS)
    else:
        assert result.reason


def test_proxy_features_are_in_expected_ranges():
    """Only runs the real feature extraction if librosa + a trained model
    are both available -- otherwise this is a no-op (see test above for
    the graceful-failure path that always runs)."""
    if not gc._HAS_LIBROSA:
        print("  (skipped: librosa not installed)")
        return
    if not gc.available():
        print("  (skipped: no trained model in ml/models/ -- run `python ml/train.py`)")
        return
    sr = 44100
    t = np.arange(sr * 2) / sr
    fake_music = (np.sin(2 * np.pi * 220 * t) * 0.3 +
                  np.sin(2 * np.pi * 440 * t) * 0.15).astype(np.float32)
    result = gc.classify_genre(fake_music, sr=sr)
    assert result.ok, result.reason
    f = result.proxy_features
    assert -60.0 <= f["loudness"] <= 0.0
    assert 0.0 <= f["energy"] <= 1.0
    assert 0.0 <= f["danceability"] <= 1.0
    assert 0.0 <= f["acousticness"] <= 1.0
    assert 0.0 <= f["instrumentalness"] <= 1.0
    assert 0.0 <= f["speechiness"] <= 1.0
    assert 0.0 <= f["valence"] <= 1.0
    assert f["mode"] in (0.0, 1.0)
    assert -1.0 <= f["key_sin"] <= 1.0
    assert -1.0 <= f["key_cos"] <= 1.0
    assert f["tempo"] > 0


if __name__ == "__main__":
    test_classify_genre_never_raises_on_silence()
    test_classify_genre_reports_missing_model_or_extracts_features()
    test_proxy_features_are_in_expected_ranges()
    print("All genre classifier tests passed.")
