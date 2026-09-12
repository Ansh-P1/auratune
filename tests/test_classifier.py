import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from perception.context_classifier import classify, _classify_noise, _envelope_modulation_score

SR = 44100


def test_noise_level_thresholds():
    quiet = np.random.randn(SR) * 0.001
    moderate = np.random.randn(SR) * 0.03
    noisy = np.random.randn(SR) * 0.2
    assert _classify_noise(quiet) == "quiet"
    assert _classify_noise(moderate) == "moderate"
    assert _classify_noise(noisy) == "noisy"


def test_content_type_hint_is_authoritative():
    ambient = np.random.randn(SR) * 0.01
    content = np.random.randn(SR) * 0.1  # ambiguous signal
    ctx = classify(ambient, content, SR, content_type_hint="movie")
    assert ctx.content_type == "movie"


def test_speech_envelope_modulation_higher_than_stationary_tone():
    t = np.arange(SR) / SR
    syllable_env = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 4 * t))
    speech_like = np.sin(2 * np.pi * 200 * t) * syllable_env
    stationary_tone = np.sin(2 * np.pi * 200 * t)
    speech_score = _envelope_modulation_score(speech_like.astype(np.float32), SR)
    tone_score = _envelope_modulation_score(stationary_tone.astype(np.float32), SR)
    assert speech_score > tone_score


if __name__ == "__main__":
    test_noise_level_thresholds()
    test_content_type_hint_is_authoritative()
    test_speech_envelope_modulation_higher_than_stationary_tone()
    print("All classifier tests passed.")
