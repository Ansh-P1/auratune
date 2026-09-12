import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from dsp.parametric_eq import ParametricEQ, TargetCurve


def test_flat_curve_is_near_unity():
    eq = ParametricEQ(44100)
    flat = TargetCurve("flat")
    freqs, mag = eq.frequency_response(flat)
    assert np.allclose(mag, 0.0, atol=1e-6)


def test_bass_boost_raises_low_freq_gain():
    eq = ParametricEQ(44100)
    curve = TargetCurve("bass_boost", bass_gain_db=6.0)
    freqs, mag = eq.frequency_response(curve)
    low_band = mag[freqs < 150]
    high_band = mag[freqs > 5000]
    assert low_band.mean() > 3.0
    assert abs(high_band.mean()) < 1.0


def test_apply_does_not_clip_beyond_range():
    eq = ParametricEQ(44100)
    curve = TargetCurve("loud", volume_db=6.0, bass_gain_db=8.0, presence_gain_db=8.0)
    audio = (np.random.randn(44100) * 0.5).astype(np.float32)
    out = eq.apply(audio, curve)
    assert out.max() <= 1.0 and out.min() >= -1.0


def test_delta_reports_correct_signs():
    eq = ParametricEQ(44100)
    before = TargetCurve("before", bass_gain_db=0.0, presence_gain_db=0.0)
    after = TargetCurve("after", bass_gain_db=-2.0, presence_gain_db=3.0)
    deltas = eq.delta(before, after)
    assert deltas["bass_gain_db"] == -2.0
    assert deltas["presence_gain_db"] == 3.0


if __name__ == "__main__":
    test_flat_curve_is_near_unity()
    test_bass_boost_raises_low_freq_gain()
    test_apply_does_not_clip_beyond_range()
    test_delta_reports_correct_signs()
    print("All DSP tests passed.")
