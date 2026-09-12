import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dsp.parametric_eq import ParametricEQ, TargetCurve
from dsp.equalizer_spec import EqualizerSpec, all_specs
from dsp.eq_projection import project_curve


def test_quantize_step_and_range():
    half = EqualizerSpec("half", [1000], gain_min_db=-6, gain_max_db=6, step_db=0.5)
    assert half.quantize_gain(2.24) == (2.0, False)
    assert half.quantize_gain(2.26) == (2.5, False)
    # beyond range -> clamped flag set
    val, clamped = half.quantize_gain(99.0)
    assert val == 6.0 and clamped is True

    whole = EqualizerSpec("whole", [1000], step_db=1.0)
    assert whole.quantize_gain(3.4)[0] == 3.0
    assert whole.quantize_gain(3.6)[0] == 4.0


def test_projection_tracks_curve_shape():
    eq = ParametricEQ(44100)
    curve = TargetCurve("shape", bass_gain_db=-4.0, presence_gain_db=6.0, treble_gain_db=0.0)
    spec = EqualizerSpec("t", [60, 250, 1000, 3000, 12000],
                         gain_min_db=-12, gain_max_db=12, step_db=0.1)
    proj = project_curve(curve, eq, spec)

    by_f = {b.freq_hz: b.set_gain_db for b in proj.bands}
    assert by_f[60] < -1.0          # bass cut shows up low
    assert by_f[3000] > 3.0         # presence boost shows up around 3 kHz
    assert abs(by_f[12000]) < 1.0   # flat where the curve is flat
    assert proj.fit_error_db < 1.5


def test_range_limit_is_reported():
    eq = ParametricEQ(44100)
    curve = TargetCurve("hot", presence_gain_db=10.0)
    spec = EqualizerSpec("tiny", [3000], gain_min_db=-4, gain_max_db=4, step_db=1.0)
    proj = project_curve(curve, eq, spec)
    assert proj.bands[0].set_gain_db == 4.0
    assert 3000 in proj.clipped_freqs


def test_preamp_cancels_largest_boost():
    eq = ParametricEQ(44100)
    curve = TargetCurve("boosty", presence_gain_db=6.0)
    spec = EqualizerSpec("p", [1000, 3000, 8000], gain_min_db=-12, gain_max_db=12,
                         step_db=0.5, has_preamp=True, preamp_min_db=-12, preamp_max_db=0)
    proj = project_curve(curve, eq, spec)
    max_boost = max(b.set_gain_db for b in proj.bands)
    assert proj.preamp_db <= 0.0
    assert abs(proj.preamp_db + max_boost) < 0.6   # preamp ~ -max_boost


def test_wavelet_screenshot_spec_loads():
    specs = all_specs()
    assert "wavelet_9band" in specs
    w = specs["wavelet_9band"]
    assert len(w.band_freqs_hz) == 9
    assert w.step_db == 0.1
    assert w.band_freqs_hz[0] == 62.5 and w.band_freqs_hz[-1] == 16000


if __name__ == "__main__":
    test_quantize_step_and_range()
    test_projection_tracks_curve_shape()
    test_range_limit_is_reported()
    test_preamp_cancels_largest_boost()
    test_wavelet_screenshot_spec_loads()
    print("All EQ projection tests passed.")
