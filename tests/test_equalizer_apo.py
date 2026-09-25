"""
Tests for the Equalizer APO live integration (integrations/equalizer_apo.py)
and the live-apply hook in dsp/eq_projection.py.

Equalizer APO itself is Windows-only, but everything except the audible part
is just rendering and writing a text file -- AURATUNE_APO_CONFIG points the
writer at a temp file so these run anywhere, including CI.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dsp.eq_projection import project_curve
from dsp.equalizer_spec import EqualizerSpec, all_specs
from dsp.parametric_eq import EQBand, ParametricEQ, TargetCurve
from integrations.equalizer_apo import (EqualizerAPOUnavailable, config_path,
                                        is_available, render_config,
                                        unavailable_reason, write_config,
                                        write_projection, apply_curve)


def _tmp_config() -> Path:
    return Path(tempfile.mkdtemp()) / "config.txt"


def test_render_matches_equalizer_apo_filter_syntax():
    text = render_config([EQBand("peaking", 3000.0, 3.0, 1.0)], preamp_db=-2.0)
    assert "Preamp: -2 dB" in text
    assert "Filter 1: ON PK Fc 3000 Hz Gain 3 dB Q 1" in text
    # Every non-comment line is one of APO's two directives.
    for line in text.splitlines():
        if line and not line.startswith("#"):
            assert line.startswith(("Preamp:", "Filter ")), line


def test_shelving_bands_use_the_shelf_filter_types():
    text = render_config(TargetCurve(name="t", bass_gain_db=2.0,
                                     treble_gain_db=-1.0).to_bands())
    assert "LSC Fc 100 Hz" in text      # low shelf
    assert "HSC Fc 8000 Hz" in text     # high shelf


def test_flat_band_is_written_off_not_dropped():
    """Numbering has to stay stable between writes, so a 0 dB band stays put."""
    text = render_config(TargetCurve(name="flat").to_bands())
    assert text.count("Filter ") == 3
    assert "OFF" in text


def test_write_config_is_atomic_and_leaves_no_temp_files():
    target = _tmp_config()
    result = write_config([EQBand("peaking", 1000.0, 1.5, 1.0)], -1.0, path=target)
    assert target.read_text().endswith("\n")
    assert result.filters == 1 and result.bytes_written > 0
    # The temp file used for the atomic replace must not survive.
    leftovers = [p.name for p in target.parent.iterdir() if p.name != target.name]
    assert leftovers == [], leftovers


def test_rewriting_replaces_rather_than_appends():
    target = _tmp_config()
    write_config([EQBand("peaking", 1000.0, 1.0, 1.0)], 0.0, path=target)
    write_config([EQBand("peaking", 1000.0, -4.0, 1.0)], 0.0, path=target)
    text = target.read_text()
    assert text.count("Filter 1:") == 1
    assert "Gain -4 dB" in text


def test_apply_curve_uses_volume_as_preamp():
    target = _tmp_config()
    apply_curve(TargetCurve(name="t", volume_db=-3.5, bass_gain_db=2.0), path=target)
    assert "Preamp: -3.5 dB" in target.read_text()


def test_missing_install_raises_a_clear_message_not_a_crash():
    missing = Path(tempfile.mkdtemp()) / "no-such-dir" / "config.txt"
    assert not is_available(missing)
    assert "equalizerapo.com" in unavailable_reason(missing) or \
        "Windows-only" in unavailable_reason(missing)
    try:
        write_config([EQBand("peaking", 1000.0, 1.0, 1.0)], 0.0, path=missing)
    except EqualizerAPOUnavailable as exc:
        assert str(exc)
    else:
        raise AssertionError("expected EqualizerAPOUnavailable")


def test_config_path_follows_the_env_override():
    saved = os.environ.get("AURATUNE_APO_CONFIG")
    try:
        os.environ["AURATUNE_APO_CONFIG"] = "/tmp/auratune-apo-test.txt"
        assert config_path() == Path("/tmp/auratune-apo-test.txt")
    finally:
        os.environ.pop("AURATUNE_APO_CONFIG", None)
        if saved is not None:
            os.environ["AURATUNE_APO_CONFIG"] = saved


def test_equalizer_apo_is_a_selectable_spec():
    spec = all_specs().get("equalizer_apo")
    assert spec is not None, "eq_specs/equalizer_apo.json should be loadable"
    assert spec.live_target == "equalizer_apo"
    assert spec.has_preamp


def test_projection_writes_a_live_file_only_when_asked():
    target = _tmp_config()
    saved = os.environ.get("AURATUNE_APO_CONFIG")
    os.environ["AURATUNE_APO_CONFIG"] = str(target)
    try:
        spec = all_specs()["equalizer_apo"]
        curve = TargetCurve(name="live", volume_db=-1.0, bass_gain_db=-2.0,
                            presence_gain_db=3.0)

        quiet = project_curve(curve, ParametricEQ(), spec)
        assert not quiet.live_applied
        assert not target.exists(), "projection must stay pure unless asked"

        applied = project_curve(curve, ParametricEQ(), spec, apply_live=True)
        assert applied.live_applied
        written = target.read_text()
        # What's written is what the dashboard's table shows.
        assert written.count("Filter ") == len(applied.bands)
        assert f"Preamp: " in written
    finally:
        os.environ.pop("AURATUNE_APO_CONFIG", None)
        if saved is not None:
            os.environ["AURATUNE_APO_CONFIG"] = saved


def test_a_normal_phone_spec_never_writes_anything():
    """Only specs that name a live target touch the filesystem."""
    spec = all_specs()["wavelet_9band"]
    assert spec.live_target == ""
    projected = project_curve(TargetCurve(name="t", bass_gain_db=1.0),
                              ParametricEQ(), spec, apply_live=True)
    assert not projected.live_applied


def test_missing_install_is_reported_not_raised_through_projection():
    """A user without Equalizer APO still gets correct numbers on screen."""
    saved = os.environ.get("AURATUNE_APO_CONFIG")
    os.environ["AURATUNE_APO_CONFIG"] = str(Path(tempfile.mkdtemp()) / "nope" / "config.txt")
    try:
        spec = all_specs()["equalizer_apo"]
        projected = project_curve(TargetCurve(name="t", bass_gain_db=1.0),
                                  ParametricEQ(), spec, apply_live=True)
        assert not projected.live_applied
        assert projected.live_detail, "should explain why it didn't apply"
        assert projected.bands, "slider values must still be produced"
    finally:
        os.environ.pop("AURATUNE_APO_CONFIG", None)
        if saved is not None:
            os.environ["AURATUNE_APO_CONFIG"] = saved


if __name__ == "__main__":
    test_render_matches_equalizer_apo_filter_syntax()
    test_shelving_bands_use_the_shelf_filter_types()
    test_flat_band_is_written_off_not_dropped()
    test_write_config_is_atomic_and_leaves_no_temp_files()
    test_rewriting_replaces_rather_than_appends()
    test_apply_curve_uses_volume_as_preamp()
    test_missing_install_raises_a_clear_message_not_a_crash()
    test_config_path_follows_the_env_override()
    test_equalizer_apo_is_a_selectable_spec()
    test_projection_writes_a_live_file_only_when_asked()
    test_a_normal_phone_spec_never_writes_anything()
    test_missing_install_is_reported_not_raised_through_projection()
    print("All Equalizer APO integration tests passed.")
