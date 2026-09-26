"""
Project an ideal continuous target curve onto a real EQ's fixed band grid.

The agent pipeline produces a `TargetCurve` (low shelf / presence peak / high
shelf + an overall level). That's a smooth curve. A phone EQ app is a row of
sliders at fixed frequencies with a fixed step and a limited range. This
module reads the smooth curve's frequency response, samples it at the user's
actual slider frequencies, snaps each to what the app can dial in, and works
out a preamp so the boosts don't clip.

Nothing here is ML -- it's interpolation + rounding -- so it runs anywhere and
is fully deterministic/auditable, same as the rest of the DSP layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from dsp.parametric_eq import ParametricEQ, TargetCurve
from dsp.equalizer_spec import EqualizerSpec


@dataclass
class BandSetting:
    freq_hz: float
    ideal_gain_db: float   # the smooth target's gain at this frequency (shape only)
    set_gain_db: float     # what the user should actually set the slider to
    clamped: bool          # True if the app's range couldn't reach ideal_gain_db

    def to_dict(self) -> dict:
        return {
            "freq_hz": self.freq_hz,
            "ideal_gain_db": round(self.ideal_gain_db, 2),
            "set_gain_db": self.set_gain_db,
            "clamped": self.clamped,
        }


@dataclass
class ProjectedEQ:
    spec_name: str
    bands: List[BandSetting]
    preamp_db: float
    has_preamp: bool
    fit_error_db: float               # RMS(set - ideal) across bands
    clipped_freqs: List[float] = field(default_factory=list)
    ideal_freqs_hz: np.ndarray = field(default_factory=lambda: np.array([]))
    ideal_shape_db: np.ndarray = field(default_factory=lambda: np.array([]))
    # Set when the spec names a live target (see EqualizerSpec.live_target)
    # and project_curve was asked to apply it -- see apply_live below.
    live_applied: bool = False
    live_detail: str = ""

    def to_dict(self) -> dict:
        return {
            "spec_name": self.spec_name,
            "preamp_db": self.preamp_db,
            "has_preamp": self.has_preamp,
            "fit_error_db": round(self.fit_error_db, 2),
            "clipped_freqs": self.clipped_freqs,
            "bands": [b.to_dict() for b in self.bands],
            "live_applied": self.live_applied,
            "live_detail": self.live_detail,
        }

    def as_table_rows(self) -> List[dict]:
        rows = [{"Frequency": _fmt_hz(b.freq_hz), "Set to (dB)": f"{b.set_gain_db:+.1f}"}
                for b in self.bands]
        label = "Preamp" if self.has_preamp else "Volume / preamp*"
        rows.append({"Frequency": label, "Set to (dB)": f"{self.preamp_db:+.1f}"})
        return rows


def _fmt_hz(f: float) -> str:
    if f >= 1000:
        s = f"{f / 1000:.1f}".rstrip("0").rstrip(".")
        return f"{s} kHz"
    s = f"{f:.1f}".rstrip("0").rstrip(".")
    return f"{s} Hz"


def project_curve(curve: TargetCurve, eq: ParametricEQ, spec: EqualizerSpec,
                  apply_live: bool = False) -> ProjectedEQ:
    """Snap `curve` onto `spec`'s slider grid.

    The band values describe the curve's *shape* (relative to 0 dB); the
    curve's overall `volume_db` is folded into the preamp instead, because a
    graphic EQ has no "volume" slider.

    apply_live: when True *and* the spec names a live target (currently only
    Equalizer APO), also push the projected bands to that target so they land
    on whatever is actually playing. Off by default: projection stays a pure
    function unless a caller explicitly asks for the side effect. A target
    that isn't installed is recorded on the result, never raised -- the
    slider numbers are still correct and the app keeps working.
    """
    freqs, mag_db = eq.frequency_response(curve, n_points=1024)
    shape_db = mag_db - curve.volume_db          # strip the overall level

    log_f = np.log10(freqs)
    lo, hi = log_f[0], log_f[-1]

    bands: List[BandSetting] = []
    clipped: List[float] = []
    residuals: List[float] = []
    for f in spec.band_freqs_hz:
        ideal = float(np.interp(np.clip(np.log10(f), lo, hi), log_f, shape_db))
        set_gain, was_clamped = spec.quantize_gain(ideal)
        bands.append(BandSetting(freq_hz=float(f), ideal_gain_db=ideal,
                                 set_gain_db=set_gain, clamped=was_clamped))
        residuals.append(set_gain - ideal)
        if was_clamped:
            clipped.append(float(f))

    # Preamp: cancel the largest boost so nothing clips, then apply whatever
    # overall level the curve asked for. Apps without a preamp still get a
    # recommended number (shown as guidance -- lower the media/app volume).
    max_boost = max((b.set_gain_db for b in bands), default=0.0)
    preamp_raw = curve.volume_db - max(0.0, max_boost)
    if spec.has_preamp:
        preamp = spec.quantize_preamp(preamp_raw)
    else:
        preamp = round(min(0.0, preamp_raw), 1)

    fit_error = float(np.sqrt(np.mean(np.square(residuals)))) if residuals else 0.0

    projected = ProjectedEQ(
        spec_name=spec.name,
        bands=bands,
        preamp_db=preamp,
        has_preamp=spec.has_preamp,
        fit_error_db=fit_error,
        clipped_freqs=clipped,
        ideal_freqs_hz=freqs,
        ideal_shape_db=shape_db,
    )

    if apply_live and spec.live_target:
        _apply_live(projected, spec)
    return projected


def _apply_live(projected: ProjectedEQ, spec: EqualizerSpec) -> None:
    """Push a projection to a live EQ, recording what happened on it.

    Imported lazily so dsp/ keeps no hard dependency on integrations/, and
    so a machine without the live EQ installed pays nothing for it.
    """
    if spec.live_target != "equalizer_apo":
        projected.live_detail = f"unknown live target {spec.live_target!r}"
        return
    try:
        from integrations.equalizer_apo import (EqualizerAPOUnavailable,
                                                write_projection)
    except Exception as exc:  # pragma: no cover -- import guard
        projected.live_detail = f"live target unavailable: {exc}"
        return
    try:
        result = write_projection(projected)
    except EqualizerAPOUnavailable as exc:
        projected.live_detail = str(exc)
        return
    projected.live_applied = True
    projected.live_detail = (f"Applied live: {result.filters} filters + "
                             f"{result.preamp_db:+.1f} dB preamp -> {result.path}")
