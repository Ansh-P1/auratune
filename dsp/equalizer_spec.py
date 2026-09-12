"""
EqualizerSpec -- a description of *the EQ the user actually has in front of
them* (a phone EQ app, a streaming-app EQ, a car head unit...).

AuraTune's agents reason about an ideal continuous target curve. Every real
EQ, though, is a fixed grid: N sliders at fixed centre frequencies, each with
a limited gain range and a fixed step (some apps move in 1 dB, some in 0.5,
Wavelet in 0.1). This class captures that grid so `dsp.eq_projection` can
snap the ideal curve onto whatever the user can actually dial in.

Specs come from three places, all interchangeable:
  - the built-in PRESETS below,
  - JSON files dropped in `eq_specs/` (e.g. one written from a screenshot),
  - the manual editor in the Streamlit app.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Tuple

_SPEC_DIR = Path(__file__).parent.parent / "eq_specs"


@dataclass
class EqualizerSpec:
    name: str
    band_freqs_hz: List[float]
    gain_min_db: float = -12.0
    gain_max_db: float = 12.0
    step_db: float = 1.0          # 0.0 => treat as continuous
    has_preamp: bool = False
    preamp_min_db: float = -12.0
    preamp_max_db: float = 0.0
    notes: str = ""

    # -- quantisation helpers -------------------------------------------
    def _snap(self, value: float, step: float) -> float:
        if step and step > 0:
            value = round(value / step) * step
        # kill -0.0 and floating-point dust from the divide/multiply
        return round(value + 0.0, 4)

    def quantize_gain(self, gain_db: float) -> Tuple[float, bool]:
        """Snap a desired band gain to this EQ's step + range.

        Returns (value_the_user_can_set, was_clamped_by_the_range).
        """
        snapped = self._snap(gain_db, self.step_db)
        clamped = min(self.gain_max_db, max(self.gain_min_db, snapped))
        was_clamped = abs(clamped - snapped) > 1e-6
        return clamped, was_clamped

    def quantize_preamp(self, preamp_db: float) -> float:
        snapped = self._snap(preamp_db, self.step_db)
        return min(self.preamp_max_db, max(self.preamp_min_db, snapped))

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "EqualizerSpec":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean = {k: v for k, v in d.items() if k in fields}
        clean["band_freqs_hz"] = [float(f) for f in clean.get("band_freqs_hz", [])]
        return cls(**clean)


# ---------------------------------------------------------------------------
# Built-in presets. band_freqs_hz are the slider centre frequencies exactly
# as the app labels them.
# ---------------------------------------------------------------------------
PRESETS: Dict[str, EqualizerSpec] = {
    "wavelet_9band": EqualizerSpec(
        name="Wavelet - Graphic EQ (9-band)",
        band_freqs_hz=[62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000],
        gain_min_db=-12.0, gain_max_db=12.0, step_db=0.1,
        has_preamp=False,
        notes="Read from the user's screenshot (2026-09-10). Wavelet shows "
              "0.1 dB precision and has no separate preamp on this screen.",
    ),
    "iso_10band": EqualizerSpec(
        name="10-band ISO (31 Hz - 16 kHz)",
        band_freqs_hz=[31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000],
        gain_min_db=-12.0, gain_max_db=12.0, step_db=1.0,
        has_preamp=True,
        notes="Classic 10-band graphic EQ (iTunes / many Android skins).",
    ),
    "spotify_5band": EqualizerSpec(
        name="Spotify (5-band)",
        band_freqs_hz=[60, 150, 400, 1000, 2400, 15000],
        gain_min_db=-6.0, gain_max_db=6.0, step_db=0.1,
        has_preamp=False,
        notes="Spotify's in-app EQ. Slider dB values are approximate.",
    ),
    "car_3band": EqualizerSpec(
        name="Car head unit (Bass / Mid / Treble)",
        band_freqs_hz=[100, 1000, 10000],
        gain_min_db=-10.0, gain_max_db=10.0, step_db=1.0,
        has_preamp=False,
        notes="Typical 3-knob car stereo tone control.",
    ),
}


def load_file_specs() -> Dict[str, EqualizerSpec]:
    """Every EqualizerSpec JSON file in `eq_specs/`, keyed by file stem."""
    out: Dict[str, EqualizerSpec] = {}
    if not _SPEC_DIR.is_dir():
        return out
    for path in sorted(_SPEC_DIR.glob("*.json")):
        try:
            out[path.stem] = EqualizerSpec.from_dict(json.loads(path.read_text()))
        except Exception:
            continue
    return out


def all_specs() -> Dict[str, EqualizerSpec]:
    """Presets first, then anything in `eq_specs/` (file wins on key clash)."""
    merged = dict(PRESETS)
    merged.update(load_file_specs())
    return merged


def save_spec(key: str, spec: EqualizerSpec) -> Path:
    """Persist a spec to `eq_specs/<key>.json` (used when reading a screenshot)."""
    _SPEC_DIR.mkdir(exist_ok=True)
    path = _SPEC_DIR / f"{key}.json"
    path.write_text(json.dumps(spec.to_dict(), indent=2))
    return path
