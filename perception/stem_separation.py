"""
Stem separation.

Wraps Demucs (htdemucs) to split playing content into vocals / drums / bass /
other, which the context classifier and EQ decision agent use (e.g. "boost
presence" really means "boost the vocal stem's band").

Demucs needs its pretrained weights downloaded on first use. In locked-down
/ offline environments that download can fail -- this module catches that
and falls back to a lightweight harmonic-percussive split (librosa.hpss),
which gives a coarser but dependency-light approximation (vocals+harmonic
content vs. percussive/FX content) so the rest of the pipeline keeps working.

Swap `HAS_DEMUCS` behavior for a real deployment with network access to
Meta's model hub, or point DEMUCS_MODEL_DIR at pre-downloaded weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

try:
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    _HAS_DEMUCS = True
except Exception:
    _HAS_DEMUCS = False

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False


@dataclass
class Stems:
    vocals: Optional[np.ndarray]
    drums: Optional[np.ndarray]
    bass: Optional[np.ndarray]
    other: Optional[np.ndarray]
    backend: str  # "demucs" | "hpss_fallback" | "passthrough"


class StemSeparator:
    def __init__(self, model_name: str = "htdemucs"):
        self.model_name = model_name
        self._model = None
        self._backend = "passthrough"
        if _HAS_DEMUCS:
            try:
                self._model = get_model(model_name)
                self._model.eval()
                self._backend = "demucs"
            except Exception:
                # e.g. no network access to fetch pretrained weights
                self._model = None
                self._backend = "hpss_fallback" if _HAS_LIBROSA else "passthrough"
        else:
            self._backend = "hpss_fallback" if _HAS_LIBROSA else "passthrough"

    @property
    def backend(self) -> str:
        return self._backend

    def separate(self, audio: np.ndarray, sr: int) -> Stems:
        if self._backend == "demucs" and self._model is not None:
            return self._separate_demucs(audio, sr)
        if self._backend == "hpss_fallback":
            return self._separate_hpss(audio)
        return Stems(vocals=audio, drums=None, bass=None, other=None, backend="passthrough")

    def _separate_demucs(self, audio: np.ndarray, sr: int) -> Stems:
        wav = torch.tensor(audio, dtype=torch.float32)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0).repeat(2, 1)  # mono -> fake stereo
        with torch.no_grad():
            sources = apply_model(self._model, wav[None], device="cpu")[0]
        names = self._model.sources  # e.g. ['drums', 'bass', 'other', 'vocals']
        stems: Dict[str, np.ndarray] = {
            name: sources[i].mean(0).numpy() for i, name in enumerate(names)
        }
        return Stems(
            vocals=stems.get("vocals"),
            drums=stems.get("drums"),
            bass=stems.get("bass"),
            other=stems.get("other"),
            backend="demucs",
        )

    def _separate_hpss(self, audio: np.ndarray) -> Stems:
        harmonic, percussive = librosa.effects.hpss(audio.astype(np.float32))
        # crude approximation: harmonic ~ vocals+other tonal content, percussive ~ drums
        return Stems(vocals=harmonic, drums=percussive, bass=None, other=None,
                      backend="hpss_fallback")
