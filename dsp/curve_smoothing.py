"""CurveRamper: glide from one TargetCurve to another instead of snapping.

Interpolation is linear per field:  old + (new - old) * (step / total_steps).
The last step is always *exactly* the target (no float drift).
"""
import time
from dataclasses import fields, replace
from typing import Callable, List, Optional

from dsp.parametric_eq import TargetCurve

# Fields that get interpolated (spec: volume/bass/presence/treble).
CURVE_FIELDS = ("volume_db", "bass_gain_db", "presence_gain_db", "treble_gain_db")

# on_curve_step(curve, is_final)
StepCallback = Callable[[TargetCurve, bool], None]


class CurveRamper:
    def __init__(self, duration_s: float = 1.5, steps: int = 15, sleep=time.sleep):
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if duration_s < 0:
            raise ValueError("duration_s must be >= 0")
        self.duration_s = duration_s
        self.steps = steps
        self._sleep = sleep

    @staticmethod
    def _numeric_fields(curve: TargetCurve):
        names = {f.name for f in fields(curve)}
        return [n for n in CURVE_FIELDS if n in names]

    def ramp(self, old: TargetCurve, new: TargetCurve) -> List[TargetCurve]:
        """N intermediate curves; steps[-1] == new exactly. Excludes `old` itself."""
        names = self._numeric_fields(new)
        out: List[TargetCurve] = []
        for i in range(1, self.steps + 1):
            if i == self.steps:
                out.append(replace(new))  # exact target, no float error
                continue
            t = i / self.steps
            vals = {n: getattr(old, n) + (getattr(new, n) - getattr(old, n)) * t
                    for n in names}
            out.append(replace(new, **vals))
        return out

    def play(self, old: Optional[TargetCurve], new: TargetCurve,
             on_curve_step: StepCallback) -> None:
        """Emit each step via on_curve_step(curve, is_final), paced over duration_s.
        old=None (nothing applied yet) -> just emit `new` once as final."""
        if old is None:
            on_curve_step(replace(new), True)
            return
        steps = self.ramp(old, new)
        delay = self.duration_s / self.steps
        for i, c in enumerate(steps):
            is_final = i == len(steps) - 1
            on_curve_step(c, is_final)
            if not is_final and delay > 0:
                self._sleep(delay)
