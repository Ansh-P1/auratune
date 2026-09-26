"""
Stand-ins for Part 1 (continuous ambient sensing) and Part 2 (real-time
decision loop + smooth EQ transitions), neither of which exist in this repo
yet -- there's no perception/live_monitor.py, agents/live_loop.py, or
dsp/curve_smoothing.py to import.

Part 5 (this file, pages/3_Live_Mode.py, validation/live_mode_eval.py) is
built against these fakes so the live dashboard isn't blocked waiting on
those branches, per the team brief's own suggested approach. Both fakes
honor the exact hand-off contracts the brief defines, so swapping them out
later is a two-line import change, not a rewrite:

  - FakeLiveMonitor.observe() stands in for
    perception.live_monitor.LiveMonitor's on_change(fn) callback: it only
    reports a change once the (noise_level, content_type) bucket differs
    from the last confirmed one AND stays different for 2 consecutive
    samples.
  - ramp_steps() stands in for dsp.curve_smoothing.CurveRamper: linear
    interpolation from an old TargetCurve to a new one, last step exact.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from dsp.parametric_eq import TargetCurve
from perception.context_classifier import Context

CURVE_FIELDS = ("volume_db", "bass_gain_db", "presence_gain_db", "treble_gain_db")


class FakeLiveMonitor:
    """Cycles through a scripted list of synth_scenario kinds (holding each
    one for a few samples in a row, the way a real room would) and applies
    Part 1's debounce rule to decide when that's a genuine context change."""

    def __init__(self, scenario_cycle: Sequence[str] = ("quiet_podcast", "noisy_music", "home_movie"),
                 repeats_per_scenario: int = 4):
        self.scenario_cycle = list(scenario_cycle)
        self.repeats_per_scenario = max(1, repeats_per_scenario)
        self._cycle_idx = 0
        self._repeat_count = 0
        self._last_bucket = None
        self._pending_bucket = None
        self._pending_count = 0

    def next_scenario_kind(self) -> str:
        """What the (fake) mic would pick up on this sample."""
        kind = self.scenario_cycle[self._cycle_idx]
        self._repeat_count += 1
        if self._repeat_count >= self.repeats_per_scenario:
            self._repeat_count = 0
            self._cycle_idx = (self._cycle_idx + 1) % len(self.scenario_cycle)
        return kind

    def observe(self, ctx: Context) -> bool:
        """Feed in one sample's Context. Returns True iff this sample just
        confirmed a stable, debounced change (2 consecutive samples with a
        bucket that differs from the last confirmed one)."""
        bucket = (ctx.noise_level, ctx.content_type)
        if bucket == self._last_bucket:
            self._pending_bucket = None
            self._pending_count = 0
            return False
        if bucket == self._pending_bucket:
            self._pending_count += 1
        else:
            self._pending_bucket = bucket
            self._pending_count = 1
        if self._pending_count >= 2:
            self._last_bucket = bucket
            self._pending_bucket = None
            self._pending_count = 0
            return True
        return False


def ramp_steps(old: TargetCurve, new: TargetCurve, n_steps: int = 5) -> List[TargetCurve]:
    """Linear interpolation old -> new over n_steps TargetCurves. The last
    step exactly equals `new`."""
    steps = []
    for i in range(1, n_steps + 1):
        frac = i / n_steps
        kwargs = {f: getattr(old, f) + (getattr(new, f) - getattr(old, f)) * frac
                  for f in CURVE_FIELDS}
        steps.append(TargetCurve(name=new.name, **kwargs))
    return steps


def dominant_field_delta(deltas: dict) -> Tuple[str, float]:
    """Pick the field that moved the most, for a one-line event description."""
    field, value = max(deltas.items(), key=lambda kv: abs(kv[1]))
    return field, value


FIELD_LABELS = {
    "volume_db": "volume",
    "bass_gain_db": "bass",
    "presence_gain_db": "presence",
    "treble_gain_db": "treble",
}
