"""Real-time decision loop: room changed -> (rate-limited) re-run pipeline -> glide EQ.

Wiring (once Part 1's perception/live_monitor.py is in the repo):
    monitor = LiveMonitor()
    loop = LiveLoop(monitor, store=store, eq=eq, user_id=user_id,
                     on_curve_step=apply_and_animate)   # Part 3/5 supply this
    loop.start()
    monitor.start()

LiveMonitor.on_change(callback) fires callback(context) -- a single
perception.context_classifier.Context -- once a room/content change has
been stable for 2 consecutive samples (including the very first stable
reading, used to seed the initial curve). This loop wraps that with an
8s cooldown so a burst of change events doesn't hammer run_pipeline, then
glides the EQ to whatever run_pipeline decides instead of snapping.
"""
import threading
import time
from typing import Any, Callable, Optional

from dsp.curve_smoothing import CurveRamper, StepCallback
from dsp.parametric_eq import TargetCurve


def _default_run_pipeline(*args, **kwargs):
    from agents.graph import run_pipeline  # lazy: keeps this module unit-testable
    return run_pipeline(*args, **kwargs)


def _extract_curve(result: Any) -> Optional[TargetCurve]:
    """run_pipeline returns a PipelineState dict with "decided_curve"."""
    if isinstance(result, TargetCurve):
        return result
    if isinstance(result, dict):
        return result.get("decided_curve")
    return getattr(result, "decided_curve", None)


class LiveLoop:
    def __init__(self, monitor, store, eq, user_id: str, on_curve_step: StepCallback,
                 run_pipeline: Optional[Callable] = None,
                 cooldown_s: float = 8.0,
                 ramper: Optional[CurveRamper] = None,
                 initial_curve: Optional[TargetCurve] = None,
                 clock: Callable[[], float] = time.monotonic,
                 background: bool = True,
                 **pipeline_kwargs):
        """pipeline_kwargs: forwarded to run_pipeline on every call
        (equalizer_spec, content_audio, ambient_audio, sample_rate,
        genre_model, noise_model, user_command) -- whatever the caller's
        Streamlit session currently has set.
        """
        self.monitor = monitor
        self.store = store
        self.eq = eq
        self.user_id = user_id
        self.on_curve_step = on_curve_step
        self.run_pipeline = run_pipeline or _default_run_pipeline
        self.pipeline_kwargs = pipeline_kwargs
        self.cooldown_s = cooldown_s
        self.ramper = ramper or CurveRamper(duration_s=1.5, steps=15)
        self.current_curve = initial_curve   # what's currently "playing"
        self._clock = clock
        self._background = background
        self._last_run: Optional[float] = None   # "last run timestamp" guard
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Subscribe to the monitor. Does NOT call monitor.start() itself --
        that's a separate decision (real mic sampling) left to the caller."""
        self.monitor.on_change(self.handle_change)

    def handle_change(self, context) -> bool:
        """LiveMonitor.on_change callback: fires with a single confirmed
        Context. Returns True if the pipeline actually ran (i.e. we weren't
        in the cooldown window)."""
        with self._lock:
            now = self._clock()
            if self._last_run is not None and now - self._last_run < self.cooldown_s:
                return False            # still cooling down: drop this event
            self._last_run = now        # claim the window before the (slow) run

        result = self.run_pipeline(self.store, self.eq, self.user_id, context,
                                    **self.pipeline_kwargs)
        new_curve = _extract_curve(result)
        if new_curve is None:
            return True
        if self._background:
            self._thread = threading.Thread(target=self._glide, args=(new_curve,), daemon=True)
            self._thread.start()
        else:
            self._glide(new_curve)
        return True

    def _glide(self, new_curve: TargetCurve):
        def step(curve, is_final):
            self.current_curve = curve   # next glide starts from wherever we are
            self.on_curve_step(curve, is_final)
        self.ramper.play(self.current_curve, new_curve, step)

    def join(self, timeout=None):
        if self._thread:
            self._thread.join(timeout)
