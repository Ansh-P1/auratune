"""
Continuous ambient sensing.

Wraps perception/live_capture.py's mic recording in a background loop so
the app can notice the room changing (quiet -> noisy -> quiet) on its own,
without a user clicking "Run adaptation". Runs in a Python thread so it
never blocks Streamlit's main loop.

Debounce rule: a context change is only reported once the new
(noise_level, content_type) pair has shown up for 2 consecutive samples
in a row -- a single stray cough or door-slam shouldn't trigger a
re-adaptation.

Coordinate with whoever owns the Streamlit page on how results get back
into st.session_state: on_change()'s callback fires from a background
thread, and session_state isn't thread-safe to write from there directly.
Push into a queue.Queue from the callback and poll it from the main
thread instead.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Optional

from perception.context_classifier import classify, Context
from perception.live_capture import record_ambient, MicUnavailableError

# fn(ambient, content, content_type_hint) -> used to substitute a scripted
# clip for a real mic recording in tests. See set_clip_source().
ClipSource = Callable[[], tuple]


class LiveMonitor:
    """Continuously samples ambient audio and fires on_change() callbacks
    when the room/content genuinely (and stably) changes.
    """

    def __init__(self, history_size: int = 3, debounce_samples: int = 2):
        self._callbacks: list[Callable[[Context], None]] = []
        self._history: deque[Context] = deque(maxlen=history_size)

        self._debounce_samples = debounce_samples
        self._last_confirmed: Optional[Context] = None
        self._pending_key: Optional[tuple] = None
        self._pending_streak = 0

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Test hook: when set, _sample_once() pulls from this instead of
        # the real mic. See set_clip_source().
        self._clip_source: Optional[ClipSource] = None

    def on_change(self, callback: Callable[[Context], None]) -> None:
        """Register a function to be called with the new Context once a
        stable change is confirmed (2 consecutive matching samples).
        """
        self._callbacks.append(callback)

    def set_clip_source(self, fn: Optional[ClipSource]) -> None:
        """Test-only hook. `fn` must return (ambient, content,
        content_type_hint) -- see perception/synth_scenarios.py. Pass None
        to go back to real mic capture via perception/live_capture.py.
        """
        self._clip_source = fn

    def _get_clip(self, sample_rate: int) -> tuple:
        """Returns (ambient, content, content_type_hint). Real mic capture
        has no content signal (see live_capture.py's docstring -- capturing
        "currently playing" audio is out of scope there), so we fall back
        to classifying ambient-as-content with no hint, matching app.py's
        existing `content if content is not None else ambient` pattern.
        """
        if self._clip_source is not None:
            return self._clip_source()
        ambient = record_ambient(duration_sec=2.0, sr=sample_rate)
        return ambient, ambient, None

    def _sample_once(self, sample_rate: int) -> Optional[Context]:
        """Record/fetch one clip, classify it, and run it through the
        debounce state machine. Returns the new Context if this sample
        triggered a confirmed change, else None. Raises MicUnavailableError
        if real mic capture fails -- callers decide whether to fall back.
        """
        ambient, content, hint = self._get_clip(sample_rate)
        ctx = classify(ambient, content, sample_rate, content_type_hint=hint)
        self._history.append(ctx)
        return self._debounce_and_maybe_fire(ctx)

    def _debounce_and_maybe_fire(self, ctx: Context) -> Optional[Context]:
        key = (ctx.noise_level, ctx.content_type)
        last_key = (
            (self._last_confirmed.noise_level, self._last_confirmed.content_type)
            if self._last_confirmed is not None
            else None
        )

        if key == last_key:
            # Back to (or still at) the confirmed state -- nothing changed,
            # and any in-progress candidate change is no longer stable.
            self._pending_key = None
            self._pending_streak = 0
            return None

        if key == self._pending_key:
            self._pending_streak += 1
        else:
            self._pending_key = key
            self._pending_streak = 1

        if self._pending_streak >= self._debounce_samples:
            is_initial_baseline = self._last_confirmed is None
            self._last_confirmed = ctx
            self._pending_key = None
            self._pending_streak = 0
            if is_initial_baseline:
                # Establishing the starting state isn't a "change" -- there
                # was nothing to change from. Only fire for genuine
                # transitions after a baseline exists.
                return None
            for callback in self._callbacks:
                callback(ctx)
            return ctx

        return None

    def start(self, sample_rate: int = 44100, interval_sec: float = 5.0) -> None:
        """Begin sampling every interval_sec seconds on a background
        thread. Safe to call again after stop(); a no-op if already
        running.
        """
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()

        def _loop():
            while not self._stop_event.is_set():
                try:
                    self._sample_once(sample_rate)
                except MicUnavailableError:
                    # No mic available (e.g. no hardware, permission
                    # denied). Stop quietly rather than spinning forever
                    # or crashing the app -- the caller (Streamlit page)
                    # is responsible for surfacing this and/or falling
                    # back to synth scenarios.
                    break
                # wait() instead of time.sleep() so stop() can interrupt
                # immediately instead of waiting out the full interval.
                self._stop_event.wait(interval_sec)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the background loop to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
