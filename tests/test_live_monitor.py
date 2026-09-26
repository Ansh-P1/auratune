import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from perception.live_monitor import LiveMonitor
from perception.synth_scenarios import synth_scenario, SCENARIO_CONTENT_TYPE

SR = 44100


def _scripted_source(sequence):
    """Returns a clip-source fn that yields synth_scenario clips from
    `sequence` in order, one per call, matching set_clip_source()'s
    expected (ambient, content, content_type_hint) shape.
    """
    it = iter(sequence)

    def _next():
        kind = next(it)
        ambient, content = synth_scenario(kind, SR)
        return ambient, content, SCENARIO_CONTENT_TYPE[kind]

    return _next


def test_on_change_fires_exactly_twice_for_stable_transitions():
    # quiet_podcast -> noisy_music -> quiet_podcast, each held for 2
    # consecutive samples so debounce (2-in-a-row) confirms each change.
    sequence = (
        ["quiet_podcast"] * 2
        + ["noisy_music"] * 2
        + ["quiet_podcast"] * 2
    )

    monitor = LiveMonitor(debounce_samples=2)
    fired = []
    monitor.on_change(fired.append)
    monitor.set_clip_source(_scripted_source(sequence))

    for _ in sequence:
        monitor._sample_once(SR)

    assert len(fired) == 2, f"expected 2 confirmed changes, got {len(fired)}"
    assert fired[0].noise_level == "noisy" and fired[0].content_type == "music"
    assert fired[1].noise_level == "quiet" and fired[1].content_type == "podcast"


def test_single_stray_sample_does_not_fire():
    # baseline (silent) -> one stray noisy_music sample (not 2 in a row,
    # must NOT fire) -> back to quiet_podcast (nothing changed) -> a real,
    # stable transition to noisy_music (must fire exactly once).
    sequence = (
        ["quiet_podcast"] * 2   # baseline, silent
        + ["noisy_music"] * 1   # stray -- must not trigger on_change
        + ["quiet_podcast"] * 2  # back to normal, still no change
        + ["noisy_music"] * 2   # genuine, stable transition
    )

    monitor = LiveMonitor(debounce_samples=2)
    fired = []
    monitor.on_change(fired.append)
    monitor.set_clip_source(_scripted_source(sequence))

    for _ in sequence:
        monitor._sample_once(SR)

    assert len(fired) == 1, f"expected exactly 1 confirmed change, got {len(fired)}"
    assert fired[0].noise_level == "noisy" and fired[0].content_type == "music"


def test_start_stop_runs_cleanly_on_a_background_thread():
    sequence = ["quiet_podcast"] * 50  # plenty of samples if the loop over-runs
    monitor = LiveMonitor(debounce_samples=2)
    fired = []
    monitor.on_change(fired.append)
    monitor.set_clip_source(_scripted_source(sequence))

    monitor.start(sample_rate=SR, interval_sec=0.01)
    import time
    time.sleep(0.2)
    monitor.stop()

    # All 50 samples are the same scenario, so only the (silent) baseline
    # is ever established -- on_change should never fire, and the loop
    # should have exited cleanly (no exception propagated, thread joined).
    assert len(fired) == 0
    assert monitor._thread is None


if __name__ == "__main__":
    test_on_change_fires_exactly_twice_for_stable_transitions()
    test_single_stray_sample_does_not_fire()
    test_start_stop_runs_cleanly_on_a_background_thread()
    print("All live_monitor tests passed.")
