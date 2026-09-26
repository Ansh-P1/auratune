import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dsp.parametric_eq import TargetCurve
from perception.context_classifier import Context
from validation.live_mode_fakes import FakeLiveMonitor, ramp_steps, dominant_field_delta


def _ctx(noise_level="quiet", content_type="podcast"):
    return Context(noise_level=noise_level, content_type=content_type,
                   ambient_rms_db=-50.0, content_features={})


def test_debounce_ignores_a_single_stray_sample():
    monitor = FakeLiveMonitor()
    assert monitor.observe(_ctx("quiet", "podcast")) is False
    # one stray sample that reverts immediately shouldn't confirm anything
    assert monitor.observe(_ctx("noisy", "podcast")) is False
    assert monitor.observe(_ctx("quiet", "podcast")) is False


def test_debounce_confirms_on_second_consecutive_sample():
    monitor = FakeLiveMonitor()
    assert monitor.observe(_ctx("quiet", "podcast")) is False
    assert monitor.observe(_ctx("noisy", "music")) is False
    assert monitor.observe(_ctx("noisy", "music")) is True
    # already-confirmed bucket shouldn't re-fire
    assert monitor.observe(_ctx("noisy", "music")) is False


def test_scripted_scenario_confirms_start_plus_each_transition():
    # quiet_podcast -> noisy_music -> quiet_podcast, held for 3 samples each.
    # The very first bucket also needs 2 consecutive samples to confirm (an
    # untouched monitor has no prior bucket to compare against), so this is
    # 3 confirmations: initial state + 2 genuine transitions -- one on the
    # 2nd sample of each held block, never on the 1st or 3rd.
    monitor = FakeLiveMonitor()
    script = (["quiet"] * 3 + ["noisy"] * 3 + ["quiet"] * 3)
    fires = [monitor.observe(_ctx(level, "podcast")) for level in script]
    assert fires == [False, True, False, False, True, False, False, True, False], fires


def test_next_scenario_kind_cycles_after_n_repeats():
    monitor = FakeLiveMonitor(scenario_cycle=("a", "b", "c"), repeats_per_scenario=2)
    seen = [monitor.next_scenario_kind() for _ in range(6)]
    assert seen == ["a", "a", "b", "b", "c", "c"]


def test_ramp_steps_moves_monotonically_and_ends_exact():
    old = TargetCurve("old", volume_db=0.0, bass_gain_db=0.0, presence_gain_db=0.0, treble_gain_db=0.0)
    new = TargetCurve("new", volume_db=1.0, bass_gain_db=-4.0, presence_gain_db=2.0, treble_gain_db=0.0)
    steps = ramp_steps(old, new, n_steps=5)
    assert len(steps) == 5

    prev = old
    for step in steps:
        assert step.bass_gain_db <= prev.bass_gain_db  # decreasing toward -4
        assert step.presence_gain_db >= prev.presence_gain_db  # increasing toward 2
        prev = step

    last = steps[-1]
    assert abs(last.volume_db - new.volume_db) < 1e-9
    assert abs(last.bass_gain_db - new.bass_gain_db) < 1e-9
    assert abs(last.presence_gain_db - new.presence_gain_db) < 1e-9
    assert abs(last.treble_gain_db - new.treble_gain_db) < 1e-9


def test_dominant_field_delta_picks_the_largest_move():
    field, value = dominant_field_delta({"bass_gain_db": 1.0, "presence_gain_db": -3.5, "treble_gain_db": 0.5})
    assert field == "presence_gain_db"
    assert value == -3.5


if __name__ == "__main__":
    test_debounce_ignores_a_single_stray_sample()
    test_debounce_confirms_on_second_consecutive_sample()
    test_scripted_scenario_confirms_start_plus_each_transition()
    test_next_scenario_kind_cycles_after_n_repeats()
    test_ramp_steps_moves_monotonically_and_ends_exact()
    test_dominant_field_delta_picks_the_largest_move()
    print("All Live Mode fakes tests passed.")
