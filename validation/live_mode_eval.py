"""
Part 5 evaluation: proves the live pipeline actually works with numbers, not
just a live demo that could get lucky.

Feeds a scripted sequence of perception.synth_scenarios clips -- quiet ->
noisy -> quiet again -- through the real classify() + agents.graph.run_pipeline
pipeline, using the FakeLiveMonitor / ramp_steps stand-ins from
validation/live_mode_fakes.py for the sensing debounce and curve smoothing
that Parts 1 & 2 haven't landed yet (see that file's docstring). Reports:

  - reaction latency: how many ticks (and simulated seconds) after the room
    actually changed did the debounced monitor confirm it
  - unnecessary flip-flops: confirmed changes that don't correspond to a
    real scripted transition
  - transition smoothness: average/max per-step size of the interpolated
    glide, and whether every glide's last step exactly hits the target curve

No microphone needed -- synth scenarios only -- so this is reproducible for
grading.

Run with: python3 validation/live_mode_eval.py
Outputs to: validation/output/live_mode_eval_report.md
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dsp.parametric_eq import ParametricEQ
from perception.context_classifier import classify
from perception.synth_scenarios import synth_scenario, SCENARIO_CONTENT_TYPE
from data.db import ProfileStore
from agents.graph import run_pipeline
from validation.live_mode_fakes import FakeLiveMonitor, ramp_steps, CURVE_FIELDS

SR = 44100
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# "quiet -> noisy cafe -> quiet again" -- noisy_music stands in for a noisy
# cafe, the closest of the 3 canned scenarios (no literal cafe clip exists).
# Each scenario is held for 4 ticks so the 2-consecutive-sample debounce has
# something real to confirm, matching how pages/3_Live_Mode.py paces it.
SCRIPT = (["quiet_podcast"] * 4 + ["noisy_music"] * 4 + ["quiet_podcast"] * 4)
TICK_INTERVAL_SEC = 5.0  # matches Part 1's suggested default sampling interval
N_RAMP_STEPS = 5


def _expected_confirmation_points(script):
    """Ticks where a correctly-behaving debounced monitor should confirm
    something: index 0 for the very first stable reading (an untouched
    monitor has no prior bucket to compare against, so even the starting
    state needs 2 consecutive samples), plus every genuine scripted
    transition after that."""
    return [0] + [i for i in range(1, len(script)) if script[i] != script[i - 1]]


def main():
    store = ProfileStore(local_path=OUT_DIR / "live_mode_eval_profiles.json")
    eq = ParametricEQ(SR)
    monitor = FakeLiveMonitor()

    transitions = _expected_confirmation_points(SCRIPT)
    confirmations = []          # tick indices where observe() returned True
    smoothness_reports = []     # one dict per confirmed change
    current_curve = None

    print(f"Running {len(SCRIPT)} scripted ticks: {' -> '.join(SCRIPT)}\n")

    for i, kind in enumerate(SCRIPT):
        ambient, content = synth_scenario(kind, SR)
        ctx = classify(ambient, content, SR, content_type_hint=SCENARIO_CONTENT_TYPE[kind])
        confirmed = monitor.observe(ctx)
        status = "CONFIRMED CHANGE" if confirmed else "  (debounced)"
        print(f"  tick {i:2d} [{kind:<13s}] noise={ctx.noise_level:<8s} "
              f"content={ctx.content_type:<8s} {status}")

        if not confirmed:
            continue
        confirmations.append(i)

        result = run_pipeline(store, eq, user_id="live_mode_eval", context=ctx, user_command="",
                              content_audio=content, ambient_audio=ambient, sample_rate=SR)
        new_curve = result["decided_curve"]
        old_curve = current_curve or result["baseline_curve"]
        steps = ramp_steps(old_curve, new_curve, N_RAMP_STEPS)

        step_sizes = []
        prev = old_curve
        for step in steps:
            step_sizes.append(sum(abs(getattr(step, f) - getattr(prev, f)) for f in CURVE_FIELDS))
            prev = step
        exact_final = all(abs(getattr(steps[-1], f) - getattr(new_curve, f)) < 1e-9 for f in CURVE_FIELDS)
        smoothness_reports.append({
            "tick": i, "avg_step_db": sum(step_sizes) / len(step_sizes),
            "max_step_db": max(step_sizes), "exact_final": exact_final,
        })
        current_curve = new_curve

    # -- reaction latency: nearest confirmation at/after each ground-truth transition --
    latencies = []
    for t in transitions:
        after = [conf for conf in confirmations if conf >= t]
        if after:
            latencies.append(after[0] - t)
    avg_latency_ticks = sum(latencies) / len(latencies) if latencies else float("nan")

    # -- flip-flops: confirmations that aren't the nearest-after match for any transition --
    matched = set()
    for t in transitions:
        after = [conf for conf in confirmations if conf >= t and conf not in matched]
        if after:
            matched.add(after[0])
    flip_flops = len(confirmations) - len(matched)

    lines = ["# Live Mode evaluation report\n",
             f"Scripted sequence: `{' -> '.join(SCRIPT)}` ({len(SCRIPT)} ticks, "
             f"{TICK_INTERVAL_SEC}s/tick simulated)\n",
             f"- Expected confirmations (startup + real transitions): {len(transitions)} (at ticks {transitions})",
             f"- Confirmed changes: {len(confirmations)} (at ticks {confirmations})",
             f"- **Reaction latency**: avg {avg_latency_ticks:.1f} ticks "
             f"({avg_latency_ticks * TICK_INTERVAL_SEC:.1f}s) after the room actually changed",
             f"- **Unnecessary flip-flops**: {flip_flops}",
             "", "## Transition smoothness\n"]
    for r in smoothness_reports:
        lines.append(f"- tick {r['tick']}: avg step {r['avg_step_db']:.2f} dB/step, "
                     f"max step {r['max_step_db']:.2f} dB, "
                     f"final step exact: {'yes' if r['exact_final'] else 'NO'}")

    report = "\n".join(lines)
    (OUT_DIR / "live_mode_eval_report.md").write_text(report)
    print("\n" + report)
    print(f"\nWrote report to {OUT_DIR / 'live_mode_eval_report.md'}")

    assert flip_flops == 0, "unexpected flip-flop(s) in a fully scripted, deterministic run"
    assert all(r["exact_final"] for r in smoothness_reports), "a glide's last step didn't hit the target exactly"
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
