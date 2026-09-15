"""
Generates the 3 validation traces called out in the project's own roadmap:
  1. Quiet room + podcast   -> minimal compensation, curve near stored target
  2. Noisy environment + music -> vocal/presence boost, bass pulled back
  3. Home + movie          -> stored movie target curve, content-type switch

For each: synthesizes ambient+content audio, runs perception -> agents -> DSP,
saves a before/after frequency-response plot, and writes a markdown report
with the agent-produced explanation and numeric deltas.

Run with: python3 validation/generate_traces.py
Outputs to: validation/output/
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dsp.parametric_eq import ParametricEQ
from perception.context_classifier import classify
from perception.synth_scenarios import synth_scenario, SCENARIOS as SCENARIO_LABELS, SCENARIO_CONTENT_TYPE
from data.db import ProfileStore
from agents.graph import run_pipeline

SR = 44100
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

SCENARIOS = list(SCENARIO_LABELS.items())


def main():
    store = ProfileStore(local_path=OUT_DIR / "validation_profiles.json")
    eq = ParametricEQ(SR)
    report_lines = ["# AuraTune Validation Traces\n"]

    for key, label in SCENARIOS:
        ambient, content = synth_scenario(key, SR)
        ctx = classify(ambient, content, SR, content_type_hint=SCENARIO_CONTENT_TYPE[key])
        result = run_pipeline(store, eq, user_id="validation_user", context=ctx, user_command="",
                              content_audio=content, sample_rate=SR)

        freqs, mag_before = eq.frequency_response(result["baseline_curve"])
        _, mag_after = eq.frequency_response(result["decided_curve"])

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogx(freqs, mag_before, "--", color="gray", label="Stored baseline")
        ax.semilogx(freqs, mag_after, "-", color="#2E86AB", linewidth=2.5, label="Live adapted")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Gain (dB)")
        ax.set_title(label)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig_path = OUT_DIR / f"{key}.png"
        fig.savefig(fig_path, dpi=140)
        plt.close(fig)

        deltas = eq.delta(result["baseline_curve"], result["decided_curve"])
        report_lines.append(f"## {label}")
        report_lines.append(f"- Detected: noise=`{ctx.noise_level}`, content=`{ctx.content_type}`, "
                             f"ambient={ctx.ambient_rms_db} dB")
        report_lines.append(f"- Deltas: {deltas}")
        if result.get("genre_bucket"):
            report_lines.append(f"- Genre (local ML, {result.get('genre_model_used')}): "
                                 f"`{result['genre_bucket']}` ({result.get('genre_confidence', 0) * 100:.0f}% confidence)")
        report_lines.append(f"- Explanation: \"{result['explanation']}\"")
        report_lines.append(f"- Curve plot: `{fig_path.name}`\n")

        print(f"[{key}] {ctx.noise_level}/{ctx.content_type} -> {result['explanation']}")

    (OUT_DIR / "report.md").write_text("\n".join(report_lines))
    print(f"\nWrote report + plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
