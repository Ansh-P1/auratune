"""
Unit and simulation tests for Part 4: Personalization / Preference Learning.
"""
import sys
import tempfile
import time
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.db import ProfileStore
from dsp.parametric_eq import ParametricEQ, TargetCurve
from perception.context_classifier import Context
from learning.preference_model import PreferenceModel
from agents.graph import run_pipeline


def _fresh_store() -> ProfileStore:
    tmp = Path(tempfile.gettempdir()) / f"auratune_pref_test_{time.time_ns()}.json"
    if tmp.exists():
        tmp.unlink()
    return ProfileStore(mongo_uri=None, local_path=tmp)


class TestPreferenceStore(unittest.TestCase):
    def test_log_and_get_feedback(self):
        store = _fresh_store()
        store.log_feedback(
            user_id="user_a",
            context_bucket="noisy_music",
            agent_curve={"bass_gain_db": 2.0},
            user_correction_delta={"bass_gain_db": -1.5},
        )
        store.log_feedback(
            user_id="user_a",
            context_bucket="quiet_podcast",
            agent_curve={"presence_gain_db": 3.0},
            user_correction_delta={"presence_gain_db": 1.0},
        )

        all_fb = store.get_feedback("user_a")
        self.assertEqual(len(all_fb), 2)

        noisy_fb = store.get_feedback("user_a", context_bucket="noisy_music")
        self.assertEqual(len(noisy_fb), 1)
        self.assertEqual(noisy_fb[0]["user_correction_delta"]["bass_gain_db"], -1.5)

        quiet_fb = store.get_feedback("user_a", context_bucket="quiet_podcast")
        self.assertEqual(len(quiet_fb), 1)
        self.assertEqual(quiet_fb[0]["user_correction_delta"]["presence_gain_db"], 1.0)


class TestPreferenceModel(unittest.TestCase):
    def test_empty_feedback_returns_zero(self):
        store = _fresh_store()
        model = PreferenceModel(store=store)
        bias, conf = model.get_bias("new_user", "noisy_music")
        self.assertEqual(conf, 0.0)
        self.assertEqual(bias["bass_gain_db"], 0.0)
        self.assertEqual(bias["presence_gain_db"], 0.0)

    def test_cold_start_confidence_gating(self):
        store = _fresh_store()
        model = PreferenceModel(store=store, confidence_scale=3)

        # 1 correction of -3.0 dB
        model.record_feedback(
            user_id="u1",
            context_bucket="noisy_music",
            agent_curve={"bass_gain_db": 2.0},
            user_correction_delta={"bass_gain_db": -3.0},
        )
        bias, conf = model.get_bias("u1", "noisy_music")
        self.assertEqual(conf, 0.33)
        # Scaled by confidence 0.33: -3.0 * 0.33 = ~ -1.0 dB
        self.assertAlmostEqual(bias["bass_gain_db"], -0.99, places=1)

        # 3 corrections reach full confidence
        model.record_feedback(
            user_id="u1",
            context_bucket="noisy_music",
            agent_curve={"bass_gain_db": 2.0},
            user_correction_delta={"bass_gain_db": -3.0},
        )
        model.record_feedback(
            user_id="u1",
            context_bucket="noisy_music",
            agent_curve={"bass_gain_db": 2.0},
            user_correction_delta={"bass_gain_db": -3.0},
        )
        bias, conf = model.get_bias("u1", "noisy_music")
        self.assertEqual(conf, 1.0)
        self.assertAlmostEqual(bias["bass_gain_db"], -3.0, places=1)

    def test_simulation_convergence(self):
        """Simulate applying the same context 10 times with synthetic user -2dB corrections."""
        store = _fresh_store()
        model = PreferenceModel(store=store)

        for _ in range(10):
            model.record_feedback(
                user_id="u_sim",
                context_bucket="cafe_noise",
                agent_curve={"bass_gain_db": 3.0},
                user_correction_delta={"bass_gain_db": -2.0},
            )

        bias, conf = model.get_bias("u_sim", "cafe_noise")
        self.assertEqual(conf, 1.0)
        self.assertAlmostEqual(bias["bass_gain_db"], -2.0, places=1)

    def test_ear_safe_clamp(self):
        store = _fresh_store()
        model = PreferenceModel(store=store, max_bias_db=4.0)

        for _ in range(5):
            model.record_feedback(
                user_id="u_loud",
                context_bucket="subway",
                agent_curve={"bass_gain_db": 0.0},
                user_correction_delta={"bass_gain_db": 10.0},
            )

        bias, conf = model.get_bias("u_loud", "subway")
        self.assertEqual(bias["bass_gain_db"], 4.0)

    def test_explanation_clause(self):
        model = PreferenceModel()
        bias = {"bass_gain_db": -1.5, "presence_gain_db": 0.0, "treble_gain_db": 0.0, "volume_db": 0.0}
        clause = model.get_explanation_clause(bias, confidence=1.0, context_bucket="noisy music")
        self.assertIn("-1.5 dB bass", clause)
        self.assertIn("past preferences", clause)

    def test_diff_curves(self):
        c1 = TargetCurve(name="c1", volume_db=0.0, bass_gain_db=2.0, presence_gain_db=1.0, treble_gain_db=0.0)
        c2 = TargetCurve(name="c2", volume_db=0.0, bass_gain_db=0.5, presence_gain_db=1.0, treble_gain_db=1.5)
        diff = PreferenceModel.diff_curves(c1, c2)
        self.assertEqual(diff, {"bass_gain_db": -1.5, "treble_gain_db": 1.5})

    def test_get_learning_summary(self):
        store = _fresh_store()
        model = PreferenceModel(store=store)

        # Empty summary
        summary_empty = model.get_learning_summary("user_empty")
        self.assertEqual(summary_empty["total_feedback_events"], 0)
        self.assertEqual(summary_empty["active_buckets"], [])

        # Record 2 feedbacks in different buckets
        model.record_feedback("user_sum", "noisy_cafe", {"bass_gain_db": 2.0}, {"bass_gain_db": -1.0})
        model.record_feedback("user_sum", "quiet_room", {"treble_gain_db": 0.0}, {"treble_gain_db": 1.5})

        summary = model.get_learning_summary("user_sum")
        self.assertEqual(summary["total_feedback_events"], 2)
        self.assertIn("noisy_cafe", summary["active_buckets"])
        self.assertIn("quiet_room", summary["active_buckets"])
        self.assertIn("Personalized: 2 adjustments", summary["summary_text"])



class TestPreferencePipelineIntegration(unittest.TestCase):
    def test_pipeline_without_and_with_feedback(self):
        store = _fresh_store()
        eq = ParametricEQ()
        context = Context(noise_level="noisy", content_type="music", ambient_rms_db=-18.0, content_features={})

        # 1. Baseline user without feedback
        res_baseline = run_pipeline(store=store, eq=eq, user_id="user_baseline", context=context)
        self.assertEqual(res_baseline["preference_confidence"], 0.0)
        self.assertEqual(res_baseline["preference_deltas"]["bass_gain_db"], 0.0)
        base_bass = res_baseline["decided_curve"].bass_gain_db

        # 2. User with 3 feedback records (-2.0 dB bass correction)
        model = PreferenceModel(store=store)
        for _ in range(3):
            model.record_feedback(
                user_id="user_learned",
                context_bucket="noisy_music",
                agent_curve=res_baseline["decided_curve"],
                user_correction_delta={"bass_gain_db": -2.0},
            )

        # 3. Pipeline run for user with feedback
        res_learned = run_pipeline(store=store, eq=eq, user_id="user_learned", context=context)
        self.assertEqual(res_learned["preference_confidence"], 1.0)
        self.assertEqual(res_learned["preference_deltas"]["bass_gain_db"], -2.0)
        self.assertAlmostEqual(res_learned["decided_curve"].bass_gain_db, base_bass - 2.0, places=1)
        self.assertIn("past preferences", res_learned["explanation"])



if __name__ == "__main__":
    unittest.main()
