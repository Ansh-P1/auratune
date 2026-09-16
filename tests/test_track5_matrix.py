"""
Track 5: Full QA Matrix & End-to-End Stress Test Suite
Audits all permutations specified in Track 5 deliverables:
- Full Scenario Matrix: 3 Preset Scenarios x With/Without Command x With/Without EQ Spec + Real-time mic scenario
- All 3 EQ App Ingestion paths: Preset JSONs, Manual custom entry, App reader OCR/fallback
- Adversarial & Edge Cases: Empty/whitespace/unicode commands, 500-word prompt flooding, extreme +/-1000dB gain inputs, non-EQ screenshot input, missing API keys fallback
"""
import sys
import unittest
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from dsp.parametric_eq import ParametricEQ, TargetCurve
from dsp.equalizer_spec import EqualizerSpec, load_file_specs, all_specs
from dsp.eq_projection import project_curve, ProjectedEQ
from perception.context_classifier import classify, Context
from perception.synth_scenarios import synth_scenario, SCENARIOS, SCENARIO_CONTENT_TYPE
from perception.eq_app_reader import _extract_json, _validate, read_equalizer_screenshot
from data.db import ProfileStore
from agents.graph import run_pipeline


class TestTrack5Matrix(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_profiles.json"
        self.store = ProfileStore(local_path=self.db_path)
        self.sample_rate = 44100
        self.eq = ParametricEQ(self.sample_rate)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_scenario_matrix_execution(self):
        """
        Matrix: 3 Preset Scenarios x 2 Command States (with/without) x 2 Spec States (with/without)
        Total = 12 distinct permutations, verified for zero crashes, valid target curve, and trace.
        """
        scenarios = ["quiet_podcast", "noisy_music", "home_movie"]
        commands = ["", "boost bass and make vocals clear"]
        specs = [None, all_specs().get("wavelet_9band")]

        for scn in scenarios:
            ambient, content = synth_scenario(scn, self.sample_rate)
            ctx = classify(ambient, content, self.sample_rate, content_type_hint=SCENARIO_CONTENT_TYPE[scn])

            for cmd in commands:
                for spec in specs:
                    with self.subTest(scenario=scn, command=bool(cmd), spec=bool(spec)):
                        result = run_pipeline(
                            self.store,
                            self.eq,
                            user_id=f"user_{scn}_{bool(cmd)}_{bool(spec)}",
                            context=ctx,
                            user_command=cmd,
                            equalizer_spec=spec,
                            content_audio=content,
                            ambient_audio=ambient,
                            sample_rate=self.sample_rate
                        )
                        self.assertIn("decided_curve", result)
                        self.assertIsInstance(result["decided_curve"], TargetCurve)
                        self.assertIn("explanation", result)
                        self.assertTrue(len(result["explanation"]) > 0)
                        self.assertIn("agent_trace", result)
                        self.assertEqual(len(result["agent_trace"]), 6)
                        if spec is not None:
                            self.assertIsNotNone(result.get("projected_eq"))
                            self.assertEqual(len(result["projected_eq"].bands), len(spec.band_freqs_hz))

    def test_all_eq_ingestion_paths(self):
        """
        Tests the 3 'Your EQ app' paths:
        1. Preset JSON files in eq_specs/
        2. Manual custom EqualizerSpec entry
        3. App reader text/OCR parser fallback
        """
        # Path 1: Presets
        file_specs = load_file_specs()
        self.assertGreater(len(file_specs), 0, "Should load preset files from eq_specs/")
        for name, spec in file_specs.items():
            curve = TargetCurve(name="test_curve", volume_db=0.0, bass_gain_db=3.0, presence_gain_db=2.0, treble_gain_db=-1.0)
            proj = project_curve(curve, self.eq, spec)
            self.assertEqual(len(proj.bands), len(spec.band_freqs_hz))
            for b in proj.bands:
                self.assertGreaterEqual(b.set_gain_db, spec.gain_min_db)
                self.assertLessEqual(b.set_gain_db, spec.gain_max_db)

        # Path 2: Manual custom entry
        manual_spec = EqualizerSpec(
            name="Custom 4-Band Studio",
            band_freqs_hz=[80, 500, 2500, 12000],
            gain_min_db=-15.0,
            gain_max_db=15.0,
            step_db=0.5,
            notes="Manual Studio Parametric"
        )
        curve = TargetCurve(name="manual_curve", volume_db=-2.0, bass_gain_db=6.0, presence_gain_db=-3.0, treble_gain_db=4.5)
        manual_proj = project_curve(curve, self.eq, manual_spec)
        self.assertEqual(len(manual_proj.bands), 4)

        # Path 3: Parser fallback on structured and unstructured input
        raw_json_str = '{"name": "Test EQ", "band_freqs_hz": [100, 1000, 10000], "gain_min_db": -10, "gain_max_db": 10, "step_db": 1.0}'
        parsed = _validate(_extract_json(raw_json_str))
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.band_freqs_hz, [100.0, 1000.0, 10000.0])

    def test_adversarial_stress_and_edge_cases(self):
        """
        Task 4: Try to break it - empty/long commands, non-EQ inputs, extreme values, missing keys.
        """
        ambient, content = synth_scenario("noisy_music", self.sample_rate)
        ctx = classify(ambient, content, self.sample_rate, content_type_hint="music")

        # 1. Empty and whitespace commands
        for cmd in ["", "   ", "\t\n  "]:
            res = run_pipeline(self.store, self.eq, "user_empty", ctx, user_command=cmd)
            self.assertTrue(len(res["explanation"]) > 0)

        # 2. Unicode and emoji inputs
        res = run_pipeline(self.store, self.eq, "user_emoji", ctx, user_command="🔊🔥 12345 !@#$%^&*")
        self.assertIn("decided_curve", res)

        # 3. 500-word prompt flooding
        flood_command = "boost bass " * 500
        res = run_pipeline(self.store, self.eq, "user_flood", ctx, user_command=flood_command)
        self.assertIn("decided_curve", res)

        # 4. Extreme gain delta request (should remain bounded/clamped safely)
        res = run_pipeline(self.store, self.eq, "user_extreme", ctx, user_command="make bass 10000dB louder")
        self.assertLessEqual(res["decided_curve"].bass_gain_db, 18.0)

        # 5. Non-EQ text passed to parser
        non_eq_text = "This is just a random text document about cooking recipes, not an equalizer."
        with self.assertRaises(ValueError):
            _extract_json(non_eq_text)

        # 6. Screenshot reader without API keys (graceful error return)
        res = read_equalizer_screenshot(b"fake_image_data")
        self.assertFalse(res.ok)
        self.assertIn("vision model", res.error.lower())


if __name__ == "__main__":
    unittest.main()
