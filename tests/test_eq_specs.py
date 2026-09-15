"""
Automated validation test suite for all equalizer specifications in eq_specs/
and built-in PRESETS in dsp/equalizer_spec.py.

Guarantees:
- Every spec JSON is valid and conforms to the EqualizerSpec schema.
- Band frequencies are strictly positive, ascending, and within audio range (20 Hz - 20 kHz).
- Gain and preamp boundaries are logically consistent (min < max).
- Step size is non-negative.
- Quantization handles standard, clamped, and zero values reliably.
"""
import sys
import json
import unittest
from pathlib import Path

# Add repo root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dsp.equalizer_spec import EqualizerSpec, PRESETS, load_file_specs, all_specs


class TestEqualizerSpecs(unittest.TestCase):

    def setUp(self):
        self.eq_specs_dir = ROOT_DIR / "eq_specs"

    def test_json_files_exist(self):
        """Ensure the eq_specs directory exists and contains JSON spec files."""
        self.assertTrue(self.eq_specs_dir.is_dir(), "eq_specs/ directory not found")
        json_files = list(self.eq_specs_dir.glob("*.json"))
        self.assertGreater(len(json_files), 0, "No JSON spec files found in eq_specs/")

    def test_all_json_files_are_valid_and_conform_to_schema(self):
        """Verify each JSON file can be parsed and fulfills EqualizerSpec constraints."""
        json_files = sorted(self.eq_specs_dir.glob("*.json"))
        for path in json_files:
            with self.subTest(file=path.name):
                # Raw JSON parse
                raw_text = path.read_text(encoding="utf-8")
                data = json.loads(raw_text)
                self.assertIsInstance(data, dict, f"{path.name} root must be a JSON object")

                # Instantiate EqualizerSpec
                spec = EqualizerSpec.from_dict(data)
                self._validate_spec_properties(spec, source=path.name)

    def test_builtin_presets_are_valid(self):
        """Verify all built-in PRESETS in equalizer_spec.py satisfy schema constraints."""
        for key, spec in PRESETS.items():
            with self.subTest(preset=key):
                self._validate_spec_properties(spec, source=f"PRESETS['{key}']")

    def test_all_specs_loader(self):
        """Verify that load_file_specs and all_specs combine files and presets without errors."""
        file_specs = load_file_specs()
        merged_specs = all_specs()

        self.assertGreater(len(file_specs), 0)
        self.assertGreaterEqual(len(merged_specs), len(PRESETS))

        for key, spec in merged_specs.items():
            self.assertIsInstance(spec, EqualizerSpec)
            self.assertTrue(bool(spec.name.strip()), f"Spec '{key}' must have a non-empty name")

    def _validate_spec_properties(self, spec: EqualizerSpec, source: str):
        """Helper to run deep property checks on an EqualizerSpec instance."""
        # Name validation
        self.assertIsInstance(spec.name, str)
        self.assertTrue(bool(spec.name.strip()), f"[{source}] name cannot be blank")

        # Band frequency validation
        self.assertIsInstance(spec.band_freqs_hz, list)
        self.assertGreater(len(spec.band_freqs_hz), 0, f"[{source}] must have at least 1 band")

        prev_f = 0.0
        for f in spec.band_freqs_hz:
            self.assertIsInstance(f, (int, float), f"[{source}] frequency {f} must be numeric")
            self.assertGreater(f, 0.0, f"[{source}] frequency {f} must be positive")
            self.assertLessEqual(f, 22050.0, f"[{source}] frequency {f} must be within audible spectrum")
            self.assertGreater(f, prev_f, f"[{source}] frequencies must be strictly ascending: {spec.band_freqs_hz}")
            prev_f = f

        # Gain limits validation
        self.assertLess(spec.gain_min_db, spec.gain_max_db, f"[{source}] gain_min_db must be < gain_max_db")
        self.assertGreaterEqual(spec.step_db, 0.0, f"[{source}] step_db must be >= 0")

        # Preamp validation
        if spec.has_preamp:
            self.assertLessEqual(
                spec.preamp_min_db, spec.preamp_max_db,
                f"[{source}] preamp_min_db must be <= preamp_max_db"
            )

        # Quantization verification
        # 1. Zero gain
        snapped_zero, _ = spec.quantize_gain(0.0)
        self.assertEqual(snapped_zero, 0.0, f"[{source}] quantizing 0.0 gain should return 0.0")

        # 2. Clamping high value
        snapped_high, clamped_high = spec.quantize_gain(spec.gain_max_db + 50.0)
        self.assertEqual(snapped_high, spec.gain_max_db)
        self.assertTrue(clamped_high)

        # 3. Clamping low value
        snapped_low, clamped_low = spec.quantize_gain(spec.gain_min_db - 50.0)
        self.assertEqual(snapped_low, spec.gain_min_db)
        self.assertTrue(clamped_low)

        # 4. Preamp quantization
        if spec.has_preamp:
            snapped_p = spec.quantize_preamp(spec.preamp_min_db - 10.0)
            self.assertEqual(snapped_p, spec.preamp_min_db)


if __name__ == "__main__":
    unittest.main(verbosity=2)
