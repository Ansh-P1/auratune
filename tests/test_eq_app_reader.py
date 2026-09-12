import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from perception import eq_app_reader
from perception.eq_app_reader import _extract_json, _validate, read_equalizer_screenshot


def test_extract_json_from_fenced_reply():
    text = 'Sure:\n```json\n{"a": 1, "b": [2, 3]}\n```\n'
    assert _extract_json(text) == {"a": 1, "b": [2, 3]}


def test_extract_json_from_bare_reply():
    assert _extract_json('{"x": 5} trailing junk') == {"x": 5}


def test_validate_sorts_and_coerces_bands():
    spec = _validate({
        "name": "Test EQ",
        "band_freqs_hz": ["1000", 60, "16000", 250],
        "gain_min_db": -6, "gain_max_db": 6, "step_db": 0.1, "has_preamp": False,
    })
    assert spec.band_freqs_hz == [60.0, 250.0, 1000.0, 16000.0]
    assert spec.step_db == 0.1


def test_validate_repairs_bad_range_and_step():
    spec = _validate({
        "band_freqs_hz": [100, 1000],
        "gain_min_db": 10, "gain_max_db": -10,   # inverted
        "step_db": 99,                            # nonsense
    })
    assert spec.gain_min_db == -12.0 and spec.gain_max_db == 12.0
    assert spec.step_db == 1.0


def test_validate_rejects_empty_bands():
    try:
        _validate({"band_freqs_hz": []})
    except ValueError:
        return
    raise AssertionError("expected ValueError for zero bands")


def test_read_screenshot_without_key_is_graceful():
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.pop(var, None)
    res = read_equalizer_screenshot(b"fake-bytes", "image/png")
    assert res.spec is None
    assert res.error and "Gemini" in res.error


def test_gemini_backend_routes_and_parses(monkeypatch_free=True):
    calls = {}

    def fake_gemini(image_bytes, media_type, api_key):
        calls["key"] = api_key
        return ('{"name": "Gemini EQ", "band_freqs_hz": [60, 230, 910, 3600, 14000], '
                '"gain_min_db": -12, "gain_max_db": 12, "step_db": 1.0, '
                '"has_preamp": false, "notes": ""}')

    orig = eq_app_reader._read_with_gemini
    eq_app_reader._read_with_gemini = fake_gemini
    try:
        res = read_equalizer_screenshot(b"img", "image/jpeg", gemini_key="AIza-test")
    finally:
        eq_app_reader._read_with_gemini = orig

    assert res.ok and res.backend == "gemini"
    assert calls["key"] == "AIza-test"
    assert res.spec.band_freqs_hz == [60.0, 230.0, 910.0, 3600.0, 14000.0]


if __name__ == "__main__":
    test_extract_json_from_fenced_reply()
    test_extract_json_from_bare_reply()
    test_validate_sorts_and_coerces_bands()
    test_validate_repairs_bad_range_and_step()
    test_validate_rejects_empty_bands()
    test_read_screenshot_without_key_is_graceful()
    test_gemini_backend_routes_and_parses()
    print("All EQ app-reader tests passed.")
