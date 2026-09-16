# Track 5: QA, Validation & Suggestions Report

**Author / QA Lead:** Track 5 Lead  
**Date:** September 16, 2026  
**Status:** Completed & Validated  

---

## 1. Executive Summary

This report documents the validation, manual matrix testing, edge-case stress tests, and automated testing gap analysis for **AuraTune**. All five pipeline stages (Live Capture $\to$ Local ML Classifiers $\to$ LangGraph Agent Decision Pipeline $\to$ DSP Equalizer Projection $\to$ Streamlit UI) were thoroughly audited.

### Test Results Summary
* **Full Scenario Matrix:** 48 / 48 permutations passed (3 Contexts × 4 Commands × 4 EQ Specs).
* **EQ Hardware Profiles:** 13 / 13 preset files in `eq_specs/` + custom manual configuration + OCR parsing validated.
* **Adversarial / Stress Cases:** 6 / 6 resilience tests passed without unhandled exceptions.
* **Automated Unit Tests:** 8 / 8 test files in `tests/` passing cleanly.
* **Validation Traces:** Regenerated and verified in `validation/output/`.

---

## 2. Testing Matrices & Execution

### 2.1 Scenario × Command × EQ Device Matrix (Task 1)
| Scenario | Ambient dB | Content Type | Command | Device Profile | Target Adjustment | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| Home Movie Night | -42 dB (quiet) | `movie` | `None` | Continuous | Flat baseline | **PASS** |
| Home Movie Night | -42 dB (quiet) | `movie` | `"make dialogue crystal clear"` | Wavelet 9-Band | +3 dB presence, snapped to 9 bands | **PASS** |
| Noisy Cafe Music | -18 dB (noisy) | `music` | `None` | Spotify 5-Band | +3.5 dB presence, -2.5 dB bass | **PASS** |
| Noisy Cafe Music | -18 dB (noisy) | `music` | `"boost bass a lot"` | Bose Music App | Bass boosted, vocal preserved | **PASS** |
| Commute Podcast | -28 dB (moderate) | `podcast` | `"reduce harsh treble"` | Continuous | -2.0 dB treble shelf | **PASS** |
| Commute Podcast | -28 dB (moderate) | `podcast` | `None` | Sony Headphones | Clamped to Sony 5-band grid | **PASS** |

### 2.2 EQ Device Ingestion Pathways (Task 2)
1. **Preset JSON Files (`eq_specs/*.json`):**
   * Validated all 13 specs: `wavelet_9band`, `iso_10band`, `spotify_5band`, `car_3band`, `apple_music_10band`, `bose_music_app`, `google_pixel_buds_pro_5band`, `nothing_x_advanced_8band`, `oneplus_heymelody_5band`, `samsung_soundalive_9band`, `sennheiser_smart_control`, `sony_headphones_app`, `soundcore_app_8band`.
   * Verified that frequency values are monotonic and slider bounds are strictly respected.
2. **Manual Spec Entry:**
   * Custom arbitrary 4-band studio spec (`[80Hz, 500Hz, 2500Hz, 12000Hz]`, `[-15dB, +15dB]`, `step=0.5dB`) properly quantized and protected against clipping via preamp attenuation.
3. **Screenshot OCR Parser (`perception/eq_app_reader.py`):**
   * Verified JSON extraction and schema validation logic with graceful fallback when API keys are absent.

### 2.3 Edge-Case & Stress Testing (Task 4)
* **500-Word Prompt Flooding:** Handled without memory or buffer issues.
* **Empty / Whitespace / Unicode Characters (`"   "`, `"🎛️🔊🔥 12345 !@#$%^&*"`):** Fallback parser returns zero deltas cleanly.
* **Extreme Gain Requests (`"make it 1000dB louder"`):** Clamped within acoustic boundaries $[-12\text{ dB}, +12\text{ dB}]$.
* **Missing API Keys:** Full pipeline switches to keyword parsing and template explanation generator without raising errors.

---

## 3. Automated Test Gaps & Manual QA Checklist (Task 3)

### Automated Test Gaps Identified
1. **Hardware-Level Audio Streams:** Unit tests cannot test live PortAudio / microphone inputs in headless CI environments.
2. **Live Vision / LLM APIs:** Claude/Gemini API calls are skipped/mocked in CI without live API credentials.
3. **UI Layout / Visual Theme Contrast:** Visual stacking and theme switching require manual inspection.

### Manual QA Checklist for Releases
- [ ] **Microphone Permission Fallback:** Deny browser/system microphone permission and confirm user-friendly warning banner displays without traceback.
- [ ] **Live 10-Second Recording:** Run `streamlit run app.py`, record live audio, and verify classified noise level.
- [ ] **LLM vs Template Badge:** Verify badge indicator shows "Claude" when `ANTHROPIC_API_KEY` is present and "Template" when absent.
- [ ] **Dark / Light Theme Legibility:** Ensure all Plotly curve lines, data tables, and badges are readable in both themes.

---

## 4. Ranked Suggestions & Improvements (Task 6)

| Rank | Severity | Area | Description & Recommendation |
| :---: | :---: | :---: | :--- |
| **1** | **High** | **CI Workflow** | **Fixed in this PR:** `.github/workflows/ci.yml` previously only ran 5 of the 8 test suites. Added `test_genre_classifier.py`, `test_noise_classifier.py`, and `test_profile_store.py` along with `scikit-learn` / `joblib` dependencies. |
| **2** | **Medium** | **Equalizer Spec** | `spotify_5band` in `dsp/equalizer_spec.py` defines 6 frequencies (`[60, 150, 400, 1000, 2400, 15000]`). Recommend renaming key or noting 6 discrete sliders in documentation. |
| **3** | **Medium** | **UI Feedback** | Real-time 10s microphone recording in `app.py` has no live visual progress bar. Adding an `st.progress` timer improves user experience during capture. |
| **4** | **Low** | **Model Cards** | Add direct dashboard notes when ML weight files (`ml/models/`) are ungenerated, explaining that the system is operating in rule-based fallback mode until `ml/train.py` is executed. |
