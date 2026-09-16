# Track 5: Testing, QA & Suggestions Report

**Author / QA Lead:** Track 5 Lead (Sarvyagya)  
**Date:** September 16, 2026  
**Status:** Completed & Validated  
**Scope:** Full End-to-End Auditing across Input (Capture) $\to$ Classification $\to$ Agent Pipeline $\to$ DSP Equalizer Projection $\to$ UI/UX  

---

## 1. Executive Summary

Track 5 validates the complete AuraTune signal chain after Tracks 1–4 have landed their respective components. We executed:
1. **The Full Manual Matrix** across all 3 built-in scenarios $\times$ with/without user command $\times$ with/without target EQ hardware profile + real-time microphone capture mode.
2. **All 3 EQ Ingestion Pathways** (Preset JSON files, custom manual entry, and screenshot OCR reading).
3. **Adversarial & Stress Testing** (empty/whitespace/unicode commands, 500-word prompt flooding, extreme +/-1000 dB gain requests, invalid image parsing, missing API keys fallback).
4. **Automated Test Coverage & Gap Analysis** (built `tests/test_track5_matrix.py` to automate end-to-end permutation testing; provided manual QA checklists for non-automatable hardware/API boundaries).
5. **Gating & CI Verification** (updated `.github/workflows/ci.yml` with full test suite across all 11 test modules).
6. **Ranked Suggestions & Bugs List** with concrete repro steps and severity rankings.

---

## 2. Testing Matrices & Execution

### 2.1 Scenario $\times$ Command $\times$ EQ Device Matrix (Task 1)

| # | Scenario | Ambient dB / Content | User Command | EQ Profile | Expected & Observed Target Adjustment | Status |
| :---: | :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | Quiet Podcast | -42 dB, `podcast` | None | Flat Baseline (Continuous) | 0.0 dB across all bands (baseline unchanged) | **PASS** |
| 2 | Quiet Podcast | -42 dB, `podcast` | `"make dialogue crystal clear"` | Flat Baseline (Continuous) | +3.0 dB presence peak centered at 2.5 kHz | **PASS** |
| 3 | Quiet Podcast | -42 dB, `podcast` | None | Wavelet 9-Band | Quantized to Wavelet grid: 0.0 dB across 9 bands | **PASS** |
| 4 | Quiet Podcast | -42 dB, `podcast` | `"make dialogue crystal clear"` | Wavelet 9-Band | +3.0 dB snapped to 2 kHz & 4 kHz sliders | **PASS** |
| 5 | Noisy Music | -18 dB, `music` | None | Continuous | +3.5 dB presence, -2.5 dB bass (compensates noise floor) | **PASS** |
| 6 | Noisy Music | -18 dB, `music` | `"boost bass a lot"` | Continuous | Bass boost +6.0 dB, presence +3.5 dB | **PASS** |
| 7 | Noisy Music | -18 dB, `music` | None | Spotify 6-Band | Snapped: 60Hz: -2.0dB, 1kHz: +2.0dB, 2.4kHz: +3.0dB | **PASS** |
| 8 | Noisy Music | -18 dB, `music` | `"boost bass a lot"` | Spotify 6-Band | 60Hz & 150Hz boosted +4.0dB, 2.4kHz +3.0dB | **PASS** |
| 9 | Home Movie | -28 dB, `movie` | None | Continuous | +1.5 dB presence, -1.0 dB bass | **PASS** |
| 10 | Home Movie | -28 dB, `movie` | `"reduce harsh treble"` | Continuous | -2.0 dB treble shelf at 8 kHz | **PASS** |
| 11 | Home Movie | -28 dB, `movie` | None | Sony Headphones (5-band) | 400Hz: 0dB, 1kHz: +1dB, 2.5kHz: +2dB, 6.3kHz: 0dB | **PASS** |
| 12 | Home Movie | -28 dB, `movie` | `"reduce harsh treble"` | Sony Headphones (5-band) | 6.3kHz slider reduced by -2.0 dB, 16kHz reduced | **PASS** |
| 13 | Real-Time Mic | Live capture | None | Continuous | Dynamic RMS calculation, classified noise level | **PASS** |
| 14 | Real-Time Mic | Live capture | `"extra punchy bass"` | Bose Music App | Preamp lowered to -3.0 dB to prevent digital clipping | **PASS** |

---

### 2.2 EQ Device Ingestion Pathways (Task 2)

We validated all 3 pathways by which an equalizer specification is ingested into AuraTune:

1. **Preset JSON Files (`eq_specs/*.json`):**
   * Validated all 13 bundled device presets:
     - `apple_music_10band.json`
     - `bose_music_app.json`
     - `car_3band.json`
     - `google_pixel_buds_pro_5band.json`
     - `iso_10band.json`
     - `nothing_x_advanced_8band.json`
     - `oneplus_heymelody_5band.json`
     - `samsung_soundalive_9band.json`
     - `sennheiser_smart_control.json`
     - `sony_headphones_app.json`
     - `soundcore_app_8band.json`
     - `spotify_5band.json`
     - `wavelet_9band.json`
   * Ensured frequency bands are strictly ascending, gain ranges $(g_{\min}, g_{\max})$ are non-inverted, and quantization steps match real hardware.

2. **Manual Custom Spec Entry:**
   * Tested custom studio 4-band spec: `[80Hz, 500Hz, 2500Hz, 12000Hz]`, range $[-15\text{ dB}, +15\text{ dB}]$, step $0.5\text{ dB}$.
   * Verified that target curve interpolation projects accurately to the custom frequencies and computes appropriate preamp attenuation when boosts exceed $0\text{ dB}$.

3. **Screenshot OCR Parser (`perception/eq_app_reader.py`):**
   * Verified Gemini (`gemini-2.0-flash`) and Claude (`claude-sonnet-5`) JSON extraction prompt pipelines.
   * Tested fallback mechanism: when API keys are absent or network fails, a structured error object is returned without throwing unhandled exceptions.

---

### 2.3 Adversarial Stress & Edge-Case Testing (Task 4)

| Test Case | Input / Condition | Observed Pipeline Behavior | Status |
| :--- | :--- | :--- | :---: |
| **Empty Command** | `""` | Bypasses command parsing, generates base environment adaptation | **PASS** |
| **Whitespace / Padding** | `"   \t\n  "` | Treated as empty command cleanly | **PASS** |
| **Unicode & Emojis** | `"🔊🔥 12345 !@#$%^&*"` | Parser handles gracefully, returns zero deltas without crashing | **PASS** |
| **Prompt Flooding** | `"boost bass " * 500` (500 words) | Parsed within memory limits without stack overflow | **PASS** |
| **Extreme Gain Request** | `"make bass 10000dB louder"` | Clamped safely by DSP safety ceiling ($+12\text{ dB}$ to $+18\text{ dB}$) | **PASS** |
| **Non-EQ Document** | Random text / non-EQ image | Schema validation rejects malformed payload and reports user error | **PASS** |
| **Missing API Keys** | Unset `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` | Gracefully falls back to rule-based parser and template explainer | **PASS** |
| **Rapid Scenario Switching** | 10 rapid scenario toggles | State dictionary and ProfileStore update synchronously without race condition | **PASS** |

---

## 3. Automated Test Gaps & Manual QA Checklist (Task 3)

### Automated Test Gaps
1. **Live Microphone Stream:** Headless CI systems lack physical soundcards / microphones (PortAudio throws `OSError` / `MicUnavailableError`).
2. **Vision LLM API Rate Limits:** Real OCR API queries require external network access and secret keys not present in pull request CI runners.
3. **Browser Audio Rendering & Theme Contrast:** Visual layout aesthetics and audio playback in Streamlit cannot be fully verified via unit tests.

### Manual QA Checklist for Pre-Release Verification
- [ ] **Live Mic Graceful Fallback:** Deny microphone permission in browser or run on headless server; verify warning banner appears instead of stack trace.
- [ ] **10-Second Mic Recording:** Click "Real-time (10s mic capture)", observe room noise classification, and check confidence level.
- [ ] **Claude vs. Template Badge:** Confirm UI badge displays "Claude" when `ANTHROPIC_API_KEY` is provided, and "Template" when absent.
- [ ] **Dark & Light Mode Contrast:** Toggle Streamlit theme settings; ensure Plotly curve traces, slider text, and data tables maintain high contrast.
- [ ] **Slider Values Export:** Click "Download Slider Settings (.txt)" and confirm file content matches projected slider values.

---

## 4. Ranked Suggestions & Bug Reports (Task 6)

| Rank | Severity | Component | Issue & Recommendation | Repro Steps |
| :---: | :---: | :---: | :--- | :--- |
| **1** | **High** | **CI Workflow** | **Resolved:** CI was originally running only a subset of test files. Updated `.github/workflows/ci.yml` to run all 11 test suites including `test_track5_matrix.py`, `test_genre_classifier.py`, `test_noise_classifier.py`, `test_profile_store.py`, and `test_agent_trace.py`. | Push PR and inspect GitHub Actions job steps. |
| **2** | **Medium** | **Equalizer Spec** | `spotify_5band.json` and preset in `dsp/equalizer_spec.py` define 6 frequencies (`[60, 150, 400, 1000, 2400, 15000] Hz`). Rename spec or document that Spotify on mobile features 6 discrete slider bands. | Inspect `dsp/equalizer_spec.py` line for Spotify preset. |
| **3** | **Medium** | **UI Feedback** | Real-time 10s audio capture in `app.py` blocks without a dynamic progress counter. Adding an animated progress bar or countdown timer will improve UX responsiveness. | Run `streamlit run app.py` and click "Start 10s Capture". |
| **4** | **Low** | **Model Cards** | When pre-trained ML weights (`ml/models/`) have not yet been trained locally, the app displays fallback messages. Adding an in-app "Train Models" helper or pre-bundling minimal sample weights would streamline onboarding. | Start app in clean environment without running `train.py`. |

---

## 5. Verification & Test Suite Status

All 11 automated test suites pass with zero errors:
```bash
=== Running tests/test_agent_trace.py ===       [PASS]
=== Running tests/test_classifier.py ===        [PASS]
=== Running tests/test_dsp.py ===               [PASS]
=== Running tests/test_eq_app_reader.py ===     [PASS]
=== Running tests/test_eq_projection.py ===     [PASS]
=== Running tests/test_eq_specs.py ===          [PASS]
=== Running tests/test_genre_classifier.py ===  [PASS]
=== Running tests/test_noise_classifier.py ===  [PASS]
=== Running tests/test_profile_store.py ===     [PASS]
=== Running tests/test_track5_matrix.py ===     [PASS]
=== Validation Traces (generate_traces.py) ===  [PASS]
```
