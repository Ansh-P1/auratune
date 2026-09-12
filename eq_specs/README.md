# eq_specs/

One JSON file per real-world EQ (a phone EQ app, a streaming-app EQ, a car
head unit). AuraTune snaps its ideal target curve onto whichever one you
pick, so the output is values you can actually dial in.

## Fields

| field | meaning |
|---|---|
| `name` | shown in the app's dropdown |
| `band_freqs_hz` | slider centre frequencies, exactly as the app labels them |
| `gain_min_db` / `gain_max_db` | how far one slider can move |
| `step_db` | slider granularity — `1.0`, `0.5`, `0.1`; `0` = continuous |
| `has_preamp` | does the app have a separate preamp / gain slider |
| `preamp_min_db` / `preamp_max_db` | preamp range (ignored if `has_preamp` is false) |
| `notes` | free text — where the spec came from, caveats |

## Adding one from a screenshot

Send Claude a screenshot of your EQ app and ask it to write the spec here.
It reads the band frequencies, the step (0.1 vs 0.5 vs 1 dB), and the range
off the picture and drops a new `<name>.json` in this folder. It then shows
up in the app's "Your EQ app" dropdown automatically.

You can also build one by hand in the app's **Manual** editor and it will be
saved here.

## Built-in presets

`wavelet_9band` here overrides nothing — the app also ships `iso_10band`,
`spotify_5band`, and `car_3band` presets in `dsp/equalizer_spec.py`.
