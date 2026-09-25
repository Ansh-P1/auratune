# Applying AuraTune's EQ live, with Equalizer APO

Everywhere else, AuraTune hands you numbers to type into your phone's EQ app.
This path skips the typing: AuraTune writes Equalizer APO's config file and
Equalizer APO re-reads it within a moment, so an adaptation lands on whatever
is playing — system-wide, no restart, nothing to click.

**Windows only.** Equalizer APO hooks Windows' audio engine (APO = Audio
Processing Object); there is no macOS or Linux build. See
[Testing without Windows](#testing-without-windows) below for what you can
still run on a Mac.

## One-time setup

1. **Install** Equalizer APO from <https://equalizerapo.com> (free, open
   source).
2. **Attach it to your output device.** The installer opens a device list —
   tick the device you actually listen through (your headphones, or
   "Speakers"). Get this wrong and everything below writes a config that
   nothing reads. You can re-run the Configurator later from the Start menu.
3. **Reboot.** Windows only loads a new APO into the audio pipeline at
   device init.
4. **Check the config file exists** at
   `C:\Program Files\EqualizerAPO\config\config.txt`. That's the default
   install path, and the one AuraTune writes by default.

## Pointing AuraTune at it

- **Default:** nothing to configure — `C:\Program Files\EqualizerAPO\config\config.txt`.
- **Installed elsewhere,** or you want AuraTune to write a config you
  `Include:` from the main one? Set the path explicitly:

  ```
  set AURATUNE_APO_CONFIG=D:\EqualizerAPO\config\auratune.txt
  ```

  (In the repo's `.env`, that's `AURATUNE_APO_CONFIG=...`.)

Then pick **"Equalizer APO (Windows, system-wide, live)"** in the app's
**Your EQ app** dropdown, the same way you'd pick a phone EQ.

### Permissions

`C:\Program Files\` is usually not writable by a normal user process. If
writes fail you'll get a message saying so — either run the app elevated, or
point `AURATUNE_APO_CONFIG` at a writable file and `Include:` it from
Equalizer APO's own config:

```
Include: D:\EqualizerAPO\config\auratune.txt
```

## What gets written

One projected band per `Filter` line, plus a preamp that cancels the largest
boost so nothing clips:

```
# Written by AuraTune -- do not edit by hand while Auto mode is on.
# Equalizer APO reloads this file automatically when it changes.
Preamp: -3.6 dB
Filter 1: ON PK Fc 31.25 Hz Gain -2.5 dB Q 1.41
Filter 2: ON PK Fc 62.5 Hz Gain -2.2 dB Q 1.41
...
```

- `PK` is a peaking filter; the three-band `TargetCurve` path also emits
  `LSC`/`HSC` shelves.
- A band at 0 dB is written `OFF` rather than dropped, so filter numbers stay
  stable between writes and a diff is readable mid-demo.
- Writes are **atomic** (temp file in the same directory, then
  `os.replace()`). Equalizer APO watches the file, so a plain overwrite could
  be caught half-written and loaded as a broken config.

## How it's called

```python
from dsp.eq_projection import project_curve
projected = project_curve(curve, eq, spec, apply_live=True)
projected.live_applied   # True if it reached Equalizer APO
projected.live_detail    # what happened, or why it didn't
```

`apply_live` is **off by default**: projection stays a pure function that
computes slider values, and only writes to your audio device when a caller
explicitly asks. Part 2's live loop calls it once per smoothing step, so a
ramp arrives as a series of small writes and the boost glides in over a
second or two rather than snapping.

If Equalizer APO isn't installed, nothing raises: `live_applied` stays False,
`live_detail` says why, and the slider numbers still show on screen — the
same graceful degradation as the mic and Demucs paths.

## Testing without Windows

The renderer and writer are plain text and plain files, so everything except
hearing it works anywhere:

```bash
AURATUNE_APO_CONFIG=/tmp/apo-config.txt python tests/test_equalizer_apo.py
```

That's how this runs in CI. What genuinely needs a Windows machine with the
driver installed is the last mile: play music, trigger a context change, and
confirm you can *hear* the bass move within a couple of seconds.
