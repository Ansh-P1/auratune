"""
Integrations -- where AuraTune stops *describing* an EQ and actually applies
one to real, playing audio.

Everything else in the repo produces numbers for a human to type into their
own EQ app. A module in here takes the same projected curve and drives a
real system-wide equalizer instead, so an adaptation is audible within
seconds without anyone touching a slider.

Current targets:
  - equalizer_apo: Equalizer APO on Windows (equalizerapo.com), driven by
    writing its plain-text config file, which it reloads by itself.

Every integration follows the repo's graceful-fallback style (see
perception/stem_separation.py, perception/live_capture.py): if the tool
isn't installed, raise a clear, specific error saying how to install it --
never crash the app that imported the module.
"""
