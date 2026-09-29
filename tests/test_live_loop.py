import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsp.parametric_eq import TargetCurve
from dsp.curve_smoothing import CurveRamper
from agents.live_loop import LiveLoop


class FakeMonitor:
    def on_change(self, cb): self.cb = cb


class FakeContext:
    """Stand-in for perception.context_classifier.Context."""
    def __init__(self, noise_level="quiet"):
        self.noise_level = noise_level


def test_cooldown_and_glide():
    calls, steps, t = [], [], [0.0]

    def fake_pipeline(store, eq, user_id, context, **kw):
        calls.append((store, eq, user_id, context))
        return {"decided_curve": TargetCurve("scenario", -3, -2, 3, 0)}

    mon = FakeMonitor()
    loop = LiveLoop(mon, store="STORE", eq="EQ", user_id="u1",
                    on_curve_step=lambda c, f: steps.append((c, f)),
                    run_pipeline=fake_pipeline, cooldown_s=8,
                    ramper=CurveRamper(steps=5, sleep=lambda s: None),
                    initial_curve=TargetCurve("baseline", 0, 0, 0, 0),
                    clock=lambda: t[0], background=False)
    loop.start()
    assert mon.cb == loop.handle_change   # subscribed with a single-arg callback

    ctx = FakeContext()
    for _ in range(20):                 # burst of on_change events at t=0..~3.8s
        mon.cb(ctx); t[0] += 0.2
    assert len(calls) == 1, calls       # only one pipeline run in the window
    assert calls[0] == ("STORE", "EQ", "u1", ctx)   # store/eq/user_id passed correctly
    assert steps[-1] == (TargetCurve("scenario", -3, -2, 3, 0), True)

    t[0] = 9.0
    mon.cb(ctx)                          # cooldown expired -> runs again
    assert len(calls) == 2
    assert loop.current_curve == TargetCurve("scenario", -3, -2, 3, 0)


if __name__ == "__main__":
    test_cooldown_and_glide(); print("live_loop tests passed")
