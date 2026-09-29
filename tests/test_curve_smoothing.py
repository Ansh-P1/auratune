import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsp.parametric_eq import TargetCurve
from dsp.curve_smoothing import CurveRamper, CURVE_FIELDS


def check(old, new, steps=10):
    r = CurveRamper(steps=steps)
    out = r.ramp(old, new)
    assert len(out) == steps
    for f in CURVE_FIELDS:
        vals = [getattr(old, f)] + [getattr(c, f) for c in out]
        d = getattr(new, f) - getattr(old, f)
        diffs = [b - a for a, b in zip(vals, vals[1:])]
        if d >= 0:
            assert all(x >= -1e-12 for x in diffs), (f, vals)
        else:
            assert all(x <= 1e-12 for x in diffs), (f, vals)
        assert getattr(out[-1], f) == getattr(new, f), f   # exact
    assert out[-1] == new


def test_ramp():
    check(TargetCurve("a", 0, 0, 0, 0), TargetCurve("b", -4.4, -2.5, 3.5, 1.0))   # mixed directions
    check(TargetCurve("a", 2, 3, -1, 0), TargetCurve("a", 2, 3, -1, 0))           # no change
    check(TargetCurve("a", -6, 4, 4, 2), TargetCurve("b", 0, 0, 0, 0), steps=7)   # all down
    out = CurveRamper(steps=4).ramp(TargetCurve("a", 0, 0, 0, 0), TargetCurve("b", 4, 8, 0, -4))
    assert [c.volume_db for c in out] == [1, 2, 3, 4]                            # linear
    assert out[-1].name == "b"                                                   # target's name/bands win


def test_play_emits_final_flag_once_and_paces():
    got, slept = [], []
    r = CurveRamper(duration_s=1.5, steps=5, sleep=slept.append)
    r.play(TargetCurve("a", 0, 0, 0, 0), TargetCurve("b", 1, 1, 1, 1), lambda c, f: got.append((c, f)))
    assert [f for _, f in got] == [False] * 4 + [True]
    assert len(slept) == 4 and abs(sum(slept) - 1.5 * 4 / 5) < 1e-9


if __name__ == "__main__":
    test_ramp(); test_play_emits_final_flag_once_and_paces()
    print("curve_smoothing tests passed")
