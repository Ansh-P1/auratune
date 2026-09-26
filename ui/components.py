"""Reusable AuraTune UI pieces — hero, stat pills, fader rack, timeline, etc."""
from __future__ import annotations
import html as _html
import streamlit as st
from .theme import tokens, is_dark


def _e(x) -> str:
    return _html.escape(str(x))


def hero(subtitle: str = "Reads your room, reads what's playing, and hands you "
                         "the exact slider values for the EQ app you already own.") -> None:
    st.markdown(
        f'<div class="at-hero">'
        f'<h1>Aura<span class="at-acc">Tune</span></h1>'
        f'<div class="at-sub">{_e(subtitle)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def eyebrow(label: str) -> None:
    """Replaces emoji-prefixed st.subheader calls."""
    st.markdown(f'<div class="at-eyebrow">{_e(label)}</div>', unsafe_allow_html=True)


def stats(items: list[tuple[str, str]], hot: int | None = None,
          mono: set[int] | None = None) -> None:
    mono = mono or set()
    cells = []
    for i, (k, v) in enumerate(items):
        cls = "at-stat hot" if i == hot else "at-stat"
        vcls = "v num" if i in mono else "v"
        cells.append(f'<div class="{cls}"><div class="k">{_e(k)}</div>'
                     f'<div class="{vcls}">{_e(v)}</div></div>')
    st.markdown(f'<div class="at-stats">{"".join(cells)}</div>', unsafe_allow_html=True)


def meter(label: str, value: float) -> None:
    pct = max(0.0, min(1.0, float(value))) * 100
    st.markdown(
        f'<div class="at-stat"><div class="k">{_e(label)}</div>'
        f'<div class="v num">{pct:.0f}%</div>'
        f'<div class="at-meter"><i style="width:{pct:.1f}%"></i></div></div>',
        unsafe_allow_html=True,
    )


def empty_state(title: str = "Nothing adapted yet",
                body: str = "Pick a scenario on the left, tell AuraTune which EQ "
                            "app you use, then click Run adaptation.") -> None:
    delays = (0, .09, .18, .27, .36, .45, .54, .45, .36, .27, .18, .09, 0)
    bars = "".join(f'<i style="animation-delay:{d}s"></i>' for d in delays)
    st.markdown(
        f'<div class="at-empty">'
        f'<div class="at-wave">{bars}</div>'
        f'<h4>{_e(title)}</h4><p>{_e(body)}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def timeline(history: list[dict]) -> None:
    if not history:
        st.caption("No adaptations logged yet.")
        return
    rows = []
    for h in reversed(history):
        head = f"{h.get('content_type', '?')} · {h.get('noise_level', '?')}"
        rows.append(f'<div class="at-tl-item"><div class="t">{_e(head)}</div>'
                    f'<div class="d">{_e(h.get("explanation", ""))}</div></div>')
    st.markdown(f'<div class="at-tl">{"".join(rows)}</div>', unsafe_allow_html=True)


def fader_rack(bands, gain_min: float = -12.0, gain_max: float = 12.0,
               clipped: set | None = None) -> None:
    """
    EQ fader rack — the visual heart of the app. Replaces `st.table` for
    `proj.as_table_rows()`; `bands` is `proj.bands` (a list of BandSetting
    with .freq_hz / .set_gain_db).
    """
    clipped = clipped or set()
    span = max(1e-9, gain_max - gain_min)
    zero_pct = (gain_max / span) * 100

    def _hz(f: float) -> str:
        return (f"{f/1000:.1f}k").replace(".0k", "k") if f >= 1000 else f"{f:.0f}"

    cells = []
    for b in bands:
        g = float(b.set_gain_db)
        knob = (gain_max - g) / span * 100
        top = min(knob, zero_pct)
        height = abs(knob - zero_pct)
        cls = "at-fader clipped" if float(b.freq_hz) in clipped else "at-fader"
        cells.append(
            f'<div class="{cls}">'
            f'<div class="track">'
            f'<div class="zero" style="top:{zero_pct:.2f}%"></div>'
            f'<div class="fill" style="top:{top:.2f}%;height:{max(height,0):.2f}%"></div>'
            f'<div class="knob" style="top:{knob:.2f}%"></div>'
            f'</div>'
            f'<div class="val">{g:+.1f}</div>'
            f'<div class="hz">{_hz(float(b.freq_hz))}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="at-rack">{"".join(cells)}</div>', unsafe_allow_html=True)


def device_icon(kind: str = "generic", size: int = 40) -> str:
    """Generic line-art icon, themed — never a real product photo/logo."""
    t = tokens(is_dark())
    paths = {
        "earbuds":
            '<circle cx="8" cy="9" r="3"/><path d="M8 12v6"/>'
            '<circle cx="16" cy="9" r="3"/><path d="M16 12v6"/>',
        "overear":
            '<path d="M4 13v-1a8 8 0 0 1 16 0v1"/>'
            '<rect x="2.5" y="13" width="4" height="7" rx="1.5"/>'
            '<rect x="17.5" y="13" width="4" height="7" rx="1.5"/>',
        "generic":
            '<line x1="5" y1="4" x2="5" y2="20"/><circle cx="5" cy="9" r="1.7" fill="currentColor"/>'
            '<line x1="12" y1="4" x2="12" y2="20"/><circle cx="12" cy="15" r="1.7" fill="currentColor"/>'
            '<line x1="19" y1="4" x2="19" y2="20"/><circle cx="19" cy="11" r="1.7" fill="currentColor"/>',
    }
    c = t["accent"]
    p = paths.get(kind, paths["generic"])
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{c}" stroke-width="1.5" stroke-linecap="round" '
            f'stroke-linejoin="round">{p}</svg>')
