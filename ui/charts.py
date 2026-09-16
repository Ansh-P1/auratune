"""Themed Plotly charts. All colour comes from ui.theme tokens."""
from __future__ import annotations
import plotly.graph_objects as go
import streamlit as st
from .theme import tokens, is_dark

_FONT = dict(family="Instrument Sans, sans-serif", size=12)
CONFIG = {"displayModeBar": False, "scrollZoom": False, "responsive": True}


def _rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return ",".join(str(int(h[i:i + 2], 16)) for i in (0, 2, 4))


def _layout(t: dict, height: int) -> dict:
    return dict(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(**_FONT, color=t["ink_soft"]),
        hoverlabel=dict(bgcolor=t["card"], bordercolor=t["border"],
                        font=dict(**_FONT, color=t["ink"])),
    )


def eq_curve(freqs_before, mag_before, freqs_after, mag_after,
             proj=None, height: int = 350) -> None:
    """Hero chart: baseline vs adapted, with the delta shaded between them."""
    t = tokens(is_dark())
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=freqs_before, y=mag_before, name="Stored baseline",
        line=dict(color=t["border_strong"], width=1.5, dash="dot"),
        hovertemplate="%{y:+.1f} dB<extra>baseline</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=freqs_after, y=mag_after, name="Live adapted",
        line=dict(color=t["accent"], width=2.6, shape="spline"),
        fill="tonexty", fillcolor=f'rgba({_rgb(t["accent"])},0.11)',
        hovertemplate="%{y:+.1f} dB<extra>adapted</extra>",
    ))

    if proj is not None and getattr(proj, "bands", None):
        fig.add_trace(go.Scatter(
            x=[b.freq_hz for b in proj.bands],
            y=[b.set_gain_db for b in proj.bands],
            name=f"{proj.spec_name} sliders", mode="markers",
            marker=dict(size=9, color=t["card"],
                        line=dict(color=t["warm"], width=2)),
            hovertemplate="%{x:.0f} Hz → set %{y:+.1f} dB<extra></extra>",
        ))

    fig.add_hline(y=0, line=dict(color=t["border_strong"], width=1))
    fig.update_xaxes(type="log", title=None, gridcolor=t["grid"], zeroline=False,
                     linecolor=t["border"], tickfont=dict(size=10),
                     ticksuffix=" Hz")
    fig.update_yaxes(title=None, gridcolor=t["grid"], zeroline=False,
                     linecolor=t["border"], tickfont=dict(size=10),
                     ticksuffix=" dB")
    fig.update_layout(
        **_layout(t, height), hovermode="x unified",
        legend=dict(orientation="h", y=1.15, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    st.plotly_chart(fig, use_container_width=True, config=CONFIG)


def probability_bars(probs: dict, accent: str = "accent", top_n: int = 6) -> None:
    """
    Horizontal sorted bars, winner highlighted. Replaces st.bar_chart, whose
    rotated multi-word category labels used to clip in the narrow column.
    """
    t = tokens(is_dark())
    if not probs:
        return
    items = sorted(probs.items(), key=lambda kv: kv[1])[-top_n:]
    labels = [k for k, _ in items]
    vals = [float(v) for _, v in items]
    hot = t[accent]
    colors = [hot if i == len(vals) - 1 else t["border_strong"]
              for i in range(len(vals))]

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=colors),
        text=[f"{v*100:.0f}%" for v in vals], textposition="outside",
        textfont=dict(size=10, color=t["muted"]),
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    fig.update_xaxes(visible=False, range=[0, max(vals) * 1.25])
    fig.update_yaxes(showgrid=False, linecolor="rgba(0,0,0,0)",
                     tickfont=dict(size=11, color=t["ink_soft"]))
    fig.update_layout(**_layout(t, 26 * len(vals) + 26),
                      bargap=0.42, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=CONFIG)


def delta_bars(deltas: dict, height: int = 130) -> None:
    """Small diverging bar strip used inside the debug/raw-deltas expander."""
    t = tokens(is_dark())
    if not deltas:
        return
    ks = [str(k) for k in deltas.keys()]
    try:
        vs = [float(v) for v in deltas.values()]
    except (TypeError, ValueError):
        return
    colors = [t["accent"] if v >= 0 else t["warm"] for v in vs]
    fig = go.Figure(go.Bar(
        x=ks, y=vs, marker=dict(color=colors),
        hovertemplate="%{x}: %{y:+.2f} dB<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=t["border_strong"], width=1))
    fig.update_xaxes(showgrid=False, tickfont=dict(size=9, color=t["muted"]),
                     linecolor="rgba(0,0,0,0)")
    fig.update_yaxes(gridcolor=t["grid"], zeroline=False,
                     tickfont=dict(size=9), ticksuffix=" dB")
    fig.update_layout(**_layout(t, height), bargap=0.45, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=CONFIG)
