"""
AuraTune design system — single source of truth for colour, type, spacing.
Nothing else in the app should hard-code a hex value; import tokens() instead.
"""
from __future__ import annotations
import streamlit as st

LIGHT: dict = {
    "bg":            "#FAFAF8",
    "bg_elev":       "#FFFFFF",
    "card":          "#FFFFFF",
    "border":        "#E6E4DF",
    "border_strong": "#D4D1C9",
    "ink":           "#16171A",
    "ink_soft":      "#54565C",
    "muted":         "#5C6069",
    "accent":        "#0B7C8C",
    "accent_soft":   "#E3F4F6",
    "accent_ink":    "#065561",
    "warm":          "#B4632A",
    "warm_soft":     "#FBEDE2",
    "grid":          "#EDEBE6",
    "danger":        "#C4453A",
    "shadow":        "0 1px 2px rgba(16,17,26,.04),0 8px 24px -12px rgba(16,17,26,.10)",
    "shadow_lift":   "0 2px 4px rgba(16,17,26,.06),0 16px 40px -16px rgba(16,17,26,.18)",
}

DARK: dict = {
    "bg":            "#0C0D10",
    "bg_elev":       "#141519",
    "card":          "#141519",
    "border":        "#25272D",
    "border_strong": "#34373F",
    "ink":           "#F2F3F5",
    "ink_soft":      "#B4B7BF",
    "muted":         "#7C808A",
    "accent":        "#4FD1C5",
    "accent_soft":   "#16302F",
    "accent_ink":    "#7FE3D9",
    "warm":          "#F0A868",
    "warm_soft":     "#31220F",
    "grid":          "#1E2026",
    "danger":        "#E5484D",
    "shadow":        "0 1px 2px rgba(0,0,0,.5),0 8px 24px -12px rgba(0,0,0,.7)",
    "shadow_lift":   "0 2px 4px rgba(0,0,0,.5),0 16px 40px -16px rgba(0,0,0,.8)",
}

TYPE: dict = {
    "display": "2.5rem",
    "h1":      "1.7rem",
    "h2":      "1.12rem",
    "h3":      "0.95rem",
    "body":    "0.93rem",
    "small":   "0.81rem",
    "micro":   "0.71rem",
}


def tokens(dark: bool) -> dict:
    return DARK if dark else LIGHT


def is_dark() -> bool:
    return bool(st.session_state.get("dark_mode", True))


def _vars(t: dict) -> str:
    lines = [f"  --at-{k}: {v};" for k, v in t.items()
             if k not in ("shadow", "shadow_lift")]
    lines.append(f"  --at-shadow: {t['shadow']};")
    lines.append(f"  --at-shadow-lift: {t['shadow_lift']};")
    lines += [f"  --at-fs-{k}: {v};" for k, v in TYPE.items()]
    return ":root {\n" + "\n".join(lines) + "\n}"


_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    'family=Instrument+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&'
    'family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">'
)

_CSS = """
/* base */
html,body,[class*="css"],.stApp,button,input,select,textarea{
  font-family:'Instrument Sans',-apple-system,BlinkMacSystemFont,sans-serif;
  -webkit-font-smoothing:antialiased;
}
code,pre,.at-mono{font-family:'JetBrains Mono',ui-monospace,monospace;}

[data-testid="stAppViewContainer"],.stApp{background:var(--at-bg)!important;}
[data-testid="stHeader"]{background:transparent!important;height:0;}
[data-testid="stToolbar"]{right:1rem;}
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:2rem;padding-bottom:5rem;max-width:1280px;}

h1,h2,h3,h4{color:var(--at-ink)!important;letter-spacing:-0.021em;}
h1{font-weight:700!important;font-size:var(--at-fs-h1)!important;}
h2{font-weight:600!important;font-size:var(--at-fs-h2)!important;}
h3{font-weight:600!important;font-size:var(--at-fs-h3)!important;}
p,span,label,li,.stMarkdown{color:var(--at-ink-soft);font-size:var(--at-fs-body);}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *{
  color:var(--at-muted)!important;font-size:var(--at-fs-small)!important;}

/* cards */
div[data-testid="stVerticalBlockBorderWrapper"]{
  border:1px solid var(--at-border)!important;
  border-radius:14px!important;
  background:var(--at-card);
  box-shadow:var(--at-shadow);
  padding:1.1rem 1.2rem!important;
  margin-bottom:.85rem;
  transition:box-shadow .22s cubic-bezier(.2,.7,.3,1),
             border-color .22s cubic-bezier(.2,.7,.3,1);
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover{
  border-color:var(--at-border-strong)!important;
  box-shadow:var(--at-shadow-lift);
}

/* eyebrow */
.at-eyebrow{
  font-size:var(--at-fs-micro);
  font-weight:600;
  letter-spacing:.09em;
  text-transform:uppercase;
  color:var(--at-ink)!important;
  display:flex;
  align-items:center;
  gap:.5rem;
  margin-bottom:.65rem;
}

.at-eyebrow *{
  color:var(--at-ink)!important;
}

/* hero */
.at-hero{padding:.4rem 0 1.5rem;}
.at-hero h1{
  font-size:var(--at-fs-display)!important;
  line-height:1.02;margin:.3rem 0 .45rem;letter-spacing:-0.04em;
}
.at-hero .at-sub{color:var(--at-muted);font-size:.97rem;max-width:46ch;line-height:1.52;}
.at-hero .at-acc{color:var(--at-accent);}

.at-chip{
  display:inline-flex;align-items:center;gap:.4rem;
  background:var(--at-accent-soft);color:var(--at-accent-ink);
  font-size:var(--at-fs-micro);font-weight:600;letter-spacing:.07em;
  text-transform:uppercase;padding:.3rem .7rem;border-radius:999px;
  border:1px solid color-mix(in srgb,var(--at-accent) 22%,transparent);
}
.at-dot{
  width:6px;height:6px;border-radius:50%;background:var(--at-accent);
  animation:at-pulse 2.4s ease-out infinite;
}
@keyframes at-pulse{
  0%  {box-shadow:0 0 0 0 color-mix(in srgb,var(--at-accent) 55%,transparent);}
  70% {box-shadow:0 0 0 7px transparent;}
  100%{box-shadow:0 0 0 0 transparent;}
}

/* stat pills */
.at-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(88px,1fr));gap:.5rem;}
.at-stat{
  background:var(--at-bg);border:1px solid var(--at-border);
  border-radius:11px;padding:.65rem .8rem;min-width:0;
}
.at-stat .k{
  font-size:var(--at-fs-micro);text-transform:uppercase;letter-spacing:.07em;
  color:var(--at-muted);font-weight:600;margin-bottom:.25rem;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.at-stat .v{font-size:1.02rem;font-weight:650;color:var(--at-ink);line-height:1.2;word-break:break-word;}
.at-stat .v.num{font-family:'JetBrains Mono',monospace;font-weight:500;}
.at-stat.hot{background:var(--at-accent-soft);border-color:color-mix(in srgb,var(--at-accent) 25%,transparent);}
.at-stat.hot .v{color:var(--at-accent-ink);}

/* confidence meter */
.at-meter{height:5px;border-radius:999px;background:var(--at-border);overflow:hidden;margin-top:.42rem;}
.at-meter>i{
  display:block;height:100%;border-radius:999px;
  background:linear-gradient(90deg,var(--at-accent),color-mix(in srgb,var(--at-accent) 55%,var(--at-warm)));
  animation:at-grow .7s cubic-bezier(.2,.8,.2,1) both;
}
@keyframes at-grow{from{width:0!important;}}

/* buttons */
div.stButton>button,div.stDownloadButton>button{
  border-radius:10px!important;font-weight:600!important;
  font-size:var(--at-fs-body)!important;
  border:1px solid var(--at-border)!important;
  background:var(--at-bg-elev)!important;color:var(--at-ink)!important;
  transition:transform .14s cubic-bezier(.2,.8,.2,1),box-shadow .14s,background .14s;
}
div.stButton>button:hover{border-color:var(--at-border-strong)!important;transform:translateY(-1px);}
div.stButton>button[kind="primary"]{
  background:var(--at-accent)!important;color:var(--at-bg)!important;
  border-color:transparent!important;letter-spacing:.01em;
  box-shadow:0 4px 16px -6px color-mix(in srgb,var(--at-accent) 80%,transparent);
}
div.stButton>button[kind="primary"]:hover{
  transform:translateY(-1px);
  box-shadow:0 8px 22px -8px color-mix(in srgb,var(--at-accent) 90%,transparent);
}
div.stButton>button[kind="primary"]:active{transform:translateY(0) scale(.99);}

/* inputs */
.stTextInput input,.stNumberInput input,[data-baseweb="select"]>div,
.stFileUploader section{
  border-radius:10px!important;border-color:var(--at-border)!important;
  background:var(--at-bg)!important;color:var(--at-ink)!important;
  font-size:var(--at-fs-body)!important;
}
.stTextInput input:focus{
  border-color:var(--at-accent)!important;
  box-shadow:0 0 0 3px color-mix(in srgb,var(--at-accent) 16%,transparent)!important;
}
[data-testid="stWidgetLabel"] p{
  color:var(--at-ink)!important;font-size:var(--at-fs-small)!important;font-weight:550!important;
}
[data-baseweb="popover"] li,[data-baseweb="menu"],[data-baseweb="popover"]>div{
  background:var(--at-card)!important;color:var(--at-ink)!important;
}
[data-baseweb="popover"] li:hover{background:var(--at-accent-soft)!important;}

/* tabs */
[data-baseweb="tab-list"]{gap:.2rem;border-bottom:1px solid var(--at-border);}
[data-baseweb="tab"]{
  font-size:var(--at-fs-small)!important;font-weight:600!important;
  color:var(--at-muted)!important;padding:.5rem .8rem!important;
}
[data-baseweb="tab"][aria-selected="true"]{color:var(--at-ink)!important;}
[data-baseweb="tab-highlight"]{background:var(--at-accent)!important;height:2px;}

/* alerts / expander */
[data-testid="stAlert"]{
  border-radius:12px!important;background:var(--at-accent-soft)!important;
  border:1px solid color-mix(in srgb,var(--at-accent) 20%,transparent)!important;
  color:var(--at-ink)!important;
}
[data-testid="stExpander"]{
  border-radius:12px!important;border-color:var(--at-border)!important;
  background:var(--at-bg)!important;overflow:hidden;
}
[data-testid="stExpander"] summary{
  color:var(--at-ink-soft)!important;font-size:var(--at-fs-small);
}

/* fader rack */
.at-rack{
  display:flex;gap:.45rem;align-items:flex-end;justify-content:space-between;
  padding:1rem .25rem .35rem;overflow-x:auto;
}
.at-fader{flex:1 1 0;min-width:32px;text-align:center;}
.at-fader .track{
  position:relative;height:120px;width:6px;margin:0 auto .45rem;
  border-radius:999px;background:var(--at-border);
}
.at-fader .zero{position:absolute;left:-4px;right:-4px;height:1px;background:var(--at-border-strong);}
.at-fader .fill{position:absolute;left:0;right:0;border-radius:999px;background:var(--at-warm);opacity:.82;}
.at-fader .knob{
  position:absolute;left:50%;transform:translate(-50%,-50%);
  width:18px;height:10px;border-radius:3px;
  background:var(--at-card);border:1.5px solid var(--at-warm);
  box-shadow:var(--at-shadow);
  transition:top .55s cubic-bezier(.2,.8,.2,1);
}
.at-fader .val{font-family:'JetBrains Mono',monospace;font-size:var(--at-fs-micro);font-weight:500;color:var(--at-ink);}
.at-fader .hz{font-size:9px;color:var(--at-muted);letter-spacing:.02em;}
.at-fader.clipped .knob{border-color:var(--at-danger);}
.at-fader.clipped .fill{background:var(--at-danger);}

/* timeline */
.at-tl{position:relative;padding-left:1rem;}
.at-tl::before{
  content:"";position:absolute;left:3px;top:.4rem;bottom:.4rem;
  width:1px;background:var(--at-border);
}
.at-tl-item{position:relative;padding:.4rem 0;}
.at-tl-item::before{
  content:"";position:absolute;left:-1rem;top:.74rem;
  width:7px;height:7px;border-radius:50%;
  background:var(--at-card);border:1.5px solid var(--at-border-strong);
}
.at-tl-item:first-child::before{background:var(--at-accent);border-color:var(--at-accent);}
.at-tl-item .t{font-size:var(--at-fs-micro);text-transform:uppercase;letter-spacing:.06em;color:var(--at-muted);font-weight:600;}
.at-tl-item .d{font-size:var(--at-fs-small);color:var(--at-ink-soft);line-height:1.45;}

/* empty state */
.at-empty{
  border:1px dashed var(--at-border-strong);border-radius:14px;
  padding:2.6rem 1.5rem;text-align:center;background:var(--at-bg);
}
.at-empty .at-wave{display:flex;gap:4px;justify-content:center;align-items:flex-end;height:40px;margin-bottom:1rem;}
.at-empty .at-wave i{
  width:4px;border-radius:999px;background:var(--at-accent);opacity:.5;
  animation:at-bounce 1.25s ease-in-out infinite;
}
@keyframes at-bounce{0%,100%{height:8px;}50%{height:36px;}}
.at-empty h4{margin:0 0 .3rem;font-size:.97rem;color:var(--at-ink);}
.at-empty p{color:var(--at-muted);font-size:var(--at-fs-small);margin:0 auto;max-width:34ch;}

/* device badge */
.at-device{display:flex;align-items:center;gap:.65rem;padding:.35rem 0;}
.at-device .lbl{font-size:var(--at-fs-small);color:var(--at-muted);}

/* misc */
[data-testid="stToggle"] label p{color:var(--at-ink-soft)!important;font-size:var(--at-fs-small)!important;}
hr{border-color:var(--at-border)!important;}
::-webkit-scrollbar{height:7px;width:7px;}
::-webkit-scrollbar-thumb{background:var(--at-border-strong);border-radius:999px;}
::-webkit-scrollbar-track{background:transparent;}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important;}}
"""


def inject(dark: bool | None = None) -> None:
    """Call once, right after st.set_page_config."""
    if dark is None:
        dark = is_dark()

    st.html(
        _FONTS
        + "<style>"
        + _vars(tokens(dark))
        + _CSS
        + "</style>"
    )