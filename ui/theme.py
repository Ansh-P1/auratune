"""
AuraTune design system — single source of truth for colour, type, spacing.
Nothing else in the app should hard-code a hex value; import tokens() instead.
"""
from __future__ import annotations
import streamlit as st

## Light theme: Vanilla Cream background with Midnight Lagoon ink/accents.
## Dark theme remains Java Brown / Transparent Yellow.
LIGHT: dict = {
    "bg":            "#FFF7E6",
    "bg_elev":       "#FFF7E6",
    "card":          "#FFF7E6",
    "border":        "#C7CFD5",
    "border_strong": "#AEB9C1",
    "ink":           "#70483E",
    "ink_soft":      "#70483E",
    "muted":         "#70483E",
    "accent":        "#2D3A47",
    "accent_soft":   "#E1E6EA",
    "accent_ink":    "#2D3A47",
    "warm":          "#2D3A47",
    "warm_soft":     "#E1E6EA",
    "grid":          "#EAE1CB",
    "danger":        "#A94335",
    "shadow":        "0 1px 2px rgba(45,58,71,.06),0 8px 24px -12px rgba(45,58,71,.14)",
    "shadow_lift":   "0 2px 4px rgba(45,58,71,.08),0 16px 40px -16px rgba(45,58,71,.22)",
}

DARK: dict = {
    "bg":            "#231815",
    "bg_elev":       "#2E211D",
    "card":          "#2E211D",
    "border":        "#3D2C26",
    "border_strong": "#4F3931",
    "ink":           "#F5EFC6",
    "ink_soft":      "#D8CBA0",
    "muted":         "#9C8B78",
    "accent":        "#F5EFC6",
    "accent_soft":   "#3D3520",
    "accent_ink":    "#F5EFC6",
    "warm":          "#F5EFC6",
    "warm_soft":     "#3D3520",
    "grid":          "#33241F",
    "danger":        "#C97A65",
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
    return bool(st.session_state.get(
        "dark_mode_enabled", st.session_state.get("dark_mode", True)))


def _vars(t: dict) -> str:
    # CSS custom properties are punctuation-exact -- every reference in _CSS
    # below uses hyphens (var(--at-border-strong)), so definitions must too,
    # not the raw underscored Python dict key (--at-border_strong), or the
    # variable silently never resolves.
    lines = [f"  --at-{k.replace('_', '-')}: {v};" for k, v in t.items()
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

html,body,[data-testid="stAppViewContainer"],.stApp,
[data-testid="stMain"],[data-testid="stMainBlockContainer"]{
  background:var(--at-bg)!important;
}
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
div[data-testid="stMarkdownContainer"] .at-hero h1{
  font-size:clamp(3.2rem,6vw,5rem)!important;
  line-height:1.02;margin:.3rem 0 .45rem;letter-spacing:-0.04em;
}
div[data-testid="stMarkdownContainer"] .at-hero h1 [data-heading-text]{
  font-size:inherit!important;
}
div[data-testid="stMarkdownContainer"] .at-hero h1 .at-acc{
  font-size:inherit!important;
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
.st-key-detected_context_panel .at-stats{padding-bottom:.7rem;}
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
  background:var(--at-bg-elev)!important;color:var(--at-ink)!important;
  border:1px solid var(--at-accent)!important;letter-spacing:0;
  box-shadow:none!important;transform:none;
}
div.stButton>button[kind="primary"]:hover{
  background:var(--at-accent-soft)!important;
  transform:none;box-shadow:none!important;
}
div.stButton>button[kind="primary"]:active{transform:none;}

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
/* Browsers dim placeholder text by default (often ~50% opacity), which on
   top of an already-muted colour made it too faint to read against the
   card background. */
.stTextInput input::placeholder,.stNumberInput input::placeholder{
  color:var(--at-muted)!important;opacity:1!important;
}
[data-testid="stWidgetLabel"] p{
  color:var(--at-ink)!important;font-size:var(--at-fs-small)!important;font-weight:550!important;
}
[data-baseweb="popover"] li,[data-baseweb="menu"],[data-baseweb="popover"]>div{
  background:var(--at-card)!important;color:var(--at-ink)!important;
}
[data-baseweb="popover"] li:hover{background:var(--at-accent-soft)!important;}
/* Newer Streamlit's selectbox is a react-aria ComboBox with no
   [data-baseweb] attributes at all, so the rules above match nothing there
   -- kept for older Streamlit builds, and targeted by role here too, which
   is stable across versions. The open dropdown list renders in a portal
   appended to <body>, not inside the app tree, so it's selected
   structurally by its listbox child. */
[data-testid="stSelectbox"] input[role="combobox"]{color:var(--at-ink)!important;}
[data-testid="stSelectbox"] [role="group"]{
  background:var(--at-card)!important;border-color:var(--at-border)!important;
}
body div:has(> [role="listbox"]){background:var(--at-card)!important;border:1px solid var(--at-border)!important;}
[role="option"]{background:transparent!important;color:var(--at-ink)!important;}
[role="option"]:hover,[role="option"][data-hovered="true"]{background:var(--at-accent-soft)!important;}

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

/* agent trace -- a flowing connected timeline, not stacked boxes */
.at-trace{position:relative;padding-left:1rem;}
.at-trace::before{
  content:"";position:absolute;left:3px;top:.5rem;bottom:.5rem;
  width:1px;background:var(--at-border);
}
.at-trace-item{position:relative;padding:.55rem 0 .75rem;}
.at-trace-item::before{
  content:"";position:absolute;left:-1rem;top:.62rem;
  width:7px;height:7px;border-radius:50%;
  background:var(--at-accent);border:1.5px solid var(--at-accent);
}
.at-trace-item.skipped::before{background:var(--at-card);border-color:var(--at-border-strong);}
.at-trace-item .head{font-size:var(--at-fs-body);font-weight:600;color:var(--at-ink);}
.at-trace-item .head .meta{
  font-weight:500;font-size:var(--at-fs-small);color:var(--at-muted);margin-left:.45rem;
}
.at-trace-item .summary{
  font-size:var(--at-fs-small);color:var(--at-ink-soft);line-height:1.45;margin-top:.15rem;
}
.at-trace-item details{margin-top:.4rem;}
.at-trace-item details summary{
  cursor:pointer;font-size:var(--at-fs-micro);color:var(--at-muted);list-style:none;
}
.at-trace-item details summary::-webkit-details-marker{display:none;}
.at-trace-item details summary::before{content:"›  ";}
.at-trace-item details[open] summary::before{content:"⌄  ";}
.at-trace-item details pre,.at-trace-item details [data-testid="stMarkdownPre"]{
  background:var(--at-bg)!important;border:1px solid var(--at-border);border-radius:8px;
  padding:.6rem .7rem;font-size:var(--at-fs-micro);overflow-x:auto;margin-top:.4rem;
  color:var(--at-ink-soft)!important;white-space:pre-wrap;word-break:break-word;
}

/* misc */
[data-testid="stToggle"] label p{color:var(--at-ink-soft)!important;font-size:var(--at-fs-small)!important;}
hr{border-color:var(--at-border)!important;}
::-webkit-scrollbar{height:7px;width:7px;}
::-webkit-scrollbar-thumb{background:var(--at-border-strong);border-radius:999px;}
::-webkit-scrollbar-track{background:transparent;}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important;}}
"""


_LIGHT_OVERRIDES = """
/* Light theme: Vanilla Cream background, Midnight Lagoon ink/accents. */
:root{--primary-color:#2D3A47;}
.at-hero .at-acc{color:#70483E!important;}
.at-stat.hot{background:#E8D6C8!important;border-color:#D3B9A8!important;}
.at-stat.hot .k,.at-stat.hot .v{color:#70483E!important;}
[data-testid="stAlert"]{
  background:#E8D6C8!important;border-color:#D3B9A8!important;
  color:#70483E!important;
}
[data-testid="stAlert"] [class*="stAlertContainer"]{
  background:#E8D6C8!important;border-color:#D3B9A8!important;
}
[data-testid="stAlert"] *{color:#70483E!important;}
/* Keep the Edit values dialog on the dark-theme palette in either theme. */
[role="dialog"], [role="dialog"]>div, [role="dialog"] section{
  background:#231815!important;color:#F5EFC6!important;
}
[role="dialog"] h1,[role="dialog"] h2,[role="dialog"] h3,
[role="dialog"] h4,[role="dialog"] p,[role="dialog"] label,
[role="dialog"] [data-testid="stWidgetLabel"] *,
[role="dialog"] [data-testid="stCaptionContainer"] *,
[role="dialog"] [data-testid="stMarkdownContainer"] *{
  color:#F5EFC6!important;
}
[role="dialog"] .stTextInput input,
[role="dialog"] .stNumberInput input,
[role="dialog"] [data-baseweb="select"]>div{
  background:#2E211D!important;color:#F5EFC6!important;
  border-color:#4F3931!important;
}
[role="dialog"] .stTextInput input::placeholder,
[role="dialog"] .stNumberInput input::placeholder{
  color:#9C8B78!important;opacity:1!important;
}
[role="dialog"] button[aria-label="Decrement"],
[role="dialog"] button[aria-label="Increment"],
[role="dialog"] button[aria-label="Decrement"] *,
[role="dialog"] button[aria-label="Increment"] *{
  color:#94B2C4!important;
}
[role="dialog"] button[aria-label="Decrement"] svg,
[role="dialog"] button[aria-label="Increment"] svg{
  fill:#94B2C4!important;stroke:#94B2C4!important;
}
[role="dialog"] div.stButton>button[kind="primary"],
[role="dialog"] div.stButton>button:not([kind="primary"]){
  border:1px solid #94B2C4!important;border-color:#94B2C4!important;
}
[role="dialog"] div.stButton>button:not([kind="primary"]),
[role="dialog"] div.stButton>button:not([kind="primary"]) *{
  color:#70483E!important;
}
[role="dialog"] button[aria-label="Close"],
[role="dialog"] button[aria-label="Close"] svg,
[role="dialog"] button[aria-label="Close"] path{
  color:#94B2C4!important;stroke:#94B2C4!important;
}
label:has(input[role="switch"])>div:nth-of-type(1){
  background:#80665C!important;
}
label:has(input[role="switch"]:checked)>div:nth-of-type(1){
  background:#70483E!important;
}
label:has(input[role="switch"])>div:nth-of-type(1)>div{
  background:#FFF7E6!important;
}
html body div[data-testid="stVerticalBlock"]{
  border-color:#2D3A47!important;
}
html body div[data-testid="stVerticalBlock"]:hover{
  border-color:#2D3A47!important;
}
.stTextInput input,.stNumberInput input,
[data-testid="stSelectbox"] [role="group"],
[data-testid="stSelectbox"] input[role="combobox"],
[data-baseweb="select"]>div,.stFileUploader section{
  background:#E1E6EA!important;
  border-color:#2D3A47!important;
}
body div:has(> [role="listbox"]),
body [role="listbox"],body [role="option"]{
  background:#FFF7E6!important;
}
body [role="option"]:hover,
body [role="option"][data-hovered="true"],
body [role="option"][data-focused="true"],
body [role="option"][aria-selected="true"],
body [role="option"][data-selected="true"],
body [data-baseweb="menu"] li:hover,
body [data-baseweb="menu"] li[aria-selected="true"]{
  background:#E8D6C8!important;color:#70483E!important;
}
div.stButton>button[kind="primary"]{
  background:#2D3A47!important;color:#FFF7E6!important;
  border:1px solid #2D3A47!important;box-shadow:none!important;transform:none;
}
div.stButton>button[kind="primary"]:hover{
  background:#232D37!important;border-color:#2D3A47!important;
  color:#FFF7E6!important;box-shadow:none!important;transform:none;
}
div.stButton>button[kind="primary"] p,
div.stButton>button[kind="primary"] span{
  color:#FFF7E6!important;
}
div.stButton>button[kind="primary"]:active{transform:none;}
.st-key-run_adaptation div.stButton>button[kind="primary"]{
  background:#70483E!important;color:#FFF7E6!important;
  border:1px solid #A77C69!important;box-shadow:none!important;
}
.st-key-run_adaptation div.stButton>button[kind="primary"]:hover,
.st-key-run_adaptation div.stButton>button[kind="primary"]:focus-visible{
  background:#5E392F!important;border-color:#A77C69!important;
  color:#FFF7E6!important;box-shadow:none!important;
}
.st-key-run_adaptation div.stButton>button[kind="primary"] p,
.st-key-run_adaptation div.stButton>button[kind="primary"] span{
  color:#FFF7E6!important;
}
.st-key-edit_values_button div.stButton>button[kind="primary"]{
  background:#E8D6C8!important;color:#70483E!important;
  border:1px solid #D3B9A8!important;
}
.st-key-edit_values_button div.stButton>button[kind="primary"]:hover{
  background:#DCC3B1!important;border-color:#D3B9A8!important;
}
.st-key-edit_values_button div.stButton>button[kind="primary"] p,
.st-key-edit_values_button div.stButton>button[kind="primary"] span{
  color:#70483E!important;
}
.at-fader .fill{background:#4A5A6B!important;opacity:1!important;}
.at-fader.clipped .fill{background:#A94335!important;}
.at-tl-item::before,.at-tl-item:first-child::before{
  background:#2D3A47;border-color:#2D3A47;
}
.at-trace-item::before{background:#2D3A47;border-color:#2D3A47;}
.at-trace-item.skipped::before{background:#FFF7E6;border-color:#2D3A47;}
.st-key-history_panel [data-testid="stExpander"]{
  background:#E8D6C8!important;border-color:#E8D6C8!important;
}
.st-key-history_panel [data-testid="stExpander"] summary,
.st-key-history_panel [data-testid="stExpander"] summary:hover,
.st-key-history_panel [data-testid="stExpander"] summary:focus,
.st-key-history_panel [data-testid="stExpander"] summary:focus-visible{
  background:#E8D6C8!important;color:#70483E!important;
}
"""


_DARK_OVERRIDES = """
/* Keep Streamlit's selected-state accents light against Java Brown surfaces. */
:root{--primary-color:#94B2C4;}
.react-aria-SelectionIndicator,[data-baseweb="tab-highlight"]{
  background:#94B2C4!important;
}
label:has(input[role="switch"]) [data-testid="stWidgetLabel"],
label:has(input[role="switch"]) [data-testid="stWidgetLabel"] *{
  color:#F5EFC6!important;
}
label:has(input[role="switch"])>div:nth-of-type(1){
  background:#4F3931!important;
}
label:has(input[role="switch"]:checked)>div:nth-of-type(1){
  background:#94B2C4!important;
}
label:has(input[role="switch"])>div:nth-of-type(1)>div{
  background:#F5EFC6!important;
}
[role="dialog"] h1,[role="dialog"] h2,[role="dialog"] h3,
[role="dialog"] h4,[role="dialog"] p,[role="dialog"] label,
[role="dialog"] [data-testid="stWidgetLabel"] *,
[role="dialog"] [data-testid="stCaptionContainer"] *,
[role="dialog"] [data-testid="stMarkdownContainer"] *{
  color:#F5EFC6!important;
}
[role="dialog"] button[aria-label="Decrement"],
[role="dialog"] button[aria-label="Increment"],
[role="dialog"] button[aria-label="Decrement"] *,
[role="dialog"] button[aria-label="Increment"] *{
  color:#94B2C4!important;
}
[role="dialog"] button[aria-label="Decrement"] svg,
[role="dialog"] button[aria-label="Increment"] svg{
  fill:#94B2C4!important;stroke:#94B2C4!important;
}
[role="dialog"] div.stButton>button[kind="primary"]{
  border:1px solid #94B2C4!important;border-color:#94B2C4!important;
}
[role="dialog"] div.stButton>button:not([kind="primary"]){
  border:1px solid #94B2C4!important;border-color:#94B2C4!important;
}
[role="dialog"] div.stButton>button:not([kind="primary"]),
[role="dialog"] div.stButton>button:not([kind="primary"]) *{
  color:#70483E!important;
}
[role="dialog"] button[aria-label="Close"],
[role="dialog"] button[aria-label="Close"] svg,
[role="dialog"] button[aria-label="Close"] path{
  color:#94B2C4!important;stroke:#94B2C4!important;
}
div[data-testid="stVerticalBlock"]:has(.at-empty){
  padding-bottom:2rem!important;
  background:transparent!important;border-color:rgba(245,239,198,.2)!important;
}
.at-empty{background:#2E211D!important;border-color:#3D2C26!important;}
.at-empty .at-wave i{background:#94B2C4!important;opacity:1!important;}
.st-key-run_adaptation div.stButton>button[kind="primary"]{
  background:#2E211D!important;color:#F5EFC6!important;
  border-color:#94B2C4!important;
  box-shadow:0 0 0 1px rgba(148,178,196,.16)!important;
}
.st-key-run_adaptation div.stButton>button[kind="primary"]:hover,
.st-key-run_adaptation div.stButton>button[kind="primary"]:focus-visible{
  background:#33241F!important;border-color:#94B2C4!important;
  color:#F5EFC6!important;
  box-shadow:0 0 0 2px rgba(148,178,196,.22)!important;
}
.st-key-run_adaptation div.stButton>button[kind="primary"] p,
.st-key-run_adaptation div.stButton>button[kind="primary"] span{
  color:#F5EFC6!important;
}
.st-key-history_panel [data-testid="stExpander"] summary,
.st-key-history_panel [data-testid="stExpander"] summary:hover,
.st-key-history_panel [data-testid="stExpander"] summary:focus,
.st-key-history_panel [data-testid="stExpander"] summary:focus-visible{
  background:#33241F!important;color:#D8CBA0!important;
}
.st-key-history_panel [data-testid="stExpander"]{
  background:#2E211D!important;border-color:#3D2C26!important;
}
"""


def inject(dark: bool | None = None) -> None:
    """Call once, right after st.set_page_config.

    Two separate st.markdown(..., unsafe_allow_html=True) calls, not one
    st.html() call with everything concatenated -- confirmed by hand in a
    live browser that the combined payload is unreliable both ways:
    st.html() sometimes dropped the whole thing (no <link>, no <style>,
    no error, no console message -- flaky across otherwise-identical
    reruns), and st.markdown() rendered the Google Fonts <link> tags
    immediately followed by a large <style> block as visible garbled text
    instead of applying it, because CommonMark's raw-HTML-block detection
    doesn't reliably recognize multiple different-tag siblings glued
    together as one HTML block. A <style> block starting a string on its
    own is unambiguous and st.markdown renders it invisibly every time --
    same for the <link> tags in their own call.
    """
    if dark is None:
        dark = is_dark()

    st.markdown(_FONTS, unsafe_allow_html=True)
    light_overrides = "" if dark else _LIGHT_OVERRIDES
    dark_overrides = _DARK_OVERRIDES if dark else ""
    st.markdown(
        "<style>" + _vars(tokens(dark)) + _CSS + light_overrides + dark_overrides + "</style>",
        unsafe_allow_html=True,
    )
