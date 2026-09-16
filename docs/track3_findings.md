# Track 3 (LLM & agents) — what I found

Written while doing Track 3 of the team brief. Diagram and node-by-node
breakdown live in [`agent_pipeline.md`](agent_pipeline.md).

## What shipped with this

- **Agent trace panel** — one row per LangGraph node, in execution order,
  timed, marked ✅ ran / ⏭️ skipped, each with its own detail expander.
  Recorded by `traced()` in `agents/trace.py`, which wraps every node in
  `agents/graph.py`.
- **Claude-vs-template badge** — the Explanation card now says which one
  wrote the sentence (`explanation_source`), and a caption says how a typed
  command was parsed (`command_parse_source`: Claude or keyword rules).
- **Claude prompts (dev view)** — the real system and user prompts, the
  response, latency, token counts and any error, per call. Anything
  matching an API-key pattern is stripped by `redact()` in
  `agents/llm_client.py` before it renders.
- **Tests** — `tests/test_agent_trace.py` (6 tests), wired into CI along
  with the three suites CI wasn't running yet (genre, noise, profile store).

## Run 1: no `ANTHROPIC_API_KEY` (done)

All 3 preset scenarios × {no command, "make voices clearer", "less bass,
this room is boomy"}. Every run: `explanation_source=template`, every
Claude call recorded as `no_api_key`, pipeline never breaks. Sample:

| Scenario | Command | Command deltas | Explanation |
|---|---|---|---|
| quiet + podcast | — | `{}` | "No change needed — your podcast curve already fits a quiet room." |
| noisy + music | — | `{}` | "Because of a noisy environment during music, I boosted vocal clarity by 3.5 dB, pulled bass back 2.5 dB." |
| noisy + music | less bass… | `{bass_gain_db: -3.0}` | "…boosted vocal clarity by 3.5 dB, pulled bass back 5.5 dB." |

## Run 2: with a live key — **not done yet**

Blocked exactly as the brief predicted: needs the shared
`ANTHROPIC_API_KEY`. Once someone supplies one, re-run the same 9
combinations and fill in the quality comparison. Everything needed to
judge it is already on screen (badge + prompts + response).

## Bugs found

**1. The demo command "make voices clearer" does nothing without a key.**
`_COMMAND_KEYWORDS` in `agents/eq_decision_agent.py` matches
`clear(er)? voice`, which needs "clearer voice" — the phrase the README and
the UI placeholder both suggest, "make voices clearer", matches nothing and
yields `{}`. So on the deployed site (no key set), the app's own example
command is silently a no-op. A plural/word-order-tolerant pattern would
fix it.

**2. The explanation credits a command that did nothing.** When a command
produces zero deltas, `_template_sentence()` still says "Because of your
command during podcast, I …" and then describes changes that actually came
from the noise rules. It should fall back to the environment trigger when
`command_deltas` is empty. Visible in the table above: the
"make voices clearer" rows read as if the command caused the change.

Both are one-line fixes in the decision/explainer agents, but they change
what the app decides rather than what it reports, so I left them out of
this observability change — flagging for the next sync.

## Notes for the demo

- The **⏭️ skipped** rows are the honest part of the trace: with no EQ app
  chosen, or a podcast scenario, three of the six nodes legitimately no-op.
  Good thing to point at when someone asks what the agents actually do.
- Claude can only ever move the curve through a **typed command**. Noise,
  genre and content deltas are lookup tables. Worth saying out loud in the
  presentation — it's a strength (deterministic, auditable), not a gap.
