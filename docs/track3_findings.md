# Track 3 (LLM & agents) — what I found

Written while doing Track 3 of the team brief. Diagram and node-by-node
breakdown live in [`agent_pipeline.md`](agent_pipeline.md).

## What shipped with this

- **Agent trace panel** — one row per LangGraph node, in execution order,
  timed, marked ✅ ran / ⏭️ skipped, each with its own detail expander.
  Recorded by `traced()` in `agents/trace.py`, which wraps every node in
  `agents/graph.py`.
- **LLM-vs-template badge** — the Explanation card now names the model that
  wrote the sentence (`explanation_source` + `explanation_model`), and a
  caption says how a typed command was parsed (`command_parse_source`: the
  LLM or keyword rules).
- **Groq provider path** — command parsing and explanations now run on
  Anthropic *or* Groq's OpenAI-compatible API (`GROQ_API_KEY`), so teammates
  without an Anthropic key can still exercise the LLM half of the pipeline.
  Anthropic is preferred when both keys are set; `AURATUNE_LLM_PROVIDER`
  forces one. No new dependency — plain HTTPS, like the Gemini path.
- **Prompt dev view** — the real system and user prompts, the
  response, latency, token counts and any error, per call. Anything
  matching an API-key pattern is stripped by `redact()` in
  `agents/llm_client.py` before it renders.
- **Tests** — `tests/test_agent_trace.py` (7 tests), wired into CI along
  with the three suites CI wasn't running yet (genre, noise, profile store).

## Run 1: no API key at all (done)

All 3 preset scenarios × {no command, "make voices clearer", "less bass,
this room is boomy"}. Every run: `explanation_source=template`, every
LLM call recorded as `no_api_key`, pipeline never breaks. Sample:

| Scenario | Command | Command deltas | Explanation |
|---|---|---|---|
| quiet + podcast | — | `{}` | "No change needed — your podcast curve already fits a quiet room." |
| noisy + music | — | `{}` | "Because of a noisy environment during music, I boosted vocal clarity by 3.5 dB, pulled bass back 2.5 dB." |
| noisy + music | less bass… | `{bass_gain_db: -3.0}` | "…boosted vocal clarity by 3.5 dB, pulled bass back 5.5 dB." |

## Run 2: with a live key (done, via Groq)

No Anthropic key was available, so the LLM path was added for **Groq**
(`openai/gpt-oss-120b`) and exercised there — same 9 combinations. The
comparison below is therefore "a real LLM vs. the template", not Claude
specifically; the Anthropic path is unchanged and still wins automatically
when `ANTHROPIC_API_KEY` is set.

| | No key (template/rules) | LLM (gpt-oss-120b) |
|---|---|---|
| quiet + podcast, no command | "No change needed — your podcast curve already fits a quiet room." | "When the room is very quiet, the EQ stays unchanged, so the podcast sounds just as originally mixed." |
| noisy + music, no command | "Because of a noisy environment during music, I boosted vocal clarity by 3.5 dB, pulled bass back 2.5 dB." | "When background noise gets loud, the system gently turns down the bass and lifts the vocals so you can hear the words more clearly." |
| quiet + podcast, "make voices clearer" | `{}` — **command silently ignored** | `{presence +3.0, treble +2.0}` |

What the comparison actually shows:

- **The template leaks jargon and dB numbers**; the LLM describes the
  audible effect. For a non-technical demo the LLM reads better.
- **The LLM is the only path that understands the command.** The keyword
  rules miss "make voices clearer" entirely (bug 1 below), so with no key
  the app's own example command does nothing.
- **The LLM sometimes describes changes it didn't make** — e.g. "lifts the
  vocals" on a run whose command deltas were bass-only. The numbers on
  screen stay correct (they're rule-derived); only the prose drifts.
- **Cost of the nicer sentence:** ~0.8 s per call, two calls per run.

Setup notes for whoever repeats this:

- Put `GROQ_API_KEY=...` in a local `.env` (gitignored; `config.py` loads it
  without overriding a real export).
- Two things had to be handled for Groq specifically: its edge rejects
  urllib's default user agent (Cloudflare error 1010), and reasoning models
  spend part of the token budget on hidden reasoning, returning an empty
  message unless `reasoning_effort` is low with headroom left over.

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
- The LLM can only ever move the curve through a **typed command**. Noise,
  genre and content deltas are lookup tables. Worth saying out loud in the
  presentation — it's a strength (deterministic, auditable), not a gap.
