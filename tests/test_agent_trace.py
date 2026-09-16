"""
Tests for the agent trace + LLM-source reporting (agents/trace.py,
agents/llm_client.py).

These run with no ANTHROPIC_API_KEY, which is the interesting case for CI:
every Claude call must be recorded as an attempted-but-fell-back call, and
the pipeline must still produce a full six-step trace.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# The whole point is the no-key path -- make sure a developer's real key in
# the environment can't silently turn these into live API calls.
os.environ.pop("ANTHROPIC_API_KEY", None)

from agents.graph import run_pipeline
from agents.llm_client import LLMCall, redact
from agents.trace import NODE_LABELS, traced
from data.db import ProfileStore
from dsp.parametric_eq import ParametricEQ
from perception.context_classifier import Context
from perception.synth_scenarios import synth_scenario, SR


def _run(command=""):
    # Throwaway profile file, so tests never touch the real dev profile
    # (same pattern as tests/test_profile_store.py).
    tmp = Path(tempfile.gettempdir()) / "auratune_trace_test_profiles.json"
    if tmp.exists():
        tmp.unlink()
    store = ProfileStore(mongo_uri=None, local_path=tmp)
    ambient, content = synth_scenario("quiet_podcast", SR)
    ctx = Context(noise_level="quiet", content_type="podcast",
                  ambient_rms_db=-60.0, content_features={})
    return run_pipeline(store, ParametricEQ(), "trace_test_user", ctx,
                        user_command=command)


def test_trace_has_one_entry_per_node_in_order():
    result = _run()
    trace = result["agent_trace"]
    assert [s["node"] for s in trace] == list(NODE_LABELS), \
        f"expected all 6 nodes in graph order, got {[s['node'] for s in trace]}"
    assert [s["step"] for s in trace] == [1, 2, 3, 4, 5, 6]
    assert all(s["duration_ms"] >= 0 for s in trace)


def test_explanation_falls_back_to_template_without_a_key():
    result = _run()
    assert result["explanation_source"] == "template"
    # ...and the attempted call is recorded rather than silently swallowed.
    calls = [c for s in result["agent_trace"] for c in s["llm_calls"]]
    assert any(c["purpose"] == "explanation" and c["status"] == "no_api_key"
               for c in calls), calls
    assert all(c["used_llm"] is False for c in calls)


def test_command_parse_source_is_reported():
    assert _run()["command_parse_source"] == "none"
    # With a command but no key, the keyword rules do the work -- and the
    # badge in the UI must say so.
    with_command = _run("less bass, this room is boomy")
    assert with_command["command_parse_source"] == "rules"
    assert with_command["command_deltas"] == {"bass_gain_db": -3.0}


def test_skipped_nodes_are_marked_skipped():
    result = _run()
    by_node = {s["node"]: s for s in result["agent_trace"]}
    # No audio buffers and no EQ spec were passed to run_pipeline above.
    assert by_node["noise_agent"]["skipped"]
    assert by_node["genre_agent"]["skipped"]
    assert by_node["projection_agent"]["skipped"]
    assert not by_node["eq_decision_agent"]["skipped"]


def test_traced_passes_the_node_result_through_untouched():
    sentinel = {"baseline_curve": None, "marker": object()}
    wrapped = traced("profile_agent", lambda s: s)
    out = wrapped(sentinel)
    assert out is sentinel
    assert out["marker"] is sentinel["marker"]


def test_api_keys_are_redacted_from_the_dev_view():
    leaked = "sk-ant-api03-AAAA1111bbbb_cccc-DDDD"
    call = LLMCall(purpose="command_parse", status="error",
                   system_prompt="be brief",
                   user_prompt=f"my key is {leaked} please",
                   error=f"401 from {leaked}")
    shown = call.to_dict()
    assert leaked not in shown["user_prompt"]
    assert leaked not in shown["error"]
    assert "***REDACTED***" in shown["user_prompt"]
    assert redact(None) is None


if __name__ == "__main__":
    test_trace_has_one_entry_per_node_in_order()
    test_explanation_falls_back_to_template_without_a_key()
    test_command_parse_source_is_reported()
    test_skipped_nodes_are_marked_skipped()
    test_traced_passes_the_node_result_through_untouched()
    test_api_keys_are_redacted_from_the_dev_view()
    print("All agent trace tests passed.")
