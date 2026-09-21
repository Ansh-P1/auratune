"""
Tests for the no-LLM command path (agents/eq_decision_agent.py) and for how
the explanation attributes a change (agents/explainer_agent.py).

Both cover bugs found while doing Track 3 (see docs/track3_findings.md):
the README's own example command matched no keyword rule, and the template
explanation credited "your command" even when the command changed nothing.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Keyword-rule tests only -- never let a local .env turn these into live,
# billable API calls (config.py loads .env, but won't override these).
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["GROQ_API_KEY"] = ""

from agents.eq_decision_agent import _rule_based_command_parse
from agents.explainer_agent import _template_sentence
from dsp.parametric_eq import ParametricEQ, TargetCurve
from perception.context_classifier import Context


def test_vocal_clarity_phrasings_all_match():
    """The phrasing the README and the UI placeholder suggest has to work --
    with no API key, these rules are the only thing parsing a command."""
    for command in ("make voices clearer",
                    "make the voices clearer",
                    "clearer voice please",
                    "I need clearer vocals",
                    "make the dialogue clearer",
                    "boost speech"):
        assert _rule_based_command_parse(command) == {"presence_gain_db": 3.0}, command


def test_other_keyword_rules_still_match():
    assert _rule_based_command_parse("less bass, this room is boomy") == {"bass_gain_db": -3.0}
    assert _rule_based_command_parse("more bass") == {"bass_gain_db": 3.0}
    assert _rule_based_command_parse("a bit brighter") == {"treble_gain_db": 2.0}
    assert _rule_based_command_parse("turn it down") == {"volume_db": -3.0}


def test_unrelated_command_matches_nothing():
    assert _rule_based_command_parse("what's the weather like") == {}
    assert _rule_based_command_parse("") == {}


def _state(user_command: str, command_deltas: dict) -> dict:
    """Baseline vs. decided differing only by a room-driven presence boost --
    i.e. a change the *environment* caused, not the user."""
    baseline = TargetCurve(name="podcast", volume_db=0.0, bass_gain_db=0.0,
                           presence_gain_db=0.0, treble_gain_db=0.0)
    decided = TargetCurve(name="podcast_live", volume_db=0.0, bass_gain_db=0.0,
                          presence_gain_db=3.5, treble_gain_db=0.0)
    return {
        "context": Context(noise_level="noisy", content_type="podcast",
                           ambient_rms_db=-20.0, content_features={}),
        "eq": ParametricEQ(),
        "baseline_curve": baseline,
        "decided_curve": decided,
        "user_command": user_command,
        "command_deltas": command_deltas,
    }


def test_unparsed_command_is_not_credited_for_the_change():
    sentence = _template_sentence(_state("play something upbeat", {}))
    assert "your command" not in sentence, sentence
    assert "noisy environment" in sentence


def test_parsed_command_is_credited():
    sentence = _template_sentence(_state("make voices clearer",
                                         {"presence_gain_db": 3.0}))
    assert "your command" in sentence, sentence


def test_no_command_still_names_the_environment():
    sentence = _template_sentence(_state("", {}))
    assert "noisy environment" in sentence, sentence


if __name__ == "__main__":
    test_vocal_clarity_phrasings_all_match()
    test_other_keyword_rules_still_match()
    test_unrelated_command_matches_nothing()
    test_unparsed_command_is_not_credited_for_the_change()
    test_parsed_command_is_credited()
    test_no_command_still_names_the_environment()
    print("All command fallback tests passed.")
