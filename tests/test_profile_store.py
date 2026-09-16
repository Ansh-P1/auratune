"""
Tests for data/db.py's ProfileStore, exercised against its local-JSON
fallback (no MongoDB needed to run these).
"""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.db import ProfileStore


def _fresh_store() -> ProfileStore:
    """A ProfileStore backed by a throwaway local JSON file, so tests never
    touch data/profiles.local.json (the real dev/demo profile file)."""
    tmp = Path(tempfile.gettempdir()) / "auratune_test_profiles.json"
    if tmp.exists():
        tmp.unlink()
    return ProfileStore(mongo_uri=None, local_path=tmp)


def test_new_user_gets_default_template():
    store = _fresh_store()
    profile = store.get_profile("new_user")
    assert set(profile["target_curves"].keys()) == {"podcast", "music", "movie"}
    assert profile["history"] == []


def test_save_profile_persists_across_get():
    store = _fresh_store()
    profile = store.get_profile("u1")  # create it
    profile["target_curves"]["music"]["bass_gain_db"] = 5
    store.save_profile("u1", {"target_curves": profile["target_curves"]})

    reloaded = store.get_profile("u1")
    assert reloaded["target_curves"]["music"]["bass_gain_db"] == 5
    # untouched content types must survive the update
    assert reloaded["target_curves"]["podcast"]["bass_gain_db"] == -2


def test_log_adjustment_appends_to_history():
    store = _fresh_store()
    store.get_profile("u2")
    store.log_adjustment("u2", {"bass_gain_db": 2})
    store.log_adjustment("u2", {"bass_gain_db": -1})

    history = store.get_history("u2")
    assert len(history) == 2
    assert history[0]["bass_gain_db"] == 2
    assert history[1]["bass_gain_db"] == -1
    assert "ts" in history[0] and "ts" in history[1]


def test_get_history_respects_limit():
    store = _fresh_store()
    store.get_profile("u3")
    for i in range(5):
        store.log_adjustment("u3", {"i": i})

    limited = store.get_history("u3", limit=2)
    assert len(limited) == 2
    assert [entry["i"] for entry in limited] == [3, 4]


if __name__ == "__main__":
    test_new_user_gets_default_template()
    test_save_profile_persists_across_get()
    test_log_adjustment_appends_to_history()
    test_get_history_respects_limit()
    print("All profile store tests passed.")
