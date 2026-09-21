"""
Profile & history store.

One document per user: audiogram (hearing boost curve), per-content-type
target curves, and adjustment history. Backed by MongoDB when MONGO_URI is
set and reachable; otherwise falls back transparently to a local JSON file
so the whole pipeline runs offline / in dev without standing up Atlas.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_LOCAL_PATH = Path(__file__).parent / "profiles.local.json"

DEFAULT_PROFILE_TEMPLATE: Dict[str, Any] = {
    "audiogram": {},  # freq_hz(str) -> boost_db, from the hearing sweep test
    "target_curves": {
        "podcast": {"volume_db": 0, "bass_gain_db": -2, "presence_gain_db": 3, "treble_gain_db": 0},
        "music":   {"volume_db": 0, "bass_gain_db": 2,  "presence_gain_db": 0, "treble_gain_db": 1},
        "movie":   {"volume_db": 0, "bass_gain_db": 1,  "presence_gain_db": 2, "treble_gain_db": 0},
    },
    "history": [],
    "feedback": [],
}



class ProfileStore:
    def __init__(self, mongo_uri: Optional[str] = None, local_path: Path = DEFAULT_LOCAL_PATH):
        self.mongo_uri = mongo_uri or os.environ.get("MONGO_URI")
        self.local_path = local_path
        self._collection = None
        self.backend = "local_json"
        if self.mongo_uri:
            try:
                from pymongo import MongoClient
                client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=1500)
                client.admin.command("ping")  # fail fast if unreachable
                db = client["auratune"]
                self._collection = db["user_profiles"]
                self.backend = "mongodb"
            except Exception:
                self._collection = None
                self.backend = "local_json"

        if self.backend == "local_json" and not self.local_path.exists():
            self.local_path.write_text(json.dumps({}))

    # -- internal local-json helpers -----------------------------------
    def _read_local(self) -> Dict[str, Any]:
        return json.loads(self.local_path.read_text())

    def _write_local(self, data: Dict[str, Any]) -> None:
        self.local_path.write_text(json.dumps(data, indent=2))

    # -- public API -------------------------------------------------------
    def get_profile(self, user_id: str) -> Dict[str, Any]:
        if self.backend == "mongodb":
            doc = self._collection.find_one({"_id": user_id})
            if doc is None:
                doc = {"_id": user_id, **json.loads(json.dumps(DEFAULT_PROFILE_TEMPLATE))}
                self._collection.insert_one(doc)
            return doc
        data = self._read_local()
        if user_id not in data:
            data[user_id] = json.loads(json.dumps(DEFAULT_PROFILE_TEMPLATE))
            self._write_local(data)
        return {"_id": user_id, **data[user_id]}

    def save_profile(self, user_id: str, profile: Dict[str, Any]) -> None:
        profile = {k: v for k, v in profile.items() if k != "_id"}
        if self.backend == "mongodb":
            self._collection.update_one({"_id": user_id}, {"$set": profile}, upsert=True)
            return
        data = self._read_local()
        data.setdefault(user_id, json.loads(json.dumps(DEFAULT_PROFILE_TEMPLATE)))
        data[user_id].update(profile)
        self._write_local(data)

    def log_adjustment(self, user_id: str, entry: Dict[str, Any]) -> None:
        entry = {"ts": time.time(), **entry}
        if self.backend == "mongodb":
            self._collection.update_one({"_id": user_id}, {"$push": {"history": entry}}, upsert=True)
            return
        data = self._read_local()
        data.setdefault(user_id, json.loads(json.dumps(DEFAULT_PROFILE_TEMPLATE)))
        data[user_id].setdefault("history", []).append(entry)
        self._write_local(data)

    def get_history(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        profile = self.get_profile(user_id)
        return profile.get("history", [])[-limit:]

    def log_feedback(
        self,
        user_id: str,
        context_bucket: str,
        agent_curve: Dict[str, Any],
        user_correction_delta: Dict[str, Any],
    ) -> None:
        """Store a feedback event: what the agent decided, what the user corrected it to,
        and the context bucket under which it happened."""
        entry = {
            "ts": time.time(),
            "context_bucket": context_bucket,
            "agent_curve": agent_curve,
            "user_correction_delta": user_correction_delta,
        }
        if self.backend == "mongodb":
            self._collection.update_one({"_id": user_id}, {"$push": {"feedback": entry}}, upsert=True)
            return
        data = self._read_local()
        data.setdefault(user_id, json.loads(json.dumps(DEFAULT_PROFILE_TEMPLATE)))
        data[user_id].setdefault("feedback", []).append(entry)
        self._write_local(data)

    def get_feedback(
        self,
        user_id: str,
        context_bucket: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieve feedback events for a user, optionally filtered by context bucket."""
        profile = self.get_profile(user_id)
        entries = profile.get("feedback", [])
        if context_bucket is not None:
            entries = [e for e in entries if e.get("context_bucket") == context_bucket]
        return entries[-limit:]

