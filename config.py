"""
Central config. All values are overridable via environment variables so the
same code runs in local dev (no keys/services) and in a real deployment.
"""
import os

SAMPLE_RATE = int(os.environ.get("AURATUNE_SAMPLE_RATE", 44100))

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")  # optional -- see agents/llm_client.py
MONGO_URI = os.environ.get("MONGO_URI")                  # optional -- see data/db.py
DEMUCS_MODEL = os.environ.get("AURATUNE_DEMUCS_MODEL", "htdemucs")

# Claude models. Text model = command parsing + explanations; vision model =
# reading an EQ-app screenshot (perception/eq_app_reader.py). Both overridable.
LLM_MODEL = os.environ.get("AURATUNE_LLM_MODEL", "claude-sonnet-5")
VISION_MODEL = os.environ.get("AURATUNE_VISION_MODEL", "claude-sonnet-5")

# Screenshot reader also supports Google Gemini (free tier) -- set
# GEMINI_API_KEY / GOOGLE_API_KEY. Preferred over Claude when present.
GEMINI_MODEL = os.environ.get("AURATUNE_GEMINI_MODEL", "gemini-2.0-flash")

# Ear-safety cap applied in eq_decision_agent.py -- no single band moves
# further than this in one adaptation step, regardless of context+command.
MAX_GAIN_DB = float(os.environ.get("AURATUNE_MAX_GAIN_DB", 12.0))
