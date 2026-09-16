"""
Central config. All values are overridable via environment variables so the
same code runs in local dev (no keys/services) and in a real deployment.
"""
import os
from pathlib import Path


def _load_dotenv() -> None:
    """Load KEY=value lines from a local .env, without overriding anything
    already in the environment (an explicit `export` always wins).

    Deliberately dependency-free -- same spirit as the rest of the repo's
    optional integrations. .env is gitignored; keys never belong in git.
    """
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

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

# Command parsing + explanations also run on Groq's OpenAI-compatible API
# (set GROQ_API_KEY), for teammates without an Anthropic key. Anthropic is
# preferred when both are set; AURATUNE_LLM_PROVIDER=groq|anthropic|none
# forces one. See agents/llm_client.py.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("AURATUNE_GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL = os.environ.get("AURATUNE_GROQ_BASE_URL",
                               "https://api.groq.com/openai/v1")
LLM_PROVIDER = os.environ.get("AURATUNE_LLM_PROVIDER", "auto").lower()

# Ear-safety cap applied in eq_decision_agent.py -- no single band moves
# further than this in one adaptation step, regardless of context+command.
MAX_GAIN_DB = float(os.environ.get("AURATUNE_MAX_GAIN_DB", 12.0))
