"""
Central configuration for the Incident Response Agent backend.

All values are read from environment variables (see .env.example at the repo
root). Nothing here should ever hard-code a real API key.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the repository root.
# config.py is:
#   repo/backend/app/config.py
# so parents[2] = repo root.
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


class Settings:
    # --- Groq (LLM) ---
    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
    GROQ_BASE_URL: str = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    GROQ_MODEL_PRIMARY: str = os.environ.get("GROQ_MODEL_PRIMARY", "openai/gpt-oss-120b")
    GROQ_MODEL_FALLBACK: str = os.environ.get("GROQ_MODEL_FALLBACK", "qwen/qwen3-32b")

    # --- Hindsight Cloud (memory layer) ---
    HINDSIGHT_API_KEY: str = os.environ.get("HINDSIGHT_API_KEY", "")
    HINDSIGHT_BASE_URL: str = os.environ.get("HINDSIGHT_BASE_URL", "https://api.hindsight.vectorize.io")

    # Project prefix used to namespace the three memory banks, e.g.
    # "incident-agent-episodic", "incident-agent-semantic", "incident-agent-procedural"
    HINDSIGHT_PROJECT: str = os.environ.get("HINDSIGHT_PROJECT", "incident-agent")

    # --- Consolidation / promotion tuning ---
    # Run a semantic-consolidation pass every N newly-resolved incidents.
    CONSOLIDATION_EVERY_N: int = int(os.environ.get("CONSOLIDATION_EVERY_N", "6"))
    # A root-cause family needs at least this many occurrences inside the
    # consolidation window before we bother writing a semantic pattern memory.
    CONSOLIDATION_MIN_OCCURRENCES: int = int(os.environ.get("CONSOLIDATION_MIN_OCCURRENCES", "3"))

    # A fix needs at least this many confirmed attempts, and at least this
    # success rate, before it gets promoted to / refreshed in procedural memory.
    PROCEDURAL_MIN_ATTEMPTS: int = int(os.environ.get("PROCEDURAL_MIN_ATTEMPTS", "2"))
    PROCEDURAL_MIN_SUCCESS_RATE: float = float(os.environ.get("PROCEDURAL_MIN_SUCCESS_RATE", "0.66"))

    # --- Recall relevance (what is allowed to count as "why this recommendation") ---
    # Absolute floor for a Hindsight recall score. Provenance alone is not
    # enough: a resolved incident about a DIFFERENT problem is historically
    # valid but is not evidence for this incident. Results below the floor are
    # dropped BEFORE they reach the model or the UI, and counted in
    # memory_trace.low_relevance so the drop stays visible.
    RELEVANCE_MIN_SCORE: float = float(os.environ.get("RELEVANCE_MIN_SCORE", "0.35"))
    # Per-search guard for score scales where a good match scores far below 1.0:
    # also drop anything below this fraction of the best score in the same
    # search, so a batch of uniformly-low scores never loses everything.
    RELEVANCE_RELATIVE_RATIO: float = float(os.environ.get("RELEVANCE_RELATIVE_RATIO", "0.5"))
    # Cap on rendered "why this recommendation" bullets (best match first).
    # The full hit list stays available under HISTORICAL EVIDENCE.
    EVIDENCE_MAX_LINES: int = int(os.environ.get("EVIDENCE_MAX_LINES", "6"))

    # --- Local bookkeeping DB (sqlite) ---
    DB_PATH: str = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "..", "incident_agent.db"))

    # --- CORS (so the Streamlit app, usually on :8501, can call the API) ---
    CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")


settings = Settings()


def validate_settings() -> list[str]:
    """Returns a list of human-readable warnings for missing required config."""
    warnings = []
    if not settings.GROQ_API_KEY:
        warnings.append("GROQ_API_KEY is not set — diagnosis calls will fail.")
    if not settings.HINDSIGHT_API_KEY:
        warnings.append("HINDSIGHT_API_KEY is not set — memory retain/recall calls will fail.")
    return warnings
