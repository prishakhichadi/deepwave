"""Central configuration for DEEPWAVE. Loads environment variables once (via python-dotenv)
and exposes typed settings used across all LangGraph nodes and the FastAPI server. Centralizing
this here means model names / temperatures / API keys are changed in one place instead of being
hardcoded inside every node file."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load variables from a .env file in the project root, if present.
# Does nothing (silently) if no .env file exists — real env vars still work.
load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ModelConfig:
    """Model + temperature for a single LangGraph node."""
    model: str
    temperature: float


@dataclass(frozen=True)
class Settings:
    # --- API credentials ---
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # --- Per-node model configuration ---
    # Cheaper/faster model for structured classification & planning tasks.
    classifier_model: ModelConfig = ModelConfig(
        model=os.getenv("DEEPWAVE_CLASSIFIER_MODEL", "gpt-4o-mini"), temperature=0.0
    )
    planner_model: ModelConfig = ModelConfig(
        model=os.getenv("DEEPWAVE_PLANNER_MODEL", "gpt-4o-mini"), temperature=0.1
    )
    # Stronger model for code generation & critique — correctness matters more here.
    rewriter_model: ModelConfig = ModelConfig(
        model=os.getenv("DEEPWAVE_REWRITER_MODEL", "gpt-4o"), temperature=0.2
    )
    critic_model: ModelConfig = ModelConfig(
        model=os.getenv("DEEPWAVE_CRITIC_MODEL", "gpt-4o"), temperature=0.0
    )

    # --- Graph behavior ---
    confidence_threshold: float = float(os.getenv("DEEPWAVE_CONFIDENCE_THRESHOLD", "0.75"))
    max_iterations: int = int(os.getenv("DEEPWAVE_MAX_ITERATIONS", "3"))

    # --- Server ---
    cors_origins: tuple = tuple(
        origin.strip()
        for origin in os.getenv(
            "DEEPWAVE_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
        ).split(",")
        if origin.strip()
    )

    # --- Misc ---
    debug: bool = _get_bool("DEEPWAVE_DEBUG", False)

    def require_api_key(self) -> None:
        """Raises a clear error if no OpenAI API key is configured. Call this at startup
        (main.py / server.py) rather than letting LangChain raise an opaque error deep
        inside a node."""
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill in your key, "
                "or export OPENAI_API_KEY in your shell."
            )


settings = Settings()