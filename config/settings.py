"""Central configuration for DEEPWAVE. Loads environment variables once (via python-dotenv)
and exposes typed settings used across all LangGraph nodes and the FastAPI server. Centralizing
this here means model names / temperatures / API keys are changed in one place instead of being
hardcoded inside every node file."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

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
    """Model + temperature for a single LangGraph node. `groq_model` is the equivalent
    model to use when DEEPWAVE_LLM_PROVIDER=groq, since Groq doesn't host OpenAI's models."""
    model: str
    temperature: float
    groq_model: str


@dataclass(frozen=True)
class Settings:
    # --- Provider selection ---
    # "openai" (paid, no free tier as of mid-2025) or "groq" (free tier, OpenAI-compatible API,
    # good for testing the pipeline end-to-end at zero cost).
    llm_provider: str = os.getenv("DEEPWAVE_LLM_PROVIDER", "openai").strip().lower()

    # --- API credentials ---
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")

    # --- Per-node model configuration ---
    # Cheaper/faster model for structured classification & planning tasks.
    classifier_model: ModelConfig = ModelConfig(
        model=os.getenv("DEEPWAVE_CLASSIFIER_MODEL", "gpt-4o-mini"), temperature=0.0,
        groq_model=os.getenv("DEEPWAVE_GROQ_CLASSIFIER_MODEL", "llama-3.3-70b-versatile"),
    )
    planner_model: ModelConfig = ModelConfig(
        model=os.getenv("DEEPWAVE_PLANNER_MODEL", "gpt-4o-mini"), temperature=0.1,
        groq_model=os.getenv("DEEPWAVE_GROQ_PLANNER_MODEL", "llama-3.3-70b-versatile"),
    )
    # Stronger model for code generation & critique — correctness matters more here.
    rewriter_model: ModelConfig = ModelConfig(
        model=os.getenv("DEEPWAVE_REWRITER_MODEL", "gpt-4o"), temperature=0.2,
        groq_model=os.getenv("DEEPWAVE_GROQ_REWRITER_MODEL", "llama-3.3-70b-versatile"),
    )
    critic_model: ModelConfig = ModelConfig(
        model=os.getenv("DEEPWAVE_CRITIC_MODEL", "gpt-4o"), temperature=0.0,
        groq_model=os.getenv("DEEPWAVE_GROQ_CRITIC_MODEL", "llama-3.3-70b-versatile"),
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
        """Raises a clear error if the API key for the selected provider isn't configured.
        Call this at startup (main.py / server.py) rather than letting LangChain raise an
        opaque error deep inside a node."""
        if self.llm_provider == "groq":
            if not self.groq_api_key:
                raise RuntimeError(
                    "DEEPWAVE_LLM_PROVIDER=groq but GROQ_API_KEY is not set. Get a free key at "
                    "https://console.groq.com and add it to .env."
                )
        else:
            if not self.openai_api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Copy .env.example to .env and fill in your key, "
                    "or export OPENAI_API_KEY in your shell. To test for free instead, set "
                    "DEEPWAVE_LLM_PROVIDER=groq and GROQ_API_KEY (see .env.example)."
                )

    def build_llm(self, cfg: ModelConfig) -> ChatOpenAI:
        """Builds the chat model for a node, honoring the selected provider. Groq exposes an
        OpenAI-compatible /v1 endpoint, so this is a base_url swap rather than a different
        LangChain integration."""
        if self.llm_provider == "groq":
            return ChatOpenAI(
                model=cfg.groq_model,
                temperature=cfg.temperature,
                api_key=self.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
            )
        return ChatOpenAI(
            model=cfg.model,
            temperature=cfg.temperature,
            api_key=self.openai_api_key or None,
        )

    @property
    def structured_output_method(self) -> str:
        """OpenAI's strict json_schema mode is proprietary — other OpenAI-compatible
        providers (Groq included) need the more broadly-supported function_calling
        method for with_structured_output() to work reliably."""
        return "json_schema" if self.llm_provider == "openai" else "function_calling"


settings = Settings()
