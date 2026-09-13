"""Reads API keys from the environment.

.env_example ships with placeholder values (e.g. "your_x_bearer_token_here")
so the file documents what each key is for. A key copied verbatim from
.env_example into .env is still unconfigured -- get_configured_env treats it
the same as an unset variable instead of handing the literal placeholder
string to whatever client reads it.
"""

import os
import sys

PLACEHOLDER_VALUES = {
    "GOOGLE_API_KEY": "your_google_api_key_here",
    "X_BEARER_TOKEN": "your_x_bearer_token_here",
    "TAVILY_API_KEY": "your_tavily_api_key_here",
}


def get_configured_env(name: str) -> str | None:
    value = os.getenv(name)
    if not value or value == PLACEHOLDER_VALUES.get(name):
        return None
    return value


def require_google_api_key() -> None:
    """Exits the process immediately when GOOGLE_API_KEY is unset or still
    the .env_example placeholder. Every entry point that talks to Gemini
    needs a real key; failing fast here beats a confusing error surfacing
    later from inside langchain_google_genai.
    """
    if get_configured_env("GOOGLE_API_KEY") is None:
        sys.exit(
            "GOOGLE_API_KEY is not configured (unset, or still the .env_example "
            "placeholder). Copy .env_example to .env and set a real key."
        )
