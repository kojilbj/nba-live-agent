"""Reads optional API keys from the environment.

.env_example ships with placeholder values (e.g. "your_x_bearer_token_here")
so the file documents what each key is for. A key copied verbatim from
.env_example into .env is still unconfigured -- get_configured_env treats it
the same as an unset variable instead of handing the literal placeholder
string to whatever client reads it.
"""

import os

PLACEHOLDER_VALUES = {
    "X_BEARER_TOKEN": "your_x_bearer_token_here",
}


def get_configured_env(name: str) -> str | None:
    value = os.getenv(name)
    if not value or value == PLACEHOLDER_VALUES.get(name):
        return None
    return value
