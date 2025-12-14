"""
Helper module that exposes the API key used by the notebooks.

It first checks the OPENAI_API_KEY environment variable. If that is unset, it
looks for a sibling .env file (ignored by Git) and loads the key from there.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

ENV_VAR = "OPENAI_API_KEY"


def _load_from_env_file() -> Optional[str]:
    """Load the API key from ../.env if present."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return None

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == ENV_VAR:
            cleaned = value.strip().strip('"').strip("'")
            os.environ.setdefault(ENV_VAR, cleaned)
            return cleaned

    return None


openai: Optional[str] = os.getenv(ENV_VAR) or _load_from_env_file()

if not openai:
    raise RuntimeError(
        "Set the OPENAI_API_KEY environment variable or define it inside .env before running the notebooks."
    )
