"""
Helper module that exposes the API key used by the notebooks.

Instead of storing the key in source control, read it from the OPENAI_API_KEY
environment variable. Configure the variable locally (e.g. direnv, uvicorn
env, shell profile) or load it from a .env file in your IDE.
"""

import os

openai = os.getenv("OPENAI_API_KEY")

if not openai:
    raise RuntimeError(
        "Set the OPENAI_API_KEY environment variable before running the notebooks."
    )
