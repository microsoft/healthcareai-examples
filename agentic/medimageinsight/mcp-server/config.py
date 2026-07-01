# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Resolve MedImageInsight client settings from env vars, an env file, or kwargs.

Deliberately free of any MCP/FastMCP imports so non-server callers (e.g.
smoketest.py) can reuse the exact same resolution without pulling in the
server-side logging stack.
"""

import os
from pathlib import Path

from errors import ToolRuntimeError
from mi2_client import MI2Client

_TRUE_VALUES = {"1", "true", "yes"}


def settings_from(values: dict[str, str]) -> dict:
    """Pull the MI2 settings we care about out of a mapping (env vars or kwargs)."""
    return {
        "endpoint": (values.get("MI2_MODEL_ENDPOINT") or "").strip(),
        "exclude_managed_identity": (
            (values.get("AZURE_IDENTITY_DISABLE_MANAGED_IDENTITY") or "")
            .strip()
            .lower()
            in _TRUE_VALUES
        ),
    }


def resolve_settings(env_file: str | None = None, **kwargs: str) -> dict:
    """Resolve settings from explicit kwargs, an env file, or the ambient environment."""
    if kwargs:
        # caller passed explicit values — use those, not the environment.
        return settings_from(kwargs)
    if env_file:
        # reason: load the file so its vars (endpoint + credential flags) are in the env.
        from dotenv import load_dotenv

        path = Path(env_file)
        if not path.is_file():
            raise ToolRuntimeError(
                f"Env file not found: {env_file}. Pass the path to an existing .env file."
            )
        load_dotenv(path, override=True)
    # already-present env (.mcp.json env/envFile or shell), or just-loaded.
    return settings_from(dict(os.environ))


def build_client(settings: dict) -> MI2Client:
    """Build an MI2 client from resolved settings, or raise if no endpoint is set."""
    if not settings["endpoint"]:
        raise ToolRuntimeError(
            "MI2_MODEL_ENDPOINT is not set. Set it in the MCP server config "
            "(env or envFile in .mcp.json), or call setup(env_file=...) or setup(endpoint=...)."
        )
    return MI2Client.from_endpoint(
        settings["endpoint"],
        exclude_managed_identity=settings["exclude_managed_identity"],
    )
