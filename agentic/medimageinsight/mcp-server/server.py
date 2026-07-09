# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""MedImageInsight MCP server.

Exposes MedImageInsight zero-shot classification and adapter inference as MCP tools.

Run with:
    uv run --directory <path-to-this-dir> python server.py

Env vars (set in .env alongside this file):
    MI2_MODEL_ENDPOINT    Full AzureML online-endpoint resource ID for MedImageInsight.
                          Short names are not supported.
"""

import functools
from pathlib import Path
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP

import adapter
import config
from errors import ToolRuntimeError
from mi2_client import MI2Client
import zero_shot

mcp = FastMCP("medimageinsight")


class Clients:
    def __init__(self) -> None:
        self._mi2_client: MI2Client | None = None

    def _build_client(self, settings: dict) -> MI2Client:
        self._mi2_client = config.build_client(settings)
        return self._mi2_client

    @property
    def mi2_client(self) -> MI2Client:
        if self._mi2_client is None:
            self._build_client(config.resolve_settings())
        return self._mi2_client

    def setup(
        self,
        endpoint: str | None = None,
        env_file: str | None = None,
    ) -> MI2Client:
        kwargs = {"MI2_MODEL_ENDPOINT": endpoint} if endpoint else {}
        return self._build_client(config.resolve_settings(env_file=env_file, **kwargs))

    @staticmethod
    def handle_tool_errors(func):
        """Let failures raise so the MCP host flags the tool call as failed."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ToolRuntimeError:
                raise
            except Exception as exc:  # slopcop: ignore[no-broad-except] -- wrap unexpected as a clear tool error
                raise ToolRuntimeError(
                    f"Unexpected error ({type(exc).__name__}): {exc}"
                ) from exc

        return wrapper


clients = Clients()


def _endpoint_host(endpoint_uri: str) -> str:
    parsed = urlparse(endpoint_uri)
    return parsed.netloc or endpoint_uri


@mcp.tool(
    description=(
        "Configure MedImageInsight from the server environment "
        "(MI2_MODEL_ENDPOINT), an env file, or an endpoint resource ID. "
        "Call this before classification tools."
    )
)
@Clients.handle_tool_errors
def setup(
    endpoint: str | None = None,
    env_file: str | None = None,
) -> str:
    """Configure MI2 and return a friendly setup result message."""
    configured_client = clients.setup(endpoint=endpoint, env_file=env_file)
    host = _endpoint_host(configured_client.scoring_uri)
    return f"MedImageInsight configured for {host}. All tools are ready."


@mcp.tool(description=zero_shot.label_examples.__doc__)
@Clients.handle_tool_errors
def zeroshot_label_examples(
    group: str | None = None,
    regex: str | None = None,
) -> list[str] | str:
    return zero_shot.label_examples(group=group, regex=regex)


def _require_image(file_path: str) -> None:
    """Raise a clear error unless the path is an http(s) URL or an existing file."""
    if file_path.startswith(("http://", "https://")):
        return
    if not Path(file_path).is_file():
        raise ToolRuntimeError(
            f"Couldn't find image file: {file_path}. If you passed a relative path, try an absolute path (or an http(s) URL)."
        )


@mcp.tool(description=zero_shot.classify.__doc__)
@Clients.handle_tool_errors
def zeroshot_classify(file_path: str, labels: list[str]) -> dict[str, float] | str:
    _require_image(file_path)
    return zero_shot.classify(file_path, labels, mi2_client=clients.mi2_client)


@mcp.tool(description=adapter.classify.__doc__)
@Clients.handle_tool_errors
def adapter_classify(file_path: str, head: str = "mlp") -> dict[str, str] | str:
    _require_image(file_path)
    return adapter.classify(file_path, head=head, mi2_client=clients.mi2_client)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
