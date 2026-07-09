# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Shared runtime exceptions for MedImageInsight MCP tools."""


class ToolRuntimeError(Exception):
    """An expected, user-facing failure with actionable advice.

    Raise this for known failure modes whose message should be shown to the
    caller verbatim. Anything else that escapes a tool is treated as an
    unexpected error.
    """
