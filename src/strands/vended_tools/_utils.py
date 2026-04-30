"""Shared utilities for vended tools.

Provides common helper functions used across all vended tools to avoid
code duplication.
"""

from typing import Any

from ..types.tools import ToolContext


def get_tool_config(tool_context: ToolContext, state_key: str) -> dict[str, Any]:
    """Read tool configuration from agent state.

    All vended tools store their configuration in agent state under
    a namespaced key. This helper standardizes the pattern.

    Args:
        tool_context: The tool context providing access to agent state.
        state_key: The state key for the tool's configuration.

    Returns:
        Configuration dict. Empty dict if no config is set.
    """
    return tool_context.agent.state.get(state_key) or {}
