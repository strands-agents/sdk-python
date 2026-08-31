"""Shared types and constants for the tool_registry tool."""

from __future__ import annotations

import re
from typing import Any, Literal, TypedDict

from typing_extensions import NotRequired

# Provider-accepted tool name: leading letter/underscore, then letters/digits/underscore,
# capped at 64 characters. Stricter than the underlying registry (which also allows '-')
# because the dynamically-registered tool name is echoed into user-visible spec strings
# and log messages; disallowing '-' keeps every generated identifier a legal Python name.
TOOL_NAME_PATTERN: re.Pattern[str] = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,63}$")
"""Regex enforced on all dynamically-registered tool names."""

MAX_DYNAMIC_TOOLS: int = 32
"""Maximum number of tools a single tool_registry instance may register."""

TOOL_REGISTRY_DESCRIPTION: str = (
    "Manage the agent's tool registry at runtime. Supports operations to `list` "
    "currently registered tools, `create` a new binding to an already-connected "
    "MCP server tool, `update` an existing binding, and `delete` a previously "
    "registered binding. Registration is limited to remote MCP tools; loading "
    "tools from a file path or inline source is intentionally not supported."
)


class RegisteredTool(TypedDict):
    """A tool entry in a `list` response.

    Attributes:
        name: The tool name as visible to the agent.
        description: The tool description as visible to the model.
        input_schema: The tool's JSON input schema. Omitted when the tool
            does not declare one.
        registered_by_tool_registry: True when this tool was registered via
            the tool_registry tool (i.e. it can be updated/deleted by this
            tool). False for developer-registered tools.
    """

    name: str
    description: str
    input_schema: NotRequired[dict[str, Any]]
    registered_by_tool_registry: bool


class ListResult(TypedDict):
    """Result payload for the `list` operation.

    Attributes:
        tools: All tools currently on the registry.
        dynamic_count: Number of tools registered by this tool_registry instance.
            Tools with ``registered_by_tool_registry=True`` in the ``tools`` list
            can be updated or deleted via this tool.
        dynamic_limit: Maximum number of tools this instance may register. When
            ``dynamic_count == dynamic_limit``, delete an existing tool before
            creating a new one.
    """

    tools: list[RegisteredTool]
    dynamic_count: int
    dynamic_limit: int


class MutationResult(TypedDict):
    """Result payload for `create`, `update`, and `delete` operations.

    Attributes:
        operation: The operation that was performed.
        name: The local tool name the operation acted on.
        dynamic_count: Number of tools registered by this tool_registry instance
            after the operation completes.
    """

    operation: Literal["create", "update", "delete"]
    name: str
    dynamic_count: int


class ToolRegistryError(ValueError):
    """Raised for validation failures inside the tool_registry tool."""
