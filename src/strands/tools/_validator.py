"""Tool validation utilities."""

import logging
import re
from typing import Tuple

from ..types.content import Message
from ..types.tools import ToolResult, ToolUse

logger = logging.getLogger(__name__)


def validate_and_prepare_tools(
    message: Message,
    tool_uses: list[ToolUse],
    tool_results: list[ToolResult],
    invalid_tool_use_ids: list[str],
) -> None:
    """Validate tool uses and prepare them for execution.

    Args:
        message: Current message.
        tool_uses: List to populate with tool uses.
        tool_results: List to populate with tool results for invalid tools.
        invalid_tool_use_ids: List to populate with invalid tool use IDs.
    """
    # Extract tool uses from message
    for content in message["content"]:
        if isinstance(content, dict) and "toolUse" in content:
            tool_uses.append(content["toolUse"])

    # Validate tool uses
    # Avoid modifying original `tool_uses` variable during iteration
    tool_uses_copy = tool_uses.copy()
    for tool in tool_uses_copy:
        is_valid, validity_message = check_tool_name_validity(tool)

        if not is_valid:
            logger.warning(validity_message)
            # Return invalid name error as ToolResult to the LLM as context;
            # The replacement of the tool name to INVALID_TOOL_NAME happens in streaming.py now
            tool_uses.remove(tool)
            invalid_tool_use_ids.append(tool["toolUseId"])
            tool_uses.append(tool)
            tool_results.append(
                {
                    "toolUseId": tool["toolUseId"],
                    "status": "error",
                    "content": [{"text": f"Error: {validity_message}"}],
                }
            )


def check_tool_name_validity(tool: ToolUse) -> Tuple[bool, str]:
    """Validate a tool use name."""
    # We need to fix some typing here, because we don't actually expect a ToolUse, but dict[str, Any]
    if "name" not in tool:
        return False, "tool name missing"  # type: ignore[unreachable]

    tool_name = tool["name"]
    tool_name_pattern = r"^[a-zA-Z0-9_\-]{1,}$"
    tool_name_max_length = 64
    valid_name_pattern = bool(re.match(tool_name_pattern, tool_name))
    tool_name_len = len(tool_name)

    if not valid_name_pattern:
        message = f"tool_name=<{tool_name}> | invalid tool name pattern"
        return False, message

    if tool_name_len > tool_name_max_length:
        message = f"tool_name=<{tool_name}>, tool_name_max_length=<{tool_name_max_length}> | invalid tool name length"
        return False, message

    return True, ""
