"""Helpers for tools."""

from ..tools.decorator import tool
from ..types.content import ContentBlock


# https://github.com/strands-agents/sdk-python/issues/998
@tool(name="noop", description="This is a fake tool that MUST be completely ignored.")
def noop_tool() -> None:
    """No-op tool to satisfy tool spec requirement when tool messages are present.

    Some model providers (e.g., Bedrock) will return an error response if tool uses and tool results are present in
    messages without any tool specs configured. Consequently, if the summarization agent has no registered tools,
    summarization will fail. As a workaround, we register the no-op tool.
    """
    pass


def generate_missing_tool_result_content(tool_use_ids: list[str]) -> list[ContentBlock]:
    """Generate ToolResult content blocks for orphaned ToolUse message."""
    return [
        {
            "toolResult": {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": "Tool was interrupted."}],
            }
        }
        for tool_use_id in tool_use_ids
    ]


def generate_missing_tool_use_content(tool_result_ids: list[str]) -> list[ContentBlock]:
    """Generate ToolUse content blocks for orphaned ToolResult message.

    Args:
        tool_result_ids: List of toolUseIds from orphaned toolResult blocks

    Returns:
        List of ContentBlock dictionaries containing dummy toolUse blocks
    """
    return [
        {
            "toolUse": {
                "toolUseId": tool_use_id,
                "name": "unknown_tool",
                "input": {"error": "toolUse is missing. Ignore."},
            }
        }
        for tool_use_id in tool_result_ids
    ]
