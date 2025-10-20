"""Helpers for tools."""

from strands.tools.decorator import tool


# https://github.com/strands-agents/sdk-python/issues/998
@tool(name="noop", description="This is a fake tool that MUST be completely ignored.")
def _noop_tool() -> None:
    """No-op tool to satisfy tool spec requirement when tool messages are present.

    Some model providers (e.g., Bedrock) will return an error response if tool uses and tool results are present in
    messages without any tool specs configured. Consequently, if the summarization agent has no registered tools,
    summarization will fail. As a workaround, we register the no-op tool.
    """
    pass
