"""Sequential tool executor implementation."""

from typing import TYPE_CHECKING, Any

from typing_extensions import override

from ...experimental.tools.executors import Executor as SAExecutor
from ...types.tools import ToolGenerator, ToolResult, ToolUse

if TYPE_CHECKING:  # pragma: no cover
    from ...agent import Agent


class Executor(SAExecutor):
    """Sequential tool executor."""

    @override
    async def execute(
        self, agent: "Agent", tool_uses: list[ToolUse], tool_results: list[ToolResult], invocation_state: dict[str, Any]
    ) -> ToolGenerator:
        """Execute tools sequentially.

        Args:
            agent: The agent for which tools are being executed.
            tool_uses: Metadata and inputs for the tools to be executed.
            tool_results: List of tool results from each tool execution.
            invocation_state: Context for the tool invocation.

        Yields:
            Events from the tool execution stream.
        """
        for tool_use in tool_uses:
            async for event in self.stream(agent, tool_use, tool_results, invocation_state):
                yield event
