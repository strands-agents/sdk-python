"""Sequential tool executor implementation."""

from typing import TYPE_CHECKING, Any, AsyncGenerator

from typing_extensions import override

from ...telemetry.metrics import Trace
from ...types._events import TypedEvent
from ...types.tools import ToolResult, ToolUse
from ._executor import ToolExecutor

if TYPE_CHECKING:  # pragma: no cover
    from ...agent import Agent


class SequentialToolExecutor(ToolExecutor):
    """Sequential tool executor."""

    @override
    async def _execute(
        self,
        agent: "Agent",
        tool_uses: list[ToolUse],
        tool_results: list[ToolResult],
        cycle_trace: Trace,
        cycle_span: Any,
        invocation_state: dict[str, Any],
    ) -> AsyncGenerator[TypedEvent, None]:
        """Execute tools sequentially.

        Args:
            agent: The agent for which tools are being executed.
            tool_uses: Metadata and inputs for the tools to be executed.
            tool_results: List of tool results from each tool execution.
            cycle_trace: Trace object for the current event loop cycle.
            cycle_span: Span object for tracing the cycle.
            invocation_state: Context for the tool invocation.

        Yields:
            Events from the tool execution stream.
        """
        for tool_use in tool_uses:
            events = ToolExecutor._stream_with_trace(
                agent, tool_use, tool_results, cycle_trace, cycle_span, invocation_state
            )
            async for event in events:
                yield event
