"""Sequential tool executor implementation."""

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import override

from ...telemetry.metrics import Trace
from ...types.tools import ToolGenerator, ToolResult, ToolUse
from ._executor import Executor as SAExecutor

if TYPE_CHECKING:  # pragma: no cover
    from ...agent import Agent


class Executor(SAExecutor):
    """Sequential tool executor."""

    def __init__(self, thread_pool: ThreadPoolExecutor | Literal["asyncio"] | None = "asyncio"):
        """Initialize the executor.

        Args:
            thread_pool: Thread pool configuration for synchronous tools.

                - "asyncio" (default): Use the asyncio thread pool
                - ThreadPoolExecutor: Use the provided custom thread pool
                - None: Run sync tools in the main thread (i.e., blocking)
        """
        self._thread_pool = thread_pool

    @override
    async def _execute(
        self,
        agent: "Agent",
        tool_uses: list[ToolUse],
        tool_results: list[ToolResult],
        cycle_trace: Trace,
        cycle_span: Any,
        invocation_state: dict[str, Any],
    ) -> ToolGenerator:
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
            events = SAExecutor._stream_with_trace(
                agent, tool_use, tool_results, cycle_trace, cycle_span, invocation_state, thread_pool=self._thread_pool
            )
            async for event in events:
                yield event
