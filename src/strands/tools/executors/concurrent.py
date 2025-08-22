"""Concurrent tool executor implementation."""

import asyncio
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from ...telemetry.metrics import Trace
from ...types.tools import ToolGenerator, ToolResult, ToolUse
from ._executor import ToolExecutor

if TYPE_CHECKING:  # pragma: no cover
    from ...agent import Agent


class ConcurrentToolExecutor(ToolExecutor):
    """Concurrent tool executor."""

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
        """Execute tools concurrently.

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
        task_queue: asyncio.Queue[tuple[int, Any]] = asyncio.Queue()
        task_events = [asyncio.Event() for _ in tool_uses]
        stop_event = object()

        tasks = [
            asyncio.create_task(
                self._task(
                    agent,
                    tool_use,
                    tool_results,
                    cycle_trace,
                    cycle_span,
                    invocation_state,
                    task_id,
                    task_queue,
                    task_events[task_id],
                    stop_event,
                )
            )
            for task_id, tool_use in enumerate(tool_uses)
        ]

        task_count = len(tasks)
        while task_count:
            task_id, event = await task_queue.get()
            if event is stop_event:
                task_count -= 1
                continue

            yield event
            task_events[task_id].set()

        asyncio.gather(*tasks)

    async def _task(
        self,
        agent: "Agent",
        tool_use: ToolUse,
        tool_results: list[ToolResult],
        cycle_trace: Trace,
        cycle_span: Any,
        invocation_state: dict[str, Any],
        task_id: int,
        task_queue: asyncio.Queue,
        task_event: asyncio.Event,
        stop_event: object,
    ) -> None:
        """Execute a single tool and put results in the task queue.

        Args:
            agent: The agent executing the tool.
            tool_use: Tool use metadata and inputs.
            tool_results: List of tool results from each tool execution.
            cycle_trace: Trace object for the current event loop cycle.
            cycle_span: Span object for tracing the cycle.
            invocation_state: Context for tool execution.
            task_id: Unique identifier for this task.
            task_queue: Queue to put tool events into.
            task_event: Event to signal when task can continue.
            stop_event: Sentinel object to signal task completion.
        """
        try:
            events = ToolExecutor._stream_with_trace(
                agent, tool_use, tool_results, cycle_trace, cycle_span, invocation_state
            )
            async for event in events:
                task_queue.put_nowait((task_id, event))
                await task_event.wait()
                task_event.clear()

        finally:
            task_queue.put_nowait((task_id, stop_event))
