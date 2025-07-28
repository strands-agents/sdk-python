"""Tool execution functionality for the event loop."""

import asyncio
import logging
import time
from typing import Any, Optional, cast

from opentelemetry import trace as trace_api

from ..telemetry.metrics import EventLoopMetrics, Trace
from ..telemetry.tracer import get_tracer
from ..tools.tools import InvalidToolUseNameException, validate_tool_use
from ..types.content import Message
from ..types.tools import RunToolHandler, ToolGenerator, ToolResult, ToolUse

logger = logging.getLogger(__name__)


async def run_tools(
    handler: RunToolHandler,
    tool_uses: list[ToolUse],
    event_loop_metrics: EventLoopMetrics,
    invalid_tool_use_ids: list[str],
    tool_results: list[ToolResult],
    cycle_trace: Trace,
    parent_span: Optional[trace_api.Span] = None,
) -> ToolGenerator:
    """Execute tools concurrently.

    Args:
        handler: Tool handler processing function.
        tool_uses: List of tool uses to execute.
        event_loop_metrics: Metrics collection object.
        invalid_tool_use_ids: List of invalid tool use IDs.
        tool_results: List to populate with tool results.
        cycle_trace: Parent trace for the current cycle.
        parent_span: Parent span for the current cycle.

    Yields:
        Events of the tool stream. Tool results are appended to `tool_results`.
    """

    async def work(
        tool_use: ToolUse,
        worker_id: int,
        worker_queue: asyncio.Queue,
        worker_event: asyncio.Event,
        stop_event: object,
    ) -> ToolResult:
        tracer = get_tracer()
        tool_call_span = tracer.start_tool_call_span(tool_use, parent_span)

        tool_name = tool_use["name"]
        tool_trace = Trace(f"Tool: {tool_name}", parent_id=cycle_trace.id, raw_name=tool_name)
        tool_start_time = time.time()
        with trace_api.use_span(tool_call_span):
            try:
                async for event in handler(tool_use):
                    worker_queue.put_nowait((worker_id, event))
                    await worker_event.wait()
                    worker_event.clear()

                result = cast(ToolResult, event)
            finally:
                worker_queue.put_nowait((worker_id, stop_event))

            tool_success = result.get("status") == "success"
            tool_duration = time.time() - tool_start_time
            message = Message(role="user", content=[{"toolResult": result}])
            event_loop_metrics.add_tool_usage(tool_use, tool_duration, tool_trace, tool_success, message)
            cycle_trace.add_child(tool_trace)

            tracer.end_tool_call_span(tool_call_span, result)

        return result

    tool_uses = [tool_use for tool_use in tool_uses if tool_use.get("toolUseId") not in invalid_tool_use_ids]
    worker_queue: asyncio.Queue[tuple[int, Any]] = asyncio.Queue()
    worker_events = [asyncio.Event() for _ in tool_uses]
    stop_event = object()

    workers = [
        asyncio.create_task(work(tool_use, worker_id, worker_queue, worker_events[worker_id], stop_event))
        for worker_id, tool_use in enumerate(tool_uses)
    ]

    worker_count = len(workers)
    while worker_count:
        worker_id, event = await worker_queue.get()
        if event is stop_event:
            worker_count -= 1
            continue

        yield event
        worker_events[worker_id].set()

    tool_results.extend([worker.result() for worker in workers])


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
        try:
            validate_tool_use(tool)
        except InvalidToolUseNameException as e:
            # Replace the invalid toolUse name and return invalid name error as ToolResult to the LLM as context
            tool_uses.remove(tool)
            tool["name"] = "INVALID_TOOL_NAME"
            invalid_tool_use_ids.append(tool["toolUseId"])
            tool_uses.append(tool)
            tool_results.append(
                {
                    "toolUseId": tool["toolUseId"],
                    "status": "error",
                    "content": [{"text": f"Error: {str(e)}"}],
                }
            )
