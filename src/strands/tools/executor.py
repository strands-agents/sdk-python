"""Tool execution functionality for the event loop."""

import logging
import queue
import threading
import time
from typing import Any, Callable, Generator, Optional, cast

from opentelemetry import trace

from ..telemetry.metrics import EventLoopMetrics, Trace
from ..telemetry.tracer import get_tracer
from ..tools.tools import InvalidToolUseNameException, validate_tool_use
from ..types.content import Message
from ..types.event_loop import ParallelToolExecutorInterface
from ..types.tools import ToolResult, ToolUse

logger = logging.getLogger(__name__)


def run_tools(
    handler: Callable[[ToolUse], ToolResult],
    tool_uses: list[ToolUse],
    event_loop_metrics: EventLoopMetrics,
    invalid_tool_use_ids: list[str],
    tool_results: list[ToolResult],
    cycle_trace: Trace,
    parent_span: Optional[trace.Span] = None,
    parallel_tool_executor: Optional[ParallelToolExecutorInterface] = None,
) -> Generator[dict[str, Any], None, None]:
    """Execute tools either in parallel or sequentially.

    Args:
        handler: Tool handler processing function.
        tool_uses: List of tool uses to execute.
        event_loop_metrics: Metrics collection object.
        invalid_tool_use_ids: List of invalid tool use IDs.
        tool_results: List to populate with tool results.
        cycle_trace: Parent trace for the current cycle.
        parent_span: Parent span for the current cycle.
        parallel_tool_executor: Optional executor for parallel processing.

    Yields:
        Events of the tool invocations. Tool results are appended to `tool_results`.
    """

    def handle(tool: ToolUse) -> Generator[dict[str, Any], None, ToolResult]:
        tracer = get_tracer()
        tool_call_span = tracer.start_tool_call_span(tool, parent_span)

        tool_name = tool["name"]
        tool_trace = Trace(f"Tool: {tool_name}", parent_id=cycle_trace.id, raw_name=tool_name)
        tool_start_time = time.time()

        result = handler(tool)
        yield {"result": result}  # Placeholder until handler becomes a generator from which we can yield from

        tool_success = result.get("status") == "success"
        tool_duration = time.time() - tool_start_time
        message = Message(role="user", content=[{"toolResult": result}])
        event_loop_metrics.add_tool_usage(tool, tool_duration, tool_trace, tool_success, message)
        cycle_trace.add_child(tool_trace)

        if tool_call_span:
            tracer.end_tool_call_span(tool_call_span, result)

        return result

    def work(
        tool: ToolUse,
        worker_id: int,
        worker_queue: queue.Queue,
        worker_event: threading.Event,
    ) -> ToolResult:
        events = handle(tool)

        while True:
            try:
                event = next(events)
                worker_queue.put((worker_id, event))
                worker_event.wait()

            except StopIteration as stop:
                return cast(ToolResult, stop.value)

    tool_uses = [tool_use for tool_use in tool_uses if tool_use.get("toolUseId") not in invalid_tool_use_ids]

    if parallel_tool_executor:
        logger.debug(
            "tool_count=<%s>, tool_executor=<%s> | executing tools in parallel",
            len(tool_uses),
            type(parallel_tool_executor).__name__,
        )

        worker_queue: queue.Queue[tuple[int, dict[str, Any]]] = queue.Queue()
        worker_events = [threading.Event() for _ in range(len(tool_uses))]

        workers = [
            parallel_tool_executor.submit(work, tool_use, worker_id, worker_queue, worker_events[worker_id])
            for worker_id, tool_use in enumerate(tool_uses)
        ]
        logger.debug("tool_count=<%s> | submitted tasks to parallel executor", len(tool_uses))

        while not all(worker.done() for worker in workers):
            if not worker_queue.empty():
                worker_id, event = worker_queue.get()
                yield event
                worker_events[worker_id].set()

        tool_results.extend([worker.result() for worker in workers])

    else:
        # Sequential execution fallback
        for tool_use in tool_uses:
            result = yield from handle(tool_use)
            tool_results.append(result)


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
