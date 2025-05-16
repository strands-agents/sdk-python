"""Tool execution functionality for the event loop."""

import logging
import time
from concurrent.futures import TimeoutError
from typing import Any, Callable, List, Optional, Tuple

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
    tool_uses: List[ToolUse],
    event_loop_metrics: EventLoopMetrics,
    request_state: Any,
    invalid_tool_use_ids: List[str],
    tool_results: List[ToolResult],
    cycle_trace: Trace,
    parent_span: Optional[trace.Span] = None,
    parallel_tool_executor: Optional[ParallelToolExecutorInterface] = None,
) -> bool:
    """Execute tools either in parallel or sequentially.

    Args:
        handler: Tool handler processing function.
        tool_uses: List of tool uses to execute.
        event_loop_metrics: Metrics collection object.
        request_state: Current request state.
        invalid_tool_use_ids: List of invalid tool use IDs.
        tool_results: List to populate with tool results.
        cycle_trace: Parent trace for the current cycle.
        parent_span: Parent span for the current cycle.
        parallel_tool_executor: Optional executor for parallel processing.

    Returns:
        bool: True if any tool failed, False otherwise.
    """

    def _handle_tool_execution(tool: ToolUse) -> Tuple[bool, Optional[ToolResult]]:
        result = None
        tool_succeeded = False

        tracer = get_tracer()
        tool_call_span = tracer.start_tool_call_span(tool, parent_span)

        try:
            if "toolUseId" not in tool or tool["toolUseId"] not in invalid_tool_use_ids:
                tool_name = tool["name"]
                tool_trace = Trace(f"Tool: {tool_name}", parent_id=cycle_trace.id, raw_name=tool_name)
                tool_start_time = time.time()
                result = handler(tool)
                tool_success = result.get("status") == "success"
                if tool_success:
                    tool_succeeded = True

                tool_duration = time.time() - tool_start_time
                message = Message(role="user", content=[{"toolResult": result}])
                event_loop_metrics.add_tool_usage(tool, tool_duration, tool_trace, tool_success, message)
                cycle_trace.add_child(tool_trace)

            if tool_call_span:
                tracer.end_tool_call_span(tool_call_span, result)
        except Exception as e:
            if tool_call_span:
                tracer.end_span_with_error(tool_call_span, str(e), e)

        return tool_succeeded, result

    any_tool_failed = False
    if parallel_tool_executor:
        logger.debug(
            "tool_count=<%s>, tool_executor=<%s> | executing tools in parallel",
            len(tool_uses),
            type(parallel_tool_executor).__name__,
        )
        # Submit all tasks with their associated tools
        future_to_tool = {
            parallel_tool_executor.submit(_handle_tool_execution, tool_use): tool_use for tool_use in tool_uses
        }
        logger.debug("tool_count=<%s> | submitted tasks to parallel executor", len(tool_uses))

        # Collect results truly in parallel using the provided executor's as_completed method
        completed_results = []
        try:
            for future in parallel_tool_executor.as_completed(future_to_tool):
                try:
                    succeeded, result = future.result()
                    if result is not None:
                        completed_results.append(result)
                    if not succeeded:
                        any_tool_failed = True
                except Exception as e:
                    tool = future_to_tool[future]
                    logger.debug("tool_name=<%s> | tool execution failed | %s", tool["name"], e)
                    any_tool_failed = True
        except TimeoutError:
            logger.error("timeout_seconds=<%s> | parallel tool execution timed out", parallel_tool_executor.timeout)
            # Process any completed tasks
            for future in future_to_tool:
                if future.done():  # type: ignore
                    try:
                        succeeded, result = future.result(timeout=0)
                        if result is not None:
                            completed_results.append(result)
                    except Exception as tool_e:
                        tool = future_to_tool[future]
                        logger.debug("tool_name=<%s> | tool execution failed | %s", tool["name"], tool_e)
                else:
                    # This future didn't complete within the timeout
                    tool = future_to_tool[future]
                    logger.debug("tool_name=<%s> | tool execution timed out", tool["name"])

            any_tool_failed = True

        # Add completed results to tool_results
        tool_results.extend(completed_results)
    else:
        # Sequential execution fallback
        for tool_use in tool_uses:
            succeeded, result = _handle_tool_execution(tool_use)
            if result is not None:
                tool_results.append(result)
            if not succeeded:
                any_tool_failed = True

    return any_tool_failed


def validate_and_prepare_tools(
    message: Message,
    tool_uses: List[ToolUse],
    tool_results: List[ToolResult],
    invalid_tool_use_ids: List[str],
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
