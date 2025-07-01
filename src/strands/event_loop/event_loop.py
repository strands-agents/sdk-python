"""This module implements the central event loop.

The event loop allows agents to:

1. Process conversation messages
2. Execute tools based on model requests
3. Handle errors and recovery strategies
4. Manage recursive execution cycles
"""

import logging
import time
import uuid
from functools import partial
from typing import Any, Generator, Optional, cast

from opentelemetry import trace

from ..telemetry.metrics import EventLoopMetrics, Trace
from ..telemetry.tracer import get_tracer
from ..tools.executor import run_tools, validate_and_prepare_tools
from ..types.content import Message, Messages
from ..types.event_loop import ParallelToolExecutorInterface
from ..types.exceptions import ContextWindowOverflowException, EventLoopException, ModelThrottledException
from ..types.models import Model
from ..types.streaming import Metrics, StopReason
from ..types.tools import ToolConfig, ToolHandler, ToolResult, ToolUse
from .message_processor import clean_orphaned_empty_tool_uses
from .streaming import stream_messages

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 6
INITIAL_DELAY = 4
MAX_DELAY = 240  # 4 minutes


def event_loop_cycle(
    model: Model,
    system_prompt: Optional[str],
    messages: Messages,
    tool_config: Optional[ToolConfig],
    tool_handler: Optional[ToolHandler],
    tool_execution_handler: Optional[ParallelToolExecutorInterface],
    event_loop_metrics: EventLoopMetrics,
    event_loop_parent_span: Optional[trace.Span],
    kwargs: dict[str, Any],
) -> Generator[dict[str, Any], None, None]:
    """Execute a single cycle of the event loop.

    This core function processes a single conversation turn, handling model inference, tool execution, and error
    recovery. It manages the entire lifecycle of a conversation turn, including:

    1. Initializing cycle state and metrics
    2. Checking execution limits
    3. Processing messages with the model
    4. Handling tool execution requests
    5. Managing recursive calls for multi-turn tool interactions
    6. Collecting and reporting metrics
    7. Error handling and recovery

    Args:
        model: Provider for running model inference.
        system_prompt: System prompt instructions for the model.
        messages: Conversation history messages.
        tool_config: Configuration for available tools.
        tool_handler: Handler for executing tools.
        tool_execution_handler: Optional handler for parallel tool execution.
        event_loop_metrics: Metrics tracking object for the event loop.
        event_loop_parent_span: Span for the parent of this event loop.
        kwargs: Additional arguments including:

            - request_state: State maintained across cycles
            - event_loop_cycle_id: Unique ID for this cycle
            - event_loop_cycle_span: Current tracing Span for this cycle

    Yields:
        Model and tool invocation events. The last event is a tuple containing:

            - StopReason: Reason the model stopped generating (e.g., "tool_use")
            - Message: The generated message from the model
            - EventLoopMetrics: Updated metrics for the event loop
            - Any: Updated request state

    Raises:
        EventLoopException: If an error occurs during execution
        ContextWindowOverflowException: If the input is too large for the model
    """
    # Initialize cycle state
    kwargs["event_loop_cycle_id"] = uuid.uuid4()

    # Initialize state and get cycle trace
    if "request_state" not in kwargs:
        kwargs["request_state"] = {}
    attributes = {"event_loop_cycle_id": str(kwargs.get("event_loop_cycle_id"))}
    cycle_start_time, cycle_trace = event_loop_metrics.start_cycle(attributes=attributes)
    kwargs["event_loop_cycle_trace"] = cycle_trace

    yield {"callback": {"start": True}}
    yield {"callback": {"start_event_loop": True}}

    # Create tracer span for this event loop cycle
    tracer = get_tracer()
    cycle_span = tracer.start_event_loop_cycle_span(
        event_loop_kwargs=kwargs, parent_span=event_loop_parent_span, messages=messages
    )
    kwargs["event_loop_cycle_span"] = cycle_span

    # Create a trace for the stream_messages call
    stream_trace = Trace("stream_messages", parent_id=cycle_trace.id)
    cycle_trace.add_child(stream_trace)

    # Clean up orphaned empty tool uses
    clean_orphaned_empty_tool_uses(messages)

    # Process messages with exponential backoff for throttling
    message: Message
    stop_reason: StopReason
    usage: Any
    metrics: Metrics

    # Retry loop for handling throttling exceptions
    current_delay = INITIAL_DELAY
    for attempt in range(MAX_ATTEMPTS):
        model_id = model.config.get("model_id") if hasattr(model, "config") else None
        model_invoke_span = tracer.start_model_invoke_span(
            parent_span=cycle_span,
            messages=messages,
            model_id=model_id,
        )

        try:
            # TODO: To maintain backwards compatability, we need to combine the stream event with kwargs before yielding
            #       to the callback handler. This will be revisited when migrating to strongly typed events.
            for event in stream_messages(model, system_prompt, messages, tool_config):
                if "callback" in event:
                    yield {"callback": {**event["callback"], **(kwargs if "delta" in event["callback"] else {})}}

            stop_reason, message, usage, metrics = event["stop"]
            kwargs.setdefault("request_state", {})

            if model_invoke_span:
                tracer.end_model_invoke_span(model_invoke_span, message, usage)
            break  # Success! Break out of retry loop

        except ContextWindowOverflowException as e:
            if model_invoke_span:
                tracer.end_span_with_error(model_invoke_span, str(e), e)
            raise e

        except ModelThrottledException as e:
            if model_invoke_span:
                tracer.end_span_with_error(model_invoke_span, str(e), e)

            if attempt + 1 == MAX_ATTEMPTS:
                yield {"callback": {"force_stop": True, "force_stop_reason": str(e)}}
                raise e

            logger.debug(
                "retry_delay_seconds=<%s>, max_attempts=<%s>, current_attempt=<%s> "
                "| throttling exception encountered "
                "| delaying before next retry",
                current_delay,
                MAX_ATTEMPTS,
                attempt + 1,
            )
            time.sleep(current_delay)
            current_delay = min(current_delay * 2, MAX_DELAY)

            yield {"callback": {"event_loop_throttled_delay": current_delay, **kwargs}}

        except Exception as e:
            if model_invoke_span:
                tracer.end_span_with_error(model_invoke_span, str(e), e)
            raise e

    try:
        # Add message in trace and mark the end of the stream messages trace
        stream_trace.add_message(message)
        stream_trace.end()

        # Add the response message to the conversation
        messages.append(message)
        yield {"callback": {"message": message}}

        # Update metrics
        event_loop_metrics.update_usage(usage)
        event_loop_metrics.update_metrics(metrics)

        # If the model is requesting to use tools
        if stop_reason == "tool_use":
            if not tool_handler:
                raise EventLoopException(
                    Exception("Model requested tool use but no tool handler provided"),
                    kwargs["request_state"],
                )

            if tool_config is None:
                raise EventLoopException(
                    Exception("Model requested tool use but no tool config provided"),
                    kwargs["request_state"],
                )

            # Handle tool execution
            yield from _handle_tool_execution(
                stop_reason,
                message,
                model,
                system_prompt,
                messages,
                tool_config,
                tool_handler,
                tool_execution_handler,
                event_loop_metrics,
                event_loop_parent_span,
                cycle_trace,
                cycle_span,
                cycle_start_time,
                kwargs,
            )
            return

        # End the cycle and return results
        event_loop_metrics.end_cycle(cycle_start_time, cycle_trace, attributes)
        if cycle_span:
            tracer.end_event_loop_cycle_span(
                span=cycle_span,
                message=message,
            )
    except EventLoopException as e:
        if cycle_span:
            tracer.end_span_with_error(cycle_span, str(e), e)

        # Don't yield or log the exception - we already did it when we
        # raised the exception and we don't need that duplication.
        raise
    except ContextWindowOverflowException as e:
        if cycle_span:
            tracer.end_span_with_error(cycle_span, str(e), e)
        raise e
    except Exception as e:
        if cycle_span:
            tracer.end_span_with_error(cycle_span, str(e), e)

        # Handle any other exceptions
        yield {"callback": {"force_stop": True, "force_stop_reason": str(e)}}
        logger.exception("cycle failed")
        raise EventLoopException(e, kwargs["request_state"]) from e

    yield {"stop": (stop_reason, message, event_loop_metrics, kwargs["request_state"])}


def recurse_event_loop(
    model: Model,
    system_prompt: Optional[str],
    messages: Messages,
    tool_config: Optional[ToolConfig],
    tool_handler: Optional[ToolHandler],
    tool_execution_handler: Optional[ParallelToolExecutorInterface],
    event_loop_metrics: EventLoopMetrics,
    event_loop_parent_span: Optional[trace.Span],
    kwargs: dict[str, Any],
) -> Generator[dict[str, Any], None, None]:
    """Make a recursive call to event_loop_cycle with the current state.

    This function is used when the event loop needs to continue processing after tool execution.

    Args:
        model: Provider for running model inference
        system_prompt: System prompt instructions for the model
        messages: Conversation history messages
        tool_config: Configuration for available tools
        tool_handler: Handler for tool execution
        tool_execution_handler: Optional handler for parallel tool execution.
        event_loop_metrics: Metrics tracking object for the event loop.
        event_loop_parent_span: Span for the parent of this event loop.
        kwargs: Arguments to pass through event_loop_cycle


    Yields:
        Results from event_loop_cycle where the last result contains:

            - StopReason: Reason the model stopped generating
            - Message: The generated message from the model
            - EventLoopMetrics: Updated metrics for the event loop
            - Any: Updated request state
    """
    cycle_trace = kwargs["event_loop_cycle_trace"]

    # Recursive call trace
    recursive_trace = Trace("Recursive call", parent_id=cycle_trace.id)
    cycle_trace.add_child(recursive_trace)

    yield {"callback": {"start": True}}
    yield from event_loop_cycle(
        model=model,
        system_prompt=system_prompt,
        messages=messages,
        tool_config=tool_config,
        tool_handler=tool_handler,
        tool_execution_handler=tool_execution_handler,
        event_loop_metrics=event_loop_metrics,
        event_loop_parent_span=event_loop_parent_span,
        kwargs=kwargs,
    )

    recursive_trace.end()


def _handle_tool_execution(
    stop_reason: StopReason,
    message: Message,
    model: Model,
    system_prompt: Optional[str],
    messages: Messages,
    tool_config: ToolConfig,
    tool_handler: ToolHandler,
    tool_execution_handler: Optional[ParallelToolExecutorInterface],
    event_loop_metrics: EventLoopMetrics,
    event_loop_parent_span: Optional[trace.Span],
    cycle_trace: Trace,
    cycle_span: Any,
    cycle_start_time: float,
    kwargs: dict[str, Any],
) -> Generator[dict[str, Any], None, None]:
    tool_uses: list[ToolUse] = []
    tool_results: list[ToolResult] = []
    invalid_tool_use_ids: list[str] = []

    """
    Handles the execution of tools requested by the model during an event loop cycle.

    Args:
        stop_reason (StopReason): The reason the model stopped generating.
        message (Message): The message from the model that may contain tool use requests.
        model (Model): The model provider instance.
        system_prompt (Optional[str]): The system prompt instructions for the model.
        messages (Messages): The conversation history messages.
        tool_config (ToolConfig): Configuration for available tools.
        tool_handler (ToolHandler): Handler for tool execution.
        tool_execution_handler (Optional[ParallelToolExecutorInterface]): Optional handler for parallel tool execution.
        event_loop_metrics (EventLoopMetrics): Metrics tracking object for the event loop.
        event_loop_parent_span (Any): Span for the parent of this event loop.
        cycle_trace (Trace): Trace object for the current event loop cycle.
        cycle_span (Any): Span object for tracing the cycle (type may vary).
        cycle_start_time (float): Start time of the current cycle.
        kwargs (dict[str, Any]): Additional keyword arguments, including request state.

    Yields:
        Tool invocation events along with events yielded from a recursive call to the event loop. The last event is a
        tuple containing:
            - The stop reason,
            - The updated message,
            - The updated event loop metrics,
            - The updated request state.
    """
    validate_and_prepare_tools(message, tool_uses, tool_results, invalid_tool_use_ids)

    if not tool_uses:
        yield {"stop": (stop_reason, message, event_loop_metrics, kwargs["request_state"])}
        return

    tool_handler_process = partial(
        tool_handler.process,
        model=model,
        system_prompt=system_prompt,
        messages=messages,
        tool_config=tool_config,
        kwargs=kwargs,
    )

    run_tools(
        handler=tool_handler_process,
        tool_uses=tool_uses,
        event_loop_metrics=event_loop_metrics,
        request_state=cast(Any, kwargs["request_state"]),
        invalid_tool_use_ids=invalid_tool_use_ids,
        tool_results=tool_results,
        cycle_trace=cycle_trace,
        parent_span=cycle_span,
        parallel_tool_executor=tool_execution_handler,
    )

    # Store parent cycle ID for the next cycle
    kwargs["event_loop_parent_cycle_id"] = kwargs["event_loop_cycle_id"]

    tool_result_message: Message = {
        "role": "user",
        "content": [{"toolResult": result} for result in tool_results],
    }

    messages.append(tool_result_message)
    yield {"callback": {"message": tool_result_message}}

    if cycle_span:
        tracer = get_tracer()
        tracer.end_event_loop_cycle_span(span=cycle_span, message=message, tool_result_message=tool_result_message)

    if kwargs["request_state"].get("stop_event_loop", False):
        event_loop_metrics.end_cycle(cycle_start_time, cycle_trace)
        yield {"stop": (stop_reason, message, event_loop_metrics, kwargs["request_state"])}
        return

    yield from recurse_event_loop(
        model=model,
        system_prompt=system_prompt,
        messages=messages,
        tool_config=tool_config,
        tool_handler=tool_handler,
        tool_execution_handler=tool_execution_handler,
        event_loop_metrics=event_loop_metrics,
        event_loop_parent_span=event_loop_parent_span,
        kwargs=kwargs,
    )
