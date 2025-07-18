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
from typing import TYPE_CHECKING, Any, AsyncGenerator, cast

from opentelemetry import trace as trace_api

from ..experimental.hooks import (
    AfterModelInvocationEvent,
    AfterToolInvocationEvent,
    BeforeModelInvocationEvent,
    BeforeToolInvocationEvent,
)
from ..hooks import (
    MessageAddedEvent,
)
from ..telemetry.metrics import Trace
from ..telemetry.tracer import get_tracer
from ..tools.executor import run_tools, validate_and_prepare_tools
from ..types.content import Message
from ..types.exceptions import ContextWindowOverflowException, EventLoopException, ModelThrottledException
from ..types.streaming import Metrics, StopReason
from ..types.tools import ToolChoice, ToolChoiceAuto, ToolConfig, ToolGenerator, ToolResult, ToolUse
from .streaming import stream_messages

if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 6
INITIAL_DELAY = 4
MAX_DELAY = 240  # 4 minutes


async def event_loop_cycle(agent: "Agent", invocation_state: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
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
        agent: The agent for which the cycle is being executed.
        invocation_state: Additional arguments including:

            - request_state: State maintained across cycles
            - event_loop_cycle_id: Unique ID for this cycle
            - event_loop_cycle_span: Current tracing Span for this cycle

    Yields:
        Model and tool stream events. The last event is a tuple containing:

            - StopReason: Reason the model stopped generating (e.g., "tool_use")
            - Message: The generated message from the model
            - EventLoopMetrics: Updated metrics for the event loop
            - Any: Updated request state

    Raises:
        EventLoopException: If an error occurs during execution
        ContextWindowOverflowException: If the input is too large for the model
    """
    # Initialize cycle state
    invocation_state["event_loop_cycle_id"] = uuid.uuid4()

    # Initialize state and get cycle trace
    if "request_state" not in invocation_state:
        invocation_state["request_state"] = {}
    attributes = {"event_loop_cycle_id": str(invocation_state.get("event_loop_cycle_id"))}
    cycle_start_time, cycle_trace = agent.event_loop_metrics.start_cycle(attributes=attributes)
    invocation_state["event_loop_cycle_trace"] = cycle_trace

    yield {"callback": {"start": True}}
    yield {"callback": {"start_event_loop": True}}

    # Create tracer span for this event loop cycle
    tracer = get_tracer()
    cycle_span = tracer.start_event_loop_cycle_span(
        invocation_state=invocation_state, messages=agent.messages, parent_span=agent.trace_span
    )
    invocation_state["event_loop_cycle_span"] = cycle_span

    # Create a trace for the stream_messages call
    stream_trace = Trace("stream_messages", parent_id=cycle_trace.id)
    cycle_trace.add_child(stream_trace)

    # Process messages with exponential backoff for throttling
    message: Message
    stop_reason: StopReason
    usage: Any
    metrics: Metrics

    # Retry loop for handling throttling exceptions
    current_delay = INITIAL_DELAY
    for attempt in range(MAX_ATTEMPTS):
        model_id = agent.model.config.get("model_id") if hasattr(agent.model, "config") else None
        model_invoke_span = tracer.start_model_invoke_span(
            messages=agent.messages,
            parent_span=cycle_span,
            model_id=model_id,
        )
        with trace_api.use_span(model_invoke_span):
            tool_specs = agent.tool_registry.get_all_tool_specs()

            agent.hooks.invoke_callbacks(
                BeforeModelInvocationEvent(
                    agent=agent,
                )
            )

            try:
                # TODO: To maintain backwards compatibility, we need to combine the stream event with invocation_state
                #       before yielding to the callback handler. This will be revisited when migrating to strongly
                #       typed events.
                async for event in stream_messages(agent.model, agent.system_prompt, agent.messages, tool_specs):
                    if "callback" in event:
                        yield {
                            "callback": {
                                **event["callback"],
                                **(invocation_state if "delta" in event["callback"] else {}),
                            }
                        }

                stop_reason, message, usage, metrics = event["stop"]
                invocation_state.setdefault("request_state", {})

                agent.hooks.invoke_callbacks(
                    AfterModelInvocationEvent(
                        agent=agent,
                        stop_response=AfterModelInvocationEvent.ModelStopResponse(
                            stop_reason=stop_reason,
                            message=message,
                        ),
                    )
                )

                if model_invoke_span:
                    tracer.end_model_invoke_span(model_invoke_span, message, usage, stop_reason)
                break  # Success! Break out of retry loop

            except Exception as e:
                if model_invoke_span:
                    tracer.end_span_with_error(model_invoke_span, str(e), e)

                agent.hooks.invoke_callbacks(
                    AfterModelInvocationEvent(
                        agent=agent,
                        exception=e,
                    )
                )

                if isinstance(e, ModelThrottledException):
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

                    yield {"callback": {"event_loop_throttled_delay": current_delay, **invocation_state}}
                else:
                    raise e

    try:
        # Add message in trace and mark the end of the stream messages trace
        stream_trace.add_message(message)
        stream_trace.end()

        # Add the response message to the conversation
        agent.messages.append(message)
        agent.hooks.invoke_callbacks(MessageAddedEvent(agent=agent, message=message))
        yield {"callback": {"message": message}}

        # Update metrics
        agent.event_loop_metrics.update_usage(usage)
        agent.event_loop_metrics.update_metrics(metrics)

        # If the model is requesting to use tools
        if stop_reason == "tool_use":
            # Handle tool execution
            events = _handle_tool_execution(
                stop_reason,
                message,
                agent=agent,
                cycle_trace=cycle_trace,
                cycle_span=cycle_span,
                cycle_start_time=cycle_start_time,
                invocation_state=invocation_state,
            )
            async for event in events:
                yield event

            return

        # End the cycle and return results
        agent.event_loop_metrics.end_cycle(cycle_start_time, cycle_trace, attributes)
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
        raise EventLoopException(e, invocation_state["request_state"]) from e

    yield {"stop": (stop_reason, message, agent.event_loop_metrics, invocation_state["request_state"])}


async def recurse_event_loop(agent: "Agent", invocation_state: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
    """Make a recursive call to event_loop_cycle with the current state.

    This function is used when the event loop needs to continue processing after tool execution.

    Args:
        agent: Agent for which the recursive call is being made.
        invocation_state: Arguments to pass through event_loop_cycle


    Yields:
        Results from event_loop_cycle where the last result contains:

            - StopReason: Reason the model stopped generating
            - Message: The generated message from the model
            - EventLoopMetrics: Updated metrics for the event loop
            - Any: Updated request state
    """
    cycle_trace = invocation_state["event_loop_cycle_trace"]

    # Recursive call trace
    recursive_trace = Trace("Recursive call", parent_id=cycle_trace.id)
    cycle_trace.add_child(recursive_trace)

    yield {"callback": {"start": True}}

    events = event_loop_cycle(agent=agent, invocation_state=invocation_state)
    async for event in events:
        yield event

    recursive_trace.end()


async def run_tool(agent: "Agent", tool_use: ToolUse, invocation_state: dict[str, Any]) -> ToolGenerator:
    """Process a tool invocation.

    Looks up the tool in the registry and streams it with the provided parameters.

    Args:
        agent: The agent for which the tool is being executed.
        tool_use: The tool object to process, containing name and parameters.
        invocation_state: Context for the tool invocation, including agent state.

    Yields:
        Tool events with the last being the tool result.
    """
    logger.debug("tool_use=<%s> | streaming", tool_use)
    tool_name = tool_use["name"]

    # Get the tool info
    tool_info = agent.tool_registry.dynamic_tools.get(tool_name)
    tool_func = tool_info if tool_info is not None else agent.tool_registry.registry.get(tool_name)

    # Add standard arguments to invocation_state for Python tools
    invocation_state.update(
        {
            "model": agent.model,
            "system_prompt": agent.system_prompt,
            "messages": agent.messages,
            "tool_config": ToolConfig(  # for backwards compatability
                tools=[{"toolSpec": tool_spec} for tool_spec in agent.tool_registry.get_all_tool_specs()],
                toolChoice=cast(ToolChoice, {"auto": ToolChoiceAuto()}),
            ),
        }
    )

    before_event = agent.hooks.invoke_callbacks(
        BeforeToolInvocationEvent(
            agent=agent,
            selected_tool=tool_func,
            tool_use=tool_use,
            invocation_state=invocation_state,
        )
    )

    try:
        selected_tool = before_event.selected_tool
        tool_use = before_event.tool_use
        invocation_state = before_event.invocation_state  # Get potentially modified invocation_state from hook

        # Check if tool exists
        if not selected_tool:
            if tool_func == selected_tool:
                logger.error(
                    "tool_name=<%s>, available_tools=<%s> | tool not found in registry",
                    tool_name,
                    list(agent.tool_registry.registry.keys()),
                )
            else:
                logger.debug(
                    "tool_name=<%s>, tool_use_id=<%s> | a hook resulted in a non-existing tool call",
                    tool_name,
                    str(tool_use.get("toolUseId")),
                )

            result: ToolResult = {
                "toolUseId": str(tool_use.get("toolUseId")),
                "status": "error",
                "content": [{"text": f"Unknown tool: {tool_name}"}],
            }
            # for every Before event call, we need to have an AfterEvent call
            after_event = agent.hooks.invoke_callbacks(
                AfterToolInvocationEvent(
                    agent=agent,
                    selected_tool=selected_tool,
                    tool_use=tool_use,
                    invocation_state=invocation_state,  # Keep as invocation_state for backward compatibility with hooks
                    result=result,
                )
            )
            yield after_event.result
            return

        async for event in selected_tool.stream(tool_use, invocation_state):
            yield event

        result = event

        after_event = agent.hooks.invoke_callbacks(
            AfterToolInvocationEvent(
                agent=agent,
                selected_tool=selected_tool,
                tool_use=tool_use,
                invocation_state=invocation_state,  # Keep as invocation_state for backward compatibility with hooks
                result=result,
            )
        )
        yield after_event.result

    except Exception as e:
        logger.exception("tool_name=<%s> | failed to process tool", tool_name)
        error_result: ToolResult = {
            "toolUseId": str(tool_use.get("toolUseId")),
            "status": "error",
            "content": [{"text": f"Error: {str(e)}"}],
        }
        after_event = agent.hooks.invoke_callbacks(
            AfterToolInvocationEvent(
                agent=agent,
                selected_tool=selected_tool,
                tool_use=tool_use,
                invocation_state=invocation_state,  # Keep as invocation_state for backward compatibility with hooks
                result=error_result,
                exception=e,
            )
        )
        yield after_event.result


async def _handle_tool_execution(
    stop_reason: StopReason,
    message: Message,
    agent: "Agent",
    cycle_trace: Trace,
    cycle_span: Any,
    cycle_start_time: float,
    invocation_state: dict[str, Any],
) -> AsyncGenerator[dict[str, Any], None]:
    tool_uses: list[ToolUse] = []
    tool_results: list[ToolResult] = []
    invalid_tool_use_ids: list[str] = []

    """
    Handles the execution of tools requested by the model during an event loop cycle.

    Args:
        stop_reason: The reason the model stopped generating.
        message: The message from the model that may contain tool use requests.
        event_loop_metrics: Metrics tracking object for the event loop.
        event_loop_parent_span: Span for the parent of this event loop.
        cycle_trace: Trace object for the current event loop cycle.
        cycle_span: Span object for tracing the cycle (type may vary).
        cycle_start_time: Start time of the current cycle.
        invocation_state: Additional keyword arguments, including request state.

    Yields:
        Tool stream events along with events yielded from a recursive call to the event loop. The last event is a tuple
        containing:
            - The stop reason,
            - The updated message,
            - The updated event loop metrics,
            - The updated request state.
    """
    validate_and_prepare_tools(message, tool_uses, tool_results, invalid_tool_use_ids)

    if not tool_uses:
        yield {"stop": (stop_reason, message, agent.event_loop_metrics, invocation_state["request_state"])}
        return

    def tool_handler(tool_use: ToolUse) -> ToolGenerator:
        return run_tool(agent, tool_use, invocation_state)

    tool_events = run_tools(
        handler=tool_handler,
        tool_uses=tool_uses,
        event_loop_metrics=agent.event_loop_metrics,
        invalid_tool_use_ids=invalid_tool_use_ids,
        tool_results=tool_results,
        cycle_trace=cycle_trace,
        parent_span=cycle_span,
    )
    async for tool_event in tool_events:
        yield tool_event

    # Store parent cycle ID for the next cycle
    invocation_state["event_loop_parent_cycle_id"] = invocation_state["event_loop_cycle_id"]

    tool_result_message: Message = {
        "role": "user",
        "content": [{"toolResult": result} for result in tool_results],
    }

    agent.messages.append(tool_result_message)
    agent.hooks.invoke_callbacks(MessageAddedEvent(agent=agent, message=tool_result_message))
    yield {"callback": {"message": tool_result_message}}

    if cycle_span:
        tracer = get_tracer()
        tracer.end_event_loop_cycle_span(span=cycle_span, message=message, tool_result_message=tool_result_message)

    if invocation_state["request_state"].get("stop_event_loop", False):
        agent.event_loop_metrics.end_cycle(cycle_start_time, cycle_trace)
        yield {"stop": (stop_reason, message, agent.event_loop_metrics, invocation_state["request_state"])}
        return

    events = recurse_event_loop(agent=agent, invocation_state=invocation_state)
    async for event in events:
        yield event
