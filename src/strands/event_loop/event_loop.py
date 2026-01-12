"""This module implements the central event loop.

The event loop allows agents to:

1. Process conversation messages
2. Execute tools based on model requests
3. Handle errors and recovery strategies
4. Manage recursive execution cycles
"""

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator

from opentelemetry import trace as trace_api

from ..hooks import AfterModelCallEvent, BeforeModelCallEvent, MessageAddedEvent
from ..telemetry.metrics import Trace
from ..telemetry.tracer import Tracer, get_tracer
from ..tools._validator import validate_and_prepare_tools
from ..tools.structured_output._structured_output_context import StructuredOutputContext
from ..types._events import (
    EventLoopStopEvent,
    EventLoopThrottleEvent,
    ForceStopEvent,
    ModelMessageEvent,
    ModelStopReason,
    StartEvent,
    StartEventLoopEvent,
    StructuredOutputEvent,
    ToolInterruptEvent,
    ToolResultMessageEvent,
    TypedEvent,
)
from ..types.content import Message, Messages
from ..types.exceptions import (
    ContextWindowOverflowException,
    EventLoopException,
    MaxTokensReachedException,
    ModelThrottledException,
    StructuredOutputException,
)
from ..types.streaming import StopReason
from ..types.tools import ToolResult, ToolUse
from ._recover_message_on_max_tokens_reached import recover_message_on_max_tokens_reached
from .streaming import stream_messages

if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 6
INITIAL_DELAY = 4
MAX_DELAY = 240  # 4 minutes


def _has_tool_use_in_latest_message(messages: "Messages") -> bool:
    """Check if the latest message contains any ToolUse content blocks.

    Args:
        messages: List of messages in the conversation.

    Returns:
        True if the latest message contains at least one ToolUse content block, False otherwise.
    """
    if len(messages) > 0:
        latest_message = messages[-1]
        content_blocks = latest_message.get("content", [])

        for content_block in content_blocks:
            if "toolUse" in content_block:
                return True

    return False


async def event_loop_cycle(
    agent: "Agent",
    invocation_state: dict[str, Any],
    structured_output_context: StructuredOutputContext | None = None,
) -> AsyncGenerator[TypedEvent, None]:
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
        structured_output_context: Optional context for structured output management.

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
    structured_output_context = structured_output_context or StructuredOutputContext()

    # Initialize cycle state
    invocation_state["event_loop_cycle_id"] = uuid.uuid4()

    # Initialize state and get cycle trace
    if "request_state" not in invocation_state:
        invocation_state["request_state"] = {}
    attributes = {"event_loop_cycle_id": str(invocation_state.get("event_loop_cycle_id"))}
    cycle_start_time, cycle_trace = agent.event_loop_metrics.start_cycle(attributes=attributes)
    invocation_state["event_loop_cycle_trace"] = cycle_trace

    yield StartEvent()
    yield StartEventLoopEvent()

    # Create tracer span for this event loop cycle
    tracer = get_tracer()
    cycle_span = tracer.start_event_loop_cycle_span(
        invocation_state=invocation_state,
        messages=agent.messages,
        parent_span=agent.trace_span,
        custom_trace_attributes=agent.trace_attributes,
    )
    invocation_state["event_loop_cycle_span"] = cycle_span

    # Skipping model invocation if in interrupt state as interrupts are currently only supported for tool calls.
    if agent._interrupt_state.activated:
        stop_reason: StopReason = "tool_use"
        message = agent._interrupt_state.context["tool_use_message"]
    # Skip model invocation if the latest message contains ToolUse
    elif _has_tool_use_in_latest_message(agent.messages):
        stop_reason = "tool_use"
        message = agent.messages[-1]
    else:
        model_events = _handle_model_execution(
            agent, cycle_span, cycle_trace, invocation_state, tracer, structured_output_context
        )
        async for model_event in model_events:
            if not isinstance(model_event, ModelStopReason):
                yield model_event

        stop_reason, message, *_ = model_event["stop"]
        yield ModelMessageEvent(message=message)

    try:
        if stop_reason == "max_tokens":
            """
            Handle max_tokens limit reached by the model.

            When the model reaches its maximum token limit, this represents a potentially unrecoverable
            state where the model's response was truncated. By default, Strands fails hard with an
            MaxTokensReachedException to maintain consistency with other failure types.
            """
            raise MaxTokensReachedException(
                message=(
                    "Agent has reached an unrecoverable state due to max_tokens limit. "
                    "For more information see: "
                    "https://strandsagents.com/latest/user-guide/concepts/agents/agent-loop/#maxtokensreachedexception"
                )
            )

        if stop_reason == "tool_use":
            # Handle tool execution
            tool_events = _handle_tool_execution(
                stop_reason,
                message,
                agent=agent,
                cycle_trace=cycle_trace,
                cycle_span=cycle_span,
                cycle_start_time=cycle_start_time,
                invocation_state=invocation_state,
                tracer=tracer,
                structured_output_context=structured_output_context,
            )
            async for tool_event in tool_events:
                yield tool_event

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
    except (ContextWindowOverflowException, MaxTokensReachedException) as e:
        # Special cased exceptions which we want to bubble up rather than get wrapped in an EventLoopException
        if cycle_span:
            tracer.end_span_with_error(cycle_span, str(e), e)
        raise e
    except Exception as e:
        if cycle_span:
            tracer.end_span_with_error(cycle_span, str(e), e)

        # Handle any other exceptions
        yield ForceStopEvent(reason=e)
        logger.exception("cycle failed")
        raise EventLoopException(e, invocation_state["request_state"]) from e

    # Force structured output tool call if LLM didn't use it automatically
    if structured_output_context.is_enabled and stop_reason == "end_turn":
        if structured_output_context.force_attempted:
            raise StructuredOutputException(
                "The model failed to invoke the structured output tool even after it was forced."
            )
        structured_output_context.set_forced_mode()
        logger.debug("Forcing structured output tool")
        await agent._append_messages(
            {"role": "user", "content": [{"text": "You must format the previous response as structured output."}]}
        )

        events = recurse_event_loop(
            agent=agent, invocation_state=invocation_state, structured_output_context=structured_output_context
        )
        async for typed_event in events:
            yield typed_event
        return

    yield EventLoopStopEvent(stop_reason, message, agent.event_loop_metrics, invocation_state["request_state"])


async def recurse_event_loop(
    agent: "Agent",
    invocation_state: dict[str, Any],
    structured_output_context: StructuredOutputContext | None = None,
) -> AsyncGenerator[TypedEvent, None]:
    """Make a recursive call to event_loop_cycle with the current state.

    This function is used when the event loop needs to continue processing after tool execution.

    Args:
        agent: Agent for which the recursive call is being made.
        invocation_state: Arguments to pass through event_loop_cycle
        structured_output_context: Optional context for structured output management.

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

    yield StartEvent()

    events = event_loop_cycle(
        agent=agent, invocation_state=invocation_state, structured_output_context=structured_output_context
    )
    async for event in events:
        yield event

    recursive_trace.end()


async def _handle_model_execution(
    agent: "Agent",
    cycle_span: Any,
    cycle_trace: Trace,
    invocation_state: dict[str, Any],
    tracer: Tracer,
    structured_output_context: StructuredOutputContext,
) -> AsyncGenerator[TypedEvent, None]:
    """Handle model execution with retry logic for throttling exceptions.

    Executes the model inference with automatic retry handling for throttling exceptions.
    Manages tracing, hooks, and metrics collection throughout the process.

    Args:
        agent: The agent executing the model.
        cycle_span: Span object for tracing the cycle.
        cycle_trace: Trace object for the current event loop cycle.
        invocation_state: State maintained across cycles.
        tracer: Tracer instance for span management.
        structured_output_context: Context for structured output management.

    Yields:
        Model stream events and throttle events during retries.

    Raises:
        ModelThrottledException: If max retry attempts are exceeded.
        Exception: Any other model execution errors.
    """
    # Create a trace for the stream_messages call
    stream_trace = Trace("stream_messages", parent_id=cycle_trace.id)
    cycle_trace.add_child(stream_trace)

    # Retry loop for handling throttling exceptions
    current_delay = INITIAL_DELAY
    for attempt in range(MAX_ATTEMPTS):
        model_id = agent.model.config.get("model_id") if hasattr(agent.model, "config") else None
        model_invoke_span = tracer.start_model_invoke_span(
            messages=agent.messages,
            parent_span=cycle_span,
            model_id=model_id,
            custom_trace_attributes=agent.trace_attributes,
        )
        with trace_api.use_span(model_invoke_span):
            await agent.hooks.invoke_callbacks_async(
                BeforeModelCallEvent(
                    agent=agent,
                )
            )

            if structured_output_context.forced_mode:
                tool_spec = structured_output_context.get_tool_spec()
                tool_specs = [tool_spec] if tool_spec else []
            else:
                tool_specs = agent.tool_registry.get_all_tool_specs()
            try:
                async for event in stream_messages(
                    agent.model,
                    agent.system_prompt,
                    agent.messages,
                    tool_specs,
                    system_prompt_content=agent._system_prompt_content,
                    tool_choice=structured_output_context.tool_choice,
                ):
                    yield event

                stop_reason, message, usage, metrics = event["stop"]
                invocation_state.setdefault("request_state", {})

                after_model_call_event = AfterModelCallEvent(
                    agent=agent,
                    stop_response=AfterModelCallEvent.ModelStopResponse(
                        stop_reason=stop_reason,
                        message=message,
                    ),
                )

                await agent.hooks.invoke_callbacks_async(after_model_call_event)

                # Check if hooks want to retry the model call
                if after_model_call_event.retry:
                    logger.debug(
                        "stop_reason=<%s>, retry_requested=<True>, attempt=<%d> | hook requested model retry",
                        stop_reason,
                        attempt + 1,
                    )
                    continue  # Retry the model call

                if stop_reason == "max_tokens":
                    message = recover_message_on_max_tokens_reached(message)

                if model_invoke_span:
                    tracer.end_model_invoke_span(model_invoke_span, message, usage, metrics, stop_reason)
                break  # Success! Break out of retry loop

            except Exception as e:
                if model_invoke_span:
                    tracer.end_span_with_error(model_invoke_span, str(e), e)

                after_model_call_event = AfterModelCallEvent(
                    agent=agent,
                    exception=e,
                )
                await agent.hooks.invoke_callbacks_async(after_model_call_event)

                # Check if hooks want to retry the model call
                if after_model_call_event.retry:
                    logger.debug(
                        "exception=<%s>, retry_requested=<True>, attempt=<%d> | hook requested model retry",
                        type(e).__name__,
                        attempt + 1,
                    )
                    continue  # Retry the model call

                if isinstance(e, ModelThrottledException):
                    if attempt + 1 == MAX_ATTEMPTS:
                        yield ForceStopEvent(reason=e)
                        raise e

                    logger.debug(
                        "retry_delay_seconds=<%s>, max_attempts=<%s>, current_attempt=<%s> "
                        "| throttling exception encountered "
                        "| delaying before next retry",
                        current_delay,
                        MAX_ATTEMPTS,
                        attempt + 1,
                    )
                    await asyncio.sleep(current_delay)
                    current_delay = min(current_delay * 2, MAX_DELAY)

                    yield EventLoopThrottleEvent(delay=current_delay)
                else:
                    raise e

    try:
        # Add message in trace and mark the end of the stream messages trace
        stream_trace.add_message(message)
        stream_trace.end()

        # Add the response message to the conversation
        agent.messages.append(message)
        await agent.hooks.invoke_callbacks_async(MessageAddedEvent(agent=agent, message=message))

        # Update metrics
        agent.event_loop_metrics.update_usage(usage)
        agent.event_loop_metrics.update_metrics(metrics)

    except Exception as e:
        if cycle_span:
            tracer.end_span_with_error(cycle_span, str(e), e)

        yield ForceStopEvent(reason=e)
        logger.exception("cycle failed")
        raise EventLoopException(e, invocation_state["request_state"]) from e


async def _handle_tool_execution(
    stop_reason: StopReason,
    message: Message,
    agent: "Agent",
    cycle_trace: Trace,
    cycle_span: Any,
    cycle_start_time: float,
    invocation_state: dict[str, Any],
    tracer: Tracer,
    structured_output_context: StructuredOutputContext,
) -> AsyncGenerator[TypedEvent, None]:
    """Handles the execution of tools requested by the model during an event loop cycle.

    Args:
        stop_reason: The reason the model stopped generating.
        message: The message from the model that may contain tool use requests.
        agent: Agent for which tools are being executed.
        cycle_trace: Trace object for the current event loop cycle.
        cycle_span: Span object for tracing the cycle (type may vary).
        cycle_start_time: Start time of the current cycle.
        invocation_state: Additional keyword arguments, including request state.
        tracer: Tracer instance for span management.
        structured_output_context: Optional context for structured output management.

    Yields:
        Tool stream events along with events yielded from a recursive call to the event loop. The last event is a tuple
        containing:
            - The stop reason,
            - The updated message,
            - The updated event loop metrics,
            - The updated request state.
    """
    tool_uses: list[ToolUse] = []
    tool_results: list[ToolResult] = []
    invalid_tool_use_ids: list[str] = []

    validate_and_prepare_tools(message, tool_uses, tool_results, invalid_tool_use_ids)
    tool_uses = [tool_use for tool_use in tool_uses if tool_use.get("toolUseId") not in invalid_tool_use_ids]

    if agent._interrupt_state.activated:
        tool_results.extend(agent._interrupt_state.context["tool_results"])

        # Filter to only the interrupted tools when resuming from interrupt (tool uses without results)
        tool_use_ids = {tool_result["toolUseId"] for tool_result in tool_results}
        tool_uses = [tool_use for tool_use in tool_uses if tool_use["toolUseId"] not in tool_use_ids]

    interrupts = []
    tool_events = agent.tool_executor._execute(
        agent, tool_uses, tool_results, cycle_trace, cycle_span, invocation_state, structured_output_context
    )
    async for tool_event in tool_events:
        if isinstance(tool_event, ToolInterruptEvent):
            interrupts.extend(tool_event["tool_interrupt_event"]["interrupts"])

        yield tool_event

    structured_output_result = None
    if structured_output_context.is_enabled:
        if structured_output_result := structured_output_context.extract_result(tool_uses):
            yield StructuredOutputEvent(structured_output=structured_output_result)
            structured_output_context.stop_loop = True

    invocation_state["event_loop_parent_cycle_id"] = invocation_state["event_loop_cycle_id"]

    if interrupts:
        # Session state stored on AfterInvocationEvent.
        agent._interrupt_state.context = {"tool_use_message": message, "tool_results": tool_results}
        agent._interrupt_state.activate()

        agent.event_loop_metrics.end_cycle(cycle_start_time, cycle_trace)
        yield EventLoopStopEvent(
            "interrupt",
            message,
            agent.event_loop_metrics,
            invocation_state["request_state"],
            interrupts,
            structured_output=structured_output_result,
        )
        if cycle_span:
            tracer.end_event_loop_cycle_span(span=cycle_span, message=message)

        return

    agent._interrupt_state.deactivate()

    tool_result_message: Message = {
        "role": "user",
        "content": [{"toolResult": result} for result in tool_results],
    }

    agent.messages.append(tool_result_message)
    await agent.hooks.invoke_callbacks_async(MessageAddedEvent(agent=agent, message=tool_result_message))

    yield ToolResultMessageEvent(message=tool_result_message)

    if cycle_span:
        tracer.end_event_loop_cycle_span(span=cycle_span, message=message, tool_result_message=tool_result_message)

    if invocation_state["request_state"].get("stop_event_loop", False) or structured_output_context.stop_loop:
        agent.event_loop_metrics.end_cycle(cycle_start_time, cycle_trace)
        yield EventLoopStopEvent(
            stop_reason,
            message,
            agent.event_loop_metrics,
            invocation_state["request_state"],
            structured_output=structured_output_result,
        )
        return

    events = recurse_event_loop(
        agent=agent, invocation_state=invocation_state, structured_output_context=structured_output_context
    )
    async for event in events:
        yield event
