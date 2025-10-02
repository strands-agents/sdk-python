"""This module implements the central event loop.

The event loop allows agents to:

1. Process conversation messages
2. Execute tools based on model requests
3. Handle errors and recovery strategies
4. Manage recursive execution cycles
"""

import asyncio
import copy
import inspect
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator

from opentelemetry import trace as trace_api

from ..hooks import AfterModelCallEvent, BeforeModelCallEvent, MessageAddedEvent
from ..telemetry.metrics import Trace
from ..telemetry.tracer import get_tracer
from ..tools._validator import validate_and_prepare_tools
from ..types._events import (
    DelegationCompleteEvent,
    DelegationProxyEvent,
    EventLoopStopEvent,
    EventLoopThrottleEvent,
    ForceStopEvent,
    ModelMessageEvent,
    ModelStopReason,
    StartEvent,
    StartEventLoopEvent,
    ToolResultMessageEvent,
    TypedEvent,
)
from ..types.content import Message
from ..types.exceptions import (
    AgentDelegationException,
    ContextWindowOverflowException,
    EventLoopException,
    MaxTokensReachedException,
    ModelThrottledException,
)
from ..types.streaming import Metrics, StopReason
from ..types.tools import ToolResult, ToolUse
from ._recover_message_on_max_tokens_reached import recover_message_on_max_tokens_reached
from .streaming import stream_messages

if TYPE_CHECKING:
    from ..agent import Agent, AgentResult

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 6
INITIAL_DELAY = 4
MAX_DELAY = 240  # 4 minutes


async def event_loop_cycle(agent: "Agent", invocation_state: dict[str, Any]) -> AsyncGenerator[TypedEvent, None]:
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

    yield StartEvent()
    yield StartEventLoopEvent()

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
            agent.hooks.invoke_callbacks(
                BeforeModelCallEvent(
                    agent=agent,
                )
            )

            tool_specs = agent.tool_registry.get_all_tool_specs()

            try:
                async for event in stream_messages(agent.model, agent.system_prompt, agent.messages, tool_specs):
                    if not isinstance(event, ModelStopReason):
                        yield event

                stop_reason, message, usage, metrics = event["stop"]
                invocation_state.setdefault("request_state", {})

                agent.hooks.invoke_callbacks(
                    AfterModelCallEvent(
                        agent=agent,
                        stop_response=AfterModelCallEvent.ModelStopResponse(
                            stop_reason=stop_reason,
                            message=message,
                        ),
                    )
                )

                if stop_reason == "max_tokens":
                    message = recover_message_on_max_tokens_reached(message)

                if model_invoke_span:
                    tracer.end_model_invoke_span(model_invoke_span, message, usage, stop_reason)
                break  # Success! Break out of retry loop

            except AgentDelegationException as delegation_exc:
                # Handle delegation immediately
                delegation_result = await _handle_delegation(
                    agent=agent,
                    delegation_exception=delegation_exc,
                    invocation_state=invocation_state,
                    cycle_trace=cycle_trace,
                    cycle_span=cycle_span,
                )

                # Yield delegation completion event and return result
                yield DelegationCompleteEvent(
                    target_agent=delegation_exc.target_agent,
                    result=delegation_result,
                )

                # Return delegation result as final response
                yield EventLoopStopEvent(
                    "delegation_complete", delegation_result.message, delegation_result.metrics, delegation_result.state
                )
                return

            except Exception as e:
                if model_invoke_span:
                    tracer.end_span_with_error(model_invoke_span, str(e), e)

                agent.hooks.invoke_callbacks(
                    AfterModelCallEvent(
                        agent=agent,
                        exception=e,
                    )
                )

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
        agent.hooks.invoke_callbacks(MessageAddedEvent(agent=agent, message=message))
        yield ModelMessageEvent(message=message)

        # Update metrics
        agent.event_loop_metrics.update_usage(usage)
        agent.event_loop_metrics.update_metrics(metrics)

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
            async for typed_event in events:
                yield typed_event

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

    yield EventLoopStopEvent(stop_reason, message, agent.event_loop_metrics, invocation_state["request_state"])


async def recurse_event_loop(agent: "Agent", invocation_state: dict[str, Any]) -> AsyncGenerator[TypedEvent, None]:
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

    yield StartEvent()

    events = event_loop_cycle(agent=agent, invocation_state=invocation_state)
    async for event in events:
        yield event

    recursive_trace.end()


def _filter_delegation_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter and optimize messages for delegation to reduce noise and token usage.

    This function implements sophisticated message filtering to preserve important
    context while removing internal tool chatter and noise that would be irrelevant
    to the delegated agent.

    Args:
        messages: List of messages from the orchestrator agent

    Returns:
        Tuple of (filtered_messages, filtering_stats) where:
        - filtered_messages: Optimized list of messages for delegation
        - filtering_stats: Dictionary with filtering metrics for observability

    The filtering logic works as follows:
    - System messages: Always included (essential for context)
    - User messages: Always included, but cleaned to remove embedded tool content
    - Assistant messages: Filtered to remove internal tool noise while preserving meaningful text
    """
    filtered_messages = []
    for msg in messages:
        msg_role = msg.get("role", "")
        msg_content = msg.get("content", [])

        # Always include system prompts for context preservation
        if msg_role == "system":
            filtered_messages.append(msg)
            continue

        # Always include user messages for conversational continuity
        if msg_role == "user":
            # For user messages, ensure content is clean text
            if isinstance(msg_content, list):
                # Filter out any embedded tool content from user messages
                clean_content = [
                    item for item in msg_content if isinstance(item, dict) and item.get("type") == "text"
                ]
                if clean_content:
                    filtered_messages.append({"role": "user", "content": clean_content})
            else:
                filtered_messages.append(msg)
            continue

        # For assistant messages, filter out internal tool chatter
        if msg_role == "assistant":
            if isinstance(msg_content, list):
                # Sophisticated content analysis for assistant messages
                has_internal_tool_content = any(
                    (content.get("type") == "toolUse" and not content.get("name", "").startswith("handoff_to_"))
                    or ("toolResult" in content and content.get("toolResult", {}).get("status") == "error")
                    for content in msg_content
                    if isinstance(content, dict)
                )

                # Check if message contains meaningful text response
                has_meaningful_text = any(
                    content.get("type") == "text" and content.get("text", "").strip()
                    for content in msg_content
                    if isinstance(content, dict)
                )

                # Include if it has meaningful text and no internal tool noise
                if has_meaningful_text and not has_internal_tool_content:
                    filtered_messages.append(msg)
                elif has_meaningful_text and has_internal_tool_content:
                    # Clean the message by removing tool content but keeping text
                    clean_content = [
                        item for item in msg_content if isinstance(item, dict) and item.get("type") == "text"
                    ]
                    if clean_content:
                        filtered_messages.append({"role": "assistant", "content": clean_content})
            else:
                # Simple text content - include as-is
                filtered_messages.append(msg)

    # Calculate filtering statistics for observability
    original_count = len(messages)
    filtered_count = len(filtered_messages)
    filtering_stats = {
        "original_message_count": original_count,
        "filtered_message_count": filtered_count,
        "noise_removed": original_count - filtered_count,
        "compression_ratio": f"{(filtered_count / original_count * 100):.1f}%"
        if original_count > 0
        else "0%",
    }

    return filtered_messages, filtering_stats


async def _handle_delegation(
    agent: "Agent",
    delegation_exception: AgentDelegationException,
    invocation_state: dict[str, Any],
    cycle_trace: Trace,
    cycle_span: Any,
) -> "AgentResult":
    """Handle agent delegation by transferring execution to sub-agent.

    Args:
        agent: The orchestrator agent
        delegation_exception: The delegation exception containing context
        invocation_state: Current invocation state
        cycle_trace: Trace object for tracking
        cycle_span: Span for tracing

    Returns:
        AgentResult from the delegated agent

    Raises:
        ValueError: If delegation fails or target agent not found
        asyncio.TimeoutError: If delegation times out
    """
    # Find the target sub-agent
    target_agent = agent._sub_agents.get(delegation_exception.target_agent)
    if not target_agent:
        raise ValueError(f"Target agent '{delegation_exception.target_agent}' not found")

    print(f"DEBUG: Delegation chain: {delegation_exception.delegation_chain}")
    print(f"DEBUG: Current agent name: {agent.name}")
    print(f"DEBUG: Target agent name: {target_agent.name}")

    # Check for circular delegation
    # The delegation chain contains agents that have already been visited in this delegation chain
    # If the target agent is already in the chain, it means we're trying to delegate to an agent that was already part of the chain
    if target_agent.name in delegation_exception.delegation_chain:
        raise ValueError(
            f"Circular delegation detected: {' -> '.join(delegation_exception.delegation_chain + [target_agent.name])}"
        )

    # Additional check: prevent self-delegation (agent delegating to itself)
    if agent.name == delegation_exception.target_agent:
        raise ValueError(f"Self-delegation detected: {agent.name} cannot delegate to itself")

    # Create delegation trace
    delegation_trace = Trace("agent_delegation", parent_id=cycle_trace.id)
    cycle_trace.add_child(delegation_trace)

    # Handle session management if present
    original_session_id = None
    if agent._session_manager:
        original_session_id = agent._session_manager.session_id
        # Create nested session for sub-agent
        sub_session_id = f"{original_session_id}/delegation/{uuid.uuid4().hex}"

        # Validate session manager constructor compatibility before creating sub-session
        session_mgr_class = type(agent._session_manager)
        if hasattr(session_mgr_class, '__init__'):
            sig = inspect.signature(session_mgr_class.__init__)
            if 'session_id' not in sig.parameters:
                raise TypeError(f"Session manager {type(agent._session_manager).__name__} doesn't accept session_id parameter")

        target_agent._session_manager = session_mgr_class(session_id=sub_session_id)
        await target_agent._session_manager.save_agent(target_agent)

    try:
        # STATE TRANSFER: Handle agent.state with explicit rules
        if delegation_exception.transfer_state and hasattr(agent, "state"):
            # Use custom serializer if provided, otherwise use deepcopy
            if agent.delegation_state_serializer:
                try:
                    target_agent.state = agent.delegation_state_serializer(agent.state)
                except Exception as e:
                    delegation_trace.metadata["state_serialization_error"] = {"error": str(e), "fallback_to_deepcopy": True}
                    target_agent.state = copy.deepcopy(agent.state)
            else:
                # Deep copy the orchestrator's state to sub-agent
                target_agent.state = copy.deepcopy(agent.state)
        # If transfer_state is False, sub-agent keeps its own state (default behavior)

        # ENHANCED: Message filtering on transfer - sophisticated context optimization
        if delegation_exception.transfer_messages:
            filtered_messages, filtering_stats = _filter_delegation_messages(agent.messages)

            # Track filtering effectiveness for observability
            delegation_trace.metadata["message_filtering_applied"] = filtering_stats

            target_agent.messages = filtered_messages
        else:
            # Start with fresh conversation history
            target_agent.messages = []

        # Always add delegation context message for clarity
        delegation_context = {
            "role": "user",
            "content": [{"text": f"Delegated from {agent.name}: {delegation_exception.message}"}],
        }
        target_agent.messages.append(delegation_context)

        # Transfer additional context if provided
        if delegation_exception.context:
            context_message = {
                "role": "user",
                "content": [{"text": f"Additional context: {json.dumps(delegation_exception.context)}"}],
            }
            target_agent.messages.append(context_message)

        # STREAMING PROXY: Check if we should proxy streaming events
        if (
            agent.delegation_streaming_proxy
            and hasattr(invocation_state, "is_streaming")
            and invocation_state.get("is_streaming")
        ):
            # Use streaming execution with event proxying
            final_event = None
            async for event in _handle_delegation_with_streaming(
                target_agent=target_agent,
                agent=agent,
                delegation_exception=delegation_exception,
                invocation_state=invocation_state,
                delegation_trace=delegation_trace,
            ):
                final_event = event
            # Extract result from the final event
            result = (
                final_event.original_event.result
                if hasattr(final_event, "original_event") and hasattr(final_event.original_event, "result")
                else None
            )
        else:
            # Execute the sub-agent with timeout support (non-streaming)
            if agent.delegation_timeout is not None:
                result = await asyncio.wait_for(target_agent.invoke_async(), timeout=agent.delegation_timeout)
            else:
                result = await target_agent.invoke_async()

        # Record delegation completion
        delegation_trace.metadata["delegation_complete"] = {
            "from_agent": agent.name,
            "to_agent": delegation_exception.target_agent,
            "message": delegation_exception.message,
            "state_transferred": delegation_exception.transfer_state,
            "messages_transferred": delegation_exception.transfer_messages,
            "streaming_proxied": agent.delegation_streaming_proxy,
        }

        return result

    except asyncio.TimeoutError:
        delegation_trace.metadata["delegation_timeout"] = {"target_agent": delegation_exception.target_agent, "timeout_seconds": agent.delegation_timeout}
        raise TimeoutError(
            f"Delegation to {delegation_exception.target_agent} timed out after {agent.delegation_timeout} seconds"
        ) from None

    finally:
        delegation_trace.end()
        # Restore original session if needed
        if original_session_id and agent._session_manager:
            agent._session_manager.session_id = original_session_id


async def _handle_delegation_with_streaming(
    target_agent: "Agent",
    agent: "Agent",
    delegation_exception: AgentDelegationException,
    invocation_state: dict[str, Any],
    delegation_trace: Trace,
) -> AsyncGenerator[TypedEvent, None]:
    """Handle delegation with streaming event proxying for real-time visibility.

    This method ensures that when the original caller expects streaming events,
    the sub-agent's streaming events are proxied back in real-time through the
    parent event loop's async generator.

    Args:
        target_agent: The sub-agent to execute
        agent: The orchestrator agent
        delegation_exception: The delegation exception
        invocation_state: Current invocation state with streaming context
        delegation_trace: Trace object for tracking

    Returns:
        AgentResult from the delegated agent

    Raises:
        asyncio.TimeoutError: If delegation times out during streaming
    """
    from ..types._events import AgentResultEvent

    # Store streamed events and final result
    streamed_events = []
    final_result = None

    try:
        # Stream events from sub-agent with timeout
        if agent.delegation_timeout is not None:
            async for event in asyncio.wait_for(target_agent.stream_async(), timeout=agent.delegation_timeout):
                # Proxy the event with delegation context
                proxy_event = DelegationProxyEvent(
                    original_event=event, from_agent=agent.name, to_agent=delegation_exception.target_agent
                )

                streamed_events.append(proxy_event)
                delegation_trace.metadata["stream_event_proxied"] = {
                    "event_type": type(event).__name__,
                    "from_agent": agent.name,
                    "to_agent": delegation_exception.target_agent,
                }

                # Integrate with parent event loop by yielding proxy events
                # This requires the parent event loop to be aware of delegation proxying
                # In practice, this would be yielded back through the event_loop_cycle generator
                yield proxy_event

                # Check if this is the final result event
                if isinstance(event, AgentResultEvent):
                    final_result = event.get("result")
        else:
            # No timeout - stream indefinitely
            async for event in target_agent.stream_async():
                proxy_event = DelegationProxyEvent(
                    original_event=event, from_agent=agent.name, to_agent=delegation_exception.target_agent
                )

                streamed_events.append(proxy_event)
                delegation_trace.metadata["stream_event_proxied"] = {
                    "event_type": type(event).__name__,
                    "from_agent": agent.name,
                    "to_agent": delegation_exception.target_agent,
                }

                yield proxy_event

                if isinstance(event, AgentResultEvent):
                    final_result = event.get("result")

    except asyncio.TimeoutError:
        delegation_trace.metadata["delegation_timeout"] = {
            "target_agent": delegation_exception.target_agent,
            "timeout_seconds": agent.delegation_timeout,
            "during_streaming": True,
        }
        raise TimeoutError(
            f"Delegation to {delegation_exception.target_agent} "
            f"timed out after {agent.delegation_timeout} seconds during streaming"
        ) from None

    # ENHANCED: Streaming proxy correctness - eliminate fallback to blocking invoke_async
    # The streaming proxy should never fall back to blocking calls for real-time UX
    if final_result is None:
        # This indicates a streaming protocol issue - all proper agent streams should end with AgentResultEvent
        delegation_trace.metadata["streaming_protocol_error"] = {
            "error": "Stream ended without AgentResultEvent",
            "events_proxied": len(streamed_events),
            "fallback_prevented": True,
        }

        # Instead of falling back to blocking invoke_async, raise a structured error
        # This maintains real-time UX guarantees and forces proper stream implementation
        raise RuntimeError(
            f"Delegation streaming protocol error: {delegation_exception.target_agent} "
            f"stream ended without final result event. "
            f"Events proxied: {len(streamed_events)}. "
            f"Sub-agent must properly implement streaming interface."
        )

    # Validate streaming completeness for real-time UX guarantees
    if not streamed_events:
        # Instead of just logging a warning and continuing (which breaks real-time UX),
        # raise an error to force proper streaming implementation
        delegation_trace.metadata["streaming_completeness_error"] = {
            "error": "No events were streamed during delegation - this violates real-time UX guarantees",
            "target_agent": delegation_exception.target_agent,
            "final_result_obtained": final_result is not None,
            "requirement": "Sub-agent must implement proper streaming interface with real-time event emission",
        }

        raise RuntimeError(
            f"Delegation streaming completeness error: {delegation_exception.target_agent} "
            f"produced no streaming events. This violates real-time UX guarantees. "
            f"Sub-agent must implement proper streaming interface with event emission."
        )

    return


async def _handle_tool_execution(
    stop_reason: StopReason,
    message: Message,
    agent: "Agent",
    cycle_trace: Trace,
    cycle_span: Any,
    cycle_start_time: float,
    invocation_state: dict[str, Any],
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
    if not tool_uses:
        yield EventLoopStopEvent(stop_reason, message, agent.event_loop_metrics, invocation_state["request_state"])
        return

    print(f"DEBUG: About to execute tools for {len(tool_uses)} tool uses")
    try:
        tool_events = agent.tool_executor._execute(
            agent, tool_uses, tool_results, cycle_trace, cycle_span, invocation_state
        )
        # Need to properly handle async generator exceptions
        try:
            async for tool_event in tool_events:
                yield tool_event
            print(f"DEBUG: Tool execution completed successfully")
        except AgentDelegationException as delegation_exc:
            print(f"DEBUG: Caught delegation exception from async generator for {delegation_exc.target_agent}")
            # Re-raise to be caught by outer try-catch
            raise delegation_exc
    except AgentDelegationException as delegation_exc:
        print(f"DEBUG: Caught delegation exception for {delegation_exc.target_agent}")
        # Handle delegation during tool execution
        delegation_result = await _handle_delegation(
            agent=agent,
            delegation_exception=delegation_exc,
            invocation_state=invocation_state,
            cycle_trace=cycle_trace,
            cycle_span=cycle_span,
        )

        # Yield delegation completion event and return result
        yield DelegationCompleteEvent(
            target_agent=delegation_exc.target_agent,
            result=delegation_result,
        )

        # Return delegation result as final response
        print(f"DEBUG: About to yield EventLoopStopEvent for delegation completion")
        yield EventLoopStopEvent(
            "delegation_complete", delegation_result.message, delegation_result.metrics, delegation_result.state
        )
        print(f"DEBUG: After yielding EventLoopStopEvent, about to return")
        return

    print("DEBUG: This should NOT be printed if delegation worked correctly")
    # Store parent cycle ID for the next cycle
    invocation_state["event_loop_parent_cycle_id"] = invocation_state["event_loop_cycle_id"]

    tool_result_message: Message = {
        "role": "user",
        "content": [{"toolResult": result} for result in tool_results],
    }

    agent.messages.append(tool_result_message)
    agent.hooks.invoke_callbacks(MessageAddedEvent(agent=agent, message=tool_result_message))
    yield ToolResultMessageEvent(message=tool_result_message)

    if cycle_span:
        tracer = get_tracer()
        tracer.end_event_loop_cycle_span(span=cycle_span, message=message, tool_result_message=tool_result_message)

    if invocation_state["request_state"].get("stop_event_loop", False):
        agent.event_loop_metrics.end_cycle(cycle_start_time, cycle_trace)
        yield EventLoopStopEvent(stop_reason, message, agent.event_loop_metrics, invocation_state["request_state"])
        return

    events = recurse_event_loop(agent=agent, invocation_state=invocation_state)
    async for event in events:
        yield event
