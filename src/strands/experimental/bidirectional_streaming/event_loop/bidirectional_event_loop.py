"""Bidirectional session management for concurrent streaming conversations.

Manages bidirectional communication sessions with concurrent processing of model events,
tool execution, and audio processing. Provides coordination between background tasks
while maintaining a simple interface for agent interaction.

Features:
- Concurrent task management for model events and tool execution
- Interruption handling with audio buffer clearing
- Tool execution with cancellation support
- Session lifecycle management
"""

import asyncio
import json
import logging
import traceback
import uuid

from ....tools._validator import validate_and_prepare_tools
from ....types.content import Message
from ....types.tools import ToolResult, ToolUse
from ..models.bidirectional_model import BidirectionalModelSession

logger = logging.getLogger(__name__)

# Session constants
TOOL_QUEUE_TIMEOUT = 0.5
SUPERVISION_INTERVAL = 0.1


class BidirectionalConnection:
    """Session wrapper for bidirectional communication with concurrent task management.

    Coordinates background tasks for model event processing, tool execution, and audio
    handling while providing a simple interface for agent interactions.
    """

    def __init__(self, model_session: BidirectionalModelSession, agent: "BidirectionalAgent") -> None:
        """Initialize session with model session and agent reference.

        Args:
            model_session: Provider-specific bidirectional model session.
            agent: BidirectionalAgent instance for tool registry access.
        """
        self.model_session = model_session
        self.agent = agent
        self.active = True

        # Background processing coordination
        self.background_tasks = []
        self.tool_queue = asyncio.Queue()
        self.audio_output_queue = asyncio.Queue()

        # Task management for cleanup
        self.pending_tool_tasks: dict[str, asyncio.Task] = {}

        # Interruption handling (model-agnostic)
        self.interrupted = False
        self.interruption_lock = asyncio.Lock()


async def start_bidirectional_connection(agent: "BidirectionalAgent") -> BidirectionalConnection:
    """Initialize bidirectional session with conycurrent background tasks.

    Creates a model-specific session and starts background tasks for processing
    model events, executing tools, and managing the session lifecycle.

    Args:
        agent: BidirectionalAgent instance.

    Returns:
        BidirectionalConnection: Active session with background tasks running.
    """
    logger.debug("Starting bidirectional session - initializing model session")

    # Create provider-specific session
    model_session = await agent.model.create_bidirectional_connection(
        system_prompt=agent.system_prompt, tools=agent.tool_registry.get_all_tool_specs(), messages=agent.messages
    )

    # Create session wrapper for background processing
    session = BidirectionalConnection(model_session=model_session, agent=agent)

    # Start concurrent background processors IMMEDIATELY after session creation
    # This is critical - Nova Sonic needs response processing during initialization
    logger.debug("Starting background processors for concurrent processing")
    session.background_tasks = [
        asyncio.create_task(_process_model_events(session)),  # Handle model responses
        asyncio.create_task(_process_tool_execution(session)),  # Execute tools concurrently
    ]

    # Start main coordination cycle
    session.main_cycle_task = asyncio.create_task(bidirectional_event_loop_cycle(session))

    logger.debug("Session ready with %d background tasks", len(session.background_tasks))
    return session


async def stop_bidirectional_connection(session: BidirectionalConnection) -> None:
    """End session and cleanup resources including background tasks.

    Args:
        session: BidirectionalConnection to cleanup.
    """
    if not session.active:
        return

    logger.debug("Session cleanup starting")
    session.active = False

    # Cancel pending tool tasks
    for _, task in session.pending_tool_tasks.items():
        if not task.done():
            task.cancel()

    # Cancel background tasks
    for task in session.background_tasks:
        if not task.done():
            task.cancel()

    # Cancel main cycle task
    if hasattr(session, "main_cycle_task") and not session.main_cycle_task.done():
        session.main_cycle_task.cancel()

    # Wait for tasks to complete
    all_tasks = session.background_tasks + list(session.pending_tool_tasks.values())
    if hasattr(session, "main_cycle_task"):
        all_tasks.append(session.main_cycle_task)

    if all_tasks:
        await asyncio.gather(*all_tasks, return_exceptions=True)

    # Close model session
    await session.model_session.close()
    logger.debug("Session closed")


async def bidirectional_event_loop_cycle(session: BidirectionalConnection) -> None:
    """Main event loop coordinator that runs continuously during the session.

    Monitors background tasks, manages session state, and handles session lifecycle.
    Provides supervision for concurrent model event processing and tool execution.

    Args:
        session: BidirectionalConnection to coordinate.
    """
    while session.active:
        try:
            # Check if background processors are still running
            if all(task.done() for task in session.background_tasks):
                logger.debug("Session end - all processors completed")
                session.active = False
                break

            # Check for failed background tasks
            for i, task in enumerate(session.background_tasks):
                if task.done() and not task.cancelled():
                    exception = task.exception()
                    if exception:
                        logger.error("Session error in processor %d: %s", i, str(exception))
                        session.active = False
                        raise exception

            # Brief pause before next supervision check
            await asyncio.sleep(SUPERVISION_INTERVAL)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Event loop error: %s", str(e))
            session.active = False
            raise


async def _handle_interruption(session: BidirectionalConnection) -> None:
    """Handle interruption detection with task cancellation and audio buffer clearing.

    Cancels pending tool tasks and clears audio output queues to ensure responsive
    interruption handling during conversations. Protected by async lock to prevent
    concurrent execution and race conditions.

    Args:
        session: BidirectionalConnection to handle interruption for.
    """
    async with session.interruption_lock:
        # If already interrupted, skip duplicate processing
        if session.interrupted:
            logger.debug("Interruption already in progress")
            return

        logger.debug("Interruption detected")
        session.interrupted = True

        # Cancel all pending tool execution tasks
        cancelled_tools = 0
        for task_id, task in list(session.pending_tool_tasks.items()):
            if not task.done():
                task.cancel()
                cancelled_tools += 1
                logger.debug("Tool task cancelled: %s", task_id)

        if cancelled_tools > 0:
            logger.debug("Tool tasks cancelled: %d", cancelled_tools)

        # Clear all queued audio output events
        cleared_count = 0
        while True:
            try:
                session.audio_output_queue.get_nowait()
                cleared_count += 1
            except asyncio.QueueEmpty:
                break

        # Import event types to avoid circular imports
        from ..types.bidirectional_streaming import AudioStreamEvent
        
        # Also clear the agent's audio output queue
        audio_cleared = 0
        # Create a temporary list to hold non-audio events
        temp_events = []
        try:
            while True:
                event = session.agent._output_queue.get_nowait()
                if isinstance(event, AudioStreamEvent):
                    audio_cleared += 1
                else:
                    # Keep non-audio events
                    temp_events.append(event)
        except asyncio.QueueEmpty:
            pass

        # Put back non-audio events
        for event in temp_events:
            session.agent._output_queue.put_nowait(event)

        if audio_cleared > 0:
            logger.debug("Agent audio queue cleared: %d events", audio_cleared)

        if cleared_count > 0:
            logger.debug("Session audio queue cleared: %d events", cleared_count)

        # Reset interruption flag after clearing (automatic recovery)
        session.interrupted = False
        logger.debug("Interruption handled - tools cancelled: %d, audio cleared: %d", cancelled_tools, cleared_count)


async def _process_model_events(session: BidirectionalConnection) -> None:
    """Process model events and convert them to Strands format.

    Background task that handles all model responses, converts provider-specific
    events to standardized formats, and manages interruption detection.

    Args:
        session: BidirectionalConnection containing model session.
    """
    logger.debug("Model events processor started")
    try:
        async for provider_event in session.model_session.receive_events():
            if not session.active:
                break

            # Import event types locally to avoid circular imports
            from ....types._events import ToolUseStreamEvent
            from ..types.bidirectional_streaming import (
                AudioStreamEvent,
                ErrorEvent,
                InterruptionEvent,
                MultimodalUsage,
                SessionEndEvent,
                SessionStartEvent,
                TranscriptStreamEvent,
                TurnCompleteEvent,
                TurnStartEvent,
            )

            # Unwrap dict-wrapped events from models (legacy format)
            if isinstance(provider_event, dict):
                # Extract the actual TypedEvent from the dict wrapper
                for key, value in provider_event.items():
                    if isinstance(value, (
                        SessionStartEvent, TurnStartEvent, AudioStreamEvent,
                        TranscriptStreamEvent, ToolUseStreamEvent, InterruptionEvent,
                        TurnCompleteEvent, MultimodalUsage, SessionEndEvent, ErrorEvent
                    )):
                        provider_event = value
                        break

            # Handle new TypedEvent instances using isinstance() checks
            if isinstance(provider_event, InterruptionEvent):
                logger.debug("Interruption detected: reason=%s, turn_id=%s", provider_event.reason, provider_event.turn_id)
                await _handle_interruption(session)
                # Forward interruption event to agent for application-level handling
                await session.agent._output_queue.put(provider_event)
                continue

            elif isinstance(provider_event, ToolUseStreamEvent):
                current_tool = provider_event.get("current_tool_use", {})
                logger.debug("Tool use requested: name=%s, tool_use_id=%s", current_tool.get("name"), current_tool.get("toolUseId"))
                # Queue tool for concurrent execution
                await session.tool_queue.put(provider_event)
                continue

            elif isinstance(provider_event, AudioStreamEvent):
                logger.debug("Audio stream chunk received: format=%s, sample_rate=%d", provider_event.format, provider_event.sample_rate)
                # Forward audio to agent output queue
                await session.agent._output_queue.put(provider_event)
                continue

            elif isinstance(provider_event, TranscriptStreamEvent):
                logger.debug("Transcript received: source=%s, is_final=%s, text=%s", provider_event.source, provider_event.is_final, provider_event.text[:50])
                # Forward transcript to agent output queue
                await session.agent._output_queue.put(provider_event)
                
                # Add final user transcripts to message history
                if provider_event.source == "user" and provider_event.is_final and provider_event.text.strip():
                    user_message = {"role": "user", "content": provider_event.text}
                    session.agent.messages.append(user_message)
                    logger.debug("User transcript added to history")
                continue

            elif isinstance(provider_event, TurnStartEvent):
                logger.debug("Turn started: turn_id=%s", provider_event.turn_id)
                # Forward to agent for turn tracking
                await session.agent._output_queue.put(provider_event)
                continue

            elif isinstance(provider_event, TurnCompleteEvent):
                logger.debug("Turn completed: turn_id=%s, stop_reason=%s", provider_event.turn_id, provider_event.stop_reason)
                # Forward to agent for turn tracking
                await session.agent._output_queue.put(provider_event)
                continue

            elif isinstance(provider_event, SessionStartEvent):
                logger.debug("Session started: session_id=%s, model=%s", provider_event.session_id, provider_event.model)
                await session.agent._output_queue.put(provider_event)
                continue

            elif isinstance(provider_event, SessionEndEvent):
                logger.debug("Session ended: reason=%s", provider_event.reason)
                await session.agent._output_queue.put(provider_event)
                continue

            elif isinstance(provider_event, MultimodalUsage):
                logger.debug("Usage event: input_tokens=%d, output_tokens=%d", provider_event.input_tokens, provider_event.output_tokens)
                await session.agent._output_queue.put(provider_event)
                continue

            elif isinstance(provider_event, ErrorEvent):
                logger.error("Error event: code=%s, message=%s", provider_event.code, provider_event.message)
                await session.agent._output_queue.put(provider_event)
                continue

    except Exception as e:
        logger.error("Model events error: %s", str(e))
        traceback.print_exc()
    finally:
        logger.debug("Model events processor stopped")


async def _process_tool_execution(session: BidirectionalConnection) -> None:
    """Execute tools concurrently with interruption support.

    Background task that manages tool execution without blocking model event
    processing or user interaction. Uses proper asyncio cancellation for 
    interruption handling rather than manual state checks.

    Args:
        session: BidirectionalConnection containing tool queue.
    """
    logger.debug("Tool execution processor started")
    while session.active:
        try:
            tool_use = await asyncio.wait_for(session.tool_queue.get(), timeout=TOOL_QUEUE_TIMEOUT)
            
            # Import event types locally to avoid circular imports
            from ....types._events import ToolUseStreamEvent
            
            # All tool uses should be ToolUseStreamEvent instances
            if not isinstance(tool_use, ToolUseStreamEvent):
                logger.error("Invalid tool use event type: %s", type(tool_use))
                continue
            
            # Extract tool info from current_tool_use
            current_tool_use = tool_use.get("current_tool_use", {})
            tool_name = current_tool_use.get("name", "")
            tool_id = current_tool_use.get("toolUseId", "")
            logger.debug("Tool execution started: %s (id: %s)", tool_name, tool_id)

            task_id = str(uuid.uuid4())
            task = asyncio.create_task(_execute_tool_with_strands(session, tool_use))
            session.pending_tool_tasks[task_id] = task

            def cleanup_task(completed_task: asyncio.Task, task_id: str = task_id) -> None:
                try:
                    # Remove from pending tasks
                    if task_id in session.pending_tool_tasks:
                        del session.pending_tool_tasks[task_id]

                    # Log completion status
                    if completed_task.cancelled():
                        logger.debug("Tool task cleanup cancelled: %s", task_id)
                    elif completed_task.exception():
                        logger.error("Tool task cleanup error: %s - %s", task_id, str(completed_task.exception()))
                    else:
                        logger.debug("Tool task cleanup success: %s", task_id)
                except Exception as e:
                    logger.error("Tool task cleanup failed: %s - %s", task_id, str(e))

            task.add_done_callback(cleanup_task)

        except asyncio.TimeoutError:
            if not session.active:
                break
            # Remove completed tasks from tracking
            completed_tasks = [task_id for task_id, task in session.pending_tool_tasks.items() if task.done()]
            for task_id in completed_tasks:
                if task_id in session.pending_tool_tasks:
                    del session.pending_tool_tasks[task_id]

            if completed_tasks:
                logger.debug("Periodic task cleanup: %d tasks", len(completed_tasks))

            continue
        except Exception as e:
            logger.error("Tool execution error: %s", str(e))
            if not session.active:
                break

    logger.debug("Tool execution processor stopped")





async def _execute_tool_with_strands(session: BidirectionalConnection, tool_use: "ToolUseStreamEvent") -> None:
    """Execute tool using Strands infrastructure with interruption support.

    Executes tools using the existing Strands tool system with proper asyncio
    cancellation handling. Tool execution is stopped via task cancellation,
    not manual state checks.

    Args:
        session: BidirectionalConnection for context.
        tool_use: Tool use event to execute (ToolUseStreamEvent instance).
    """
    # Import event types locally to avoid circular imports
    from ....types._events import ToolResultEvent
    
    # Extract tool information from ToolUseStreamEvent
    current_tool_use = tool_use.get("current_tool_use", {})
    tool_name = current_tool_use.get("name", "")
    tool_id = current_tool_use.get("toolUseId", "")
    tool_input_str = current_tool_use.get("input", "{}")
    
    # Parse tool input from JSON string
    import json
    try:
        tool_input = json.loads(tool_input_str) if tool_input_str else {}
    except json.JSONDecodeError:
        logger.error("Failed to parse tool input JSON: %s", tool_input_str)
        tool_input = {}
    
    # Convert to dict for Strands tool system
    tool_use_dict = {
        "toolUseId": tool_id,
        "name": tool_name,
        "input": tool_input,
    }

    try:
        # Create message structure for existing tool system
        tool_message: Message = {"role": "assistant", "content": [{"toolUse": tool_use_dict}]}

        tool_uses: list[ToolUse] = []
        tool_results: list[ToolResult] = []
        invalid_tool_use_ids: list[str] = []

        # Validate using existing Strands validation
        validate_and_prepare_tools(tool_message, tool_uses, tool_results, invalid_tool_use_ids)

        # Filter valid tool uses
        valid_tool_uses = [tu for tu in tool_uses if tu.get("toolUseId") not in invalid_tool_use_ids]

        if not valid_tool_uses:
            logger.warning("Tool validation failed: %s (id: %s)", tool_name, tool_id)
            return

        # Execute tools directly (simpler approach for bidirectional)
        for tool_use_item in valid_tool_uses:
            tool_func = session.agent.tool_registry.registry.get(tool_use_item["name"])

            if tool_func:
                try:
                    actual_func = _extract_callable_function(tool_func)

                    # Execute tool function with provided input
                    result = actual_func(**tool_use_item.get("input", {}))

                    tool_result = _create_success_result(tool_use_item["toolUseId"], result)
                    tool_results.append(tool_result)

                except Exception as e:
                    logger.error("Tool execution failed: %s - %s", tool_name, str(e))
                    tool_result = _create_error_result(tool_use_item["toolUseId"], str(e))
                    tool_results.append(tool_result)
            else:
                logger.warning("Tool not found: %s", tool_name)

        # Send results through provider-specific session using unified send() method
        for result in tool_results:
            # ToolResultEvent expects a ToolResult dict (which already contains toolUseId)
            tool_result_event = ToolResultEvent(tool_result=result)
            await session.model_session.send(tool_result_event)

        logger.debug("Tool execution completed: %s (%d results)", tool_name, len(tool_results))

    except asyncio.CancelledError:
        # Task was cancelled due to interruption - this is expected behavior
        logger.debug("Tool task cancelled gracefully: %s (id: %s)", tool_name, tool_id)
        raise  # Re-raise to properly handle cancellation
    except Exception as e:
        logger.error("Tool execution error: %s - %s", tool_name, str(e))
        
        try:
            # Create error result with toolUseId
            error_result = {
                "toolUseId": tool_id,
                "status": "error",
                "content": [{"text": f"Error: {str(e)}"}]
            }
            error_result_event = ToolResultEvent(tool_result=error_result)
            await session.model_session.send(error_result_event)
        except Exception as send_error:
            logger.error("Tool error send failed: %s", str(send_error))


def _extract_callable_function(tool_func: any) -> any:
    """Extract the callable function from different tool object types."""
    if hasattr(tool_func, "_tool_func"):
        return tool_func._tool_func
    elif hasattr(tool_func, "func"):
        return tool_func.func
    elif callable(tool_func):
        return tool_func
    else:
        raise ValueError(f"Tool function not callable: {type(tool_func).__name__}")


def _create_success_result(tool_use_id: str, result: any) -> dict[str, any]:
    """Create a successful tool result."""
    return {"toolUseId": tool_use_id, "status": "success", "content": [{"text": json.dumps(result)}]}


def _create_error_result(tool_use_id: str, error: str) -> dict[str, any]:
    """Create an error tool result."""
    return {"toolUseId": tool_use_id, "status": "error", "content": [{"text": f"Error: {error}"}]}
