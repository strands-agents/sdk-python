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
import logging
import traceback
import uuid

from ....tools._validator import validate_and_prepare_tools
from ....telemetry.metrics import Trace
from ....types._events import ToolResultEvent, ToolStreamEvent
from ....types.content import Message
from ....types.tools import ToolResult, ToolUse
from ..models.bidirectional_model import BidirectionalModel

logger = logging.getLogger(__name__)

# Session constants
TOOL_QUEUE_TIMEOUT = 0.5
SUPERVISION_INTERVAL = 0.1


class BidirectionalConnection:
    """Session wrapper for bidirectional communication with concurrent task management.

    Coordinates background tasks for model event processing, tool execution, and audio
    handling while providing a simple interface for agent interactions.
    """

    def __init__(self, model: BidirectionalModel, agent: "BidirectionalAgent") -> None:
        """Initialize connection with model and agent reference.

        Args:
            model: Bidirectional model instance.
            agent: BidirectionalAgent instance for tool registry access.
        """
        self.model = model
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
        
        # Tool execution tracking
        self.tool_count = 0


async def start_bidirectional_connection(agent: "BidirectionalAgent") -> BidirectionalConnection:
    """Initialize bidirectional session with conycurrent background tasks.

    Creates a model-specific session and starts background tasks for processing
    model events, executing tools, and managing the session lifecycle.

    Args:
        agent: BidirectionalAgent instance.

    Returns:
        BidirectionalConnection: Active session with background tasks running.
    """
    logger.debug("Starting bidirectional session - initializing model connection")

    # Connect to model
    await agent.model.connect(
        system_prompt=agent.system_prompt, tools=agent.tool_registry.get_all_tool_specs(), messages=agent.messages
    )

    # Create connection wrapper for background processing
    session = BidirectionalConnection(model=agent.model, agent=agent)

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

    # Close model connection
    await session.model.close()
    logger.debug("Connection closed")


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
        for _task_id, task in list(session.pending_tool_tasks.items()):
            if not task.done():
                task.cancel()
                cancelled_tools += 1
                logger.debug("Tool task cancelled: %s", _task_id)

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

        # Also clear the agent's audio output queue
        audio_cleared = 0
        # Create a temporary list to hold non-audio events
        temp_events = []
        try:
            while True:
                event = session.agent._output_queue.get_nowait()
                # Check for audio events
                event_type = event.get("type", "")
                if event_type == "bidirectional_audio_stream":
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
        session: BidirectionalConnection containing model.
    """
    logger.debug("Model events processor started")
    try:
        async for provider_event in session.model.receive():
            if not session.active:
                break

            # Basic validation - skip invalid events
            if not isinstance(provider_event, dict):
                continue
            
            strands_event = provider_event

            # Get event type
            event_type = strands_event.get("type", "")
            
            # Handle interruption detection
            if event_type == "bidirectional_interruption":
                logger.debug("Interruption forwarded")
                await _handle_interruption(session)
                # Forward interruption event to agent for application-level handling
                await session.agent._output_queue.put(strands_event)
                continue

            # Queue tool requests for concurrent execution
            # Check for ToolUseStreamEvent (standard agent event)
            if "current_tool_use" in strands_event:
                tool_use = strands_event.get("current_tool_use")
                if tool_use:
                    tool_name = tool_use.get("name")
                    logger.debug("Tool usage detected: %s", tool_name)
                    await session.tool_queue.put(tool_use)
                # Forward ToolUseStreamEvent to output queue for client visibility
                await session.agent._output_queue.put(strands_event)
                continue

            # Send all output events to Agent for receive() method
            await session.agent._output_queue.put(strands_event)

            # Update Agent conversation history for user transcripts
            if event_type == "bidirectional_transcript_stream":
                source = strands_event.get("source")
                text = strands_event.get("text", "")
                if source == "user" and text.strip():
                    user_message = {"role": "user", "content": text}
                    session.agent.messages.append(user_message)
                    logger.debug("User transcript added to history")

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
            tool_name = tool_use.get("name")
            tool_id = tool_use.get("toolUseId")
            
            session.tool_count += 1
            print(f"\nTool #{session.tool_count}: {tool_name}")
            
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
                        logger.debug("Tool task cancelled: %s", task_id)
                    elif completed_task.exception():
                        logger.error("Tool task error: %s - %s", task_id, str(completed_task.exception()))
                    else:
                        logger.debug("Tool task completed: %s", task_id)
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





async def _execute_tool_with_strands(session: BidirectionalConnection, tool_use: dict) -> None:
    """Execute tool using the complete Strands tool execution system.
    
    Uses proper Strands ToolExecutor system with validation, error handling,
    and event streaming.
    
    Args:
        session: BidirectionalConnection for context.
        tool_use: Tool use event to execute.
    """
    tool_name = tool_use.get("name")
    tool_id = tool_use.get("toolUseId")
    
    logger.debug("Executing tool: %s (id: %s)", tool_name, tool_id)
    
    try:
        # Create message structure for validation 
        tool_message: Message = {"role": "assistant", "content": [{"toolUse": tool_use}]}
        
        # Use Strands validation system
        tool_uses: list[ToolUse] = []
        tool_results: list[ToolResult] = []
        invalid_tool_use_ids: list[str] = []
        
        validate_and_prepare_tools(tool_message, tool_uses, tool_results, invalid_tool_use_ids)
        
        # Filter valid tools
        valid_tool_uses = [tu for tu in tool_uses if tu.get("toolUseId") not in invalid_tool_use_ids]
        
        if not valid_tool_uses:
            logger.warning("No valid tools after validation: %s", tool_name)
            return
        
        # Create invocation state for tool execution
        invocation_state = {
            "agent": session.agent,
            "model": session.agent.model,
            "messages": session.agent.messages,
            "system_prompt": session.agent.system_prompt,
        }
        
        # Create cycle trace and span
        cycle_trace = Trace("Bidirectional Tool Execution")
        cycle_span = None
        
        tool_events = session.agent.tool_executor._execute(
            session.agent,
            valid_tool_uses,
            tool_results,
            cycle_trace,
            cycle_span,
            invocation_state
        )
        
        # Process tool events and send results to provider
        async for tool_event in tool_events:
            if isinstance(tool_event, ToolResultEvent):
                tool_result = tool_event.tool_result
                tool_use_id = tool_result.get("toolUseId")
                
                # Send ToolResultEvent through send() method to model
                await session.model.send(tool_event)
                logger.debug("Tool result sent to model: %s", tool_use_id)
                
                # Also forward ToolResultEvent to output queue for client visibility
                await session.agent._output_queue.put(tool_event.as_dict())
                logger.debug("Tool result sent to client: %s", tool_use_id)
                
            # Handle streaming events if needed later
            elif isinstance(tool_event, ToolStreamEvent):
                logger.debug("Tool stream event: %s", tool_event)
                # Forward tool stream events to output queue
                await session.agent._output_queue.put(tool_event.as_dict())
        
        # Add tool result message to conversation history
        if tool_results:
            from ....hooks import MessageAddedEvent
            
            tool_result_message: Message = {
                "role": "user",
                "content": [{"toolResult": result} for result in tool_results],
            }
            
            session.agent.messages.append(tool_result_message)
            session.agent.hooks.invoke_callbacks(MessageAddedEvent(agent=session.agent, message=tool_result_message))
            logger.debug("Tool result message added to history: %s", tool_name)
        
        logger.debug("Tool execution completed: %s", tool_name)
        
    except asyncio.CancelledError:
        logger.debug("Tool execution cancelled: %s (id: %s)", tool_name, tool_id)
        raise
    except Exception as e:
        logger.error("Tool execution error: %s - %s", tool_name, str(e))
        
        # Send error result wrapped in ToolResultEvent
        error_result: ToolResult = {
            "toolUseId": tool_id,
            "status": "error",
            "content": [{"text": f"Error: {str(e)}"}]
        }
        try:
            await session.model.send(ToolResultEvent(error_result))
            logger.debug("Error result sent: %s", tool_id)
        except Exception as send_error:
            logger.error("Failed to send error result: %s - %s", tool_id, str(send_error))
            raise  # Propagate exception since this is experimental code


