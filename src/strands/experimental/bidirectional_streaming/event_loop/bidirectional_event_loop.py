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
from typing import Any, Dict

from ....tools._validator import validate_and_prepare_tools
from ....types.content import Message
from ....types.tools import ToolResult, ToolUse
from ..agent.agent import BidirectionalAgent
from ..models.bidirectional_model import BidirectionalModelSession
from ..utils.debug import log_event, log_flow

logger = logging.getLogger(__name__)

# Session constants
TOOL_QUEUE_TIMEOUT = 0.5
SUPERVISION_INTERVAL = 0.1


class BidirectionalConnection:
    """Session wrapper for bidirectional communication with concurrent task management.

    Coordinates background tasks for model event processing, tool execution, and audio
    handling while providing a simple interface for agent interactions.
    """

    def __init__(self, model_session: BidirectionalModelSession, agent):
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
        self.pending_tool_tasks: Dict[str, asyncio.Task] = {}

        # Interruption handling (model-agnostic)
        self.interrupted = False


async def start_bidirectional_connection(agent: BidirectionalAgent) -> BidirectionalConnection:
    """Initialize bidirectional session with concurrent background tasks.

    Creates a model-specific session and starts background tasks for processing
    model events, executing tools, and managing the session lifecycle.

    Args:
        agent: BidirectionalAgent instance.

    Returns:
        BidirectionalConnection: Active session with background tasks running.
    """
    log_flow("session_start", "initializing model session")

    # Create provider-specific session
    model_session = await agent.model.create_bidirectional_connection(
        system_prompt=agent.system_prompt, tools=agent.tool_registry.get_all_tool_specs(), messages=agent.messages
    )

    # Create session wrapper for background processing
    session = BidirectionalConnection(model_session=model_session, agent=agent)

    # Start concurrent background processors IMMEDIATELY after session creation
    # This is critical - Nova Sonic needs response processing during initialization
    log_flow("background_tasks", "starting processors")
    session.background_tasks = [
        asyncio.create_task(_process_model_events(session)),  # Handle model responses
        asyncio.create_task(_process_tool_execution(session)),  # Execute tools concurrently
    ]

    # Start main coordination cycle
    session.main_cycle_task = asyncio.create_task(bidirectional_event_loop_cycle(session))

    # Give background tasks a moment to start
    await asyncio.sleep(0.1)
    log_event("session_ready", tasks=len(session.background_tasks))

    return session


async def stop_bidirectional_connection(session: BidirectionalConnection) -> None:
    """End session and cleanup resources including background tasks.

    Args:
        session: BidirectionalConnection to cleanup.
    """
    if not session.active:
        return

    log_flow("session_cleanup", "starting")
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
    log_event("session_closed")


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
                log_event("session_end", reason="all_processors_completed")
                session.active = False
                break

            # Check for failed background tasks
            for i, task in enumerate(session.background_tasks):
                if task.done() and not task.cancelled():
                    exception = task.exception()
                    if exception:
                        log_event("session_error", processor=i, error=str(exception))
                        session.active = False
                        raise exception

            # Brief pause before next supervision check
            await asyncio.sleep(SUPERVISION_INTERVAL)

        except asyncio.CancelledError:
            break
        except Exception as e:
            log_event("event_loop_error", error=str(e))
            session.active = False
            raise


async def _handle_interruption(session: BidirectionalConnection) -> None:
    """Handle interruption detection with task cancellation and audio buffer clearing.

    Cancels pending tool tasks and clears audio output queues to ensure responsive
    interruption handling during conversations.

    Args:
        session: BidirectionalConnection to handle interruption for.
    """
    log_event("interruption_detected")
    session.interrupted = True

    # 🔥 CANCEL ALL PENDING TOOL TASKS (Nova Sonic pattern)
    cancelled_tools = 0
    for task_id, task in list(session.pending_tool_tasks.items()):
        if not task.done():
            task.cancel()
            cancelled_tools += 1
            log_event("tool_task_cancelled", task_id=task_id)

    if cancelled_tools > 0:
        log_event("tool_tasks_cancelled", count=cancelled_tools)

    # 🔥 AGGRESSIVELY CLEAR AUDIO OUTPUT QUEUE (Nova Sonic pattern)
    cleared_count = 0
    while True:
        try:
            session.audio_output_queue.get_nowait()
            cleared_count += 1
        except asyncio.QueueEmpty:
            break

    # Also clear the agent's audio output queue if it exists
    if hasattr(session.agent, "_output_queue"):
        audio_cleared = 0
        # Create a temporary list to hold non-audio events
        temp_events = []
        try:
            while True:
                event = session.agent._output_queue.get_nowait()
                if event.get("audioOutput"):
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
            log_event("agent_audio_queue_cleared", count=audio_cleared)

    if cleared_count > 0:
        log_event("session_audio_queue_cleared", count=cleared_count)

    # Brief sleep to allow audio system to settle (matches Nova Sonic timing)
    await asyncio.sleep(0.05)

    # Reset interruption flag after clearing (automatic recovery)
    session.interrupted = False
    log_event("interruption_handled", tools_cancelled=cancelled_tools, audio_cleared=cleared_count)


async def _process_model_events(session: BidirectionalConnection) -> None:
    """Process model events and convert them to Strands format.

    Background task that handles all model responses, converts provider-specific
    events to standardized formats, and manages interruption detection.

    Args:
        session: BidirectionalConnection containing model session.
    """
    log_flow("model_events", "processor started")
    try:
        async for provider_event in session.model_session.receive_events():
            if not session.active:
                break

            # Convert provider events to Strands format
            strands_event = _convert_to_strands_event(provider_event)

            # Handle interruption detection (multiple patterns)
            if strands_event.get("interruptionDetected"):
                log_event("interruption_forwarded")
                await _handle_interruption(session)
                # Forward interruption event to agent for application-level handling
                await session.agent._output_queue.put(strands_event)
                continue

            # Check for text-based interruption (Nova Sonic pattern)
            if strands_event.get("textOutput"):
                text_content = strands_event["textOutput"].get("content", "")
                if '{ "interrupted" : true }' in text_content:
                    log_event("text_interruption_detected")
                    await _handle_interruption(session)
                    # Still forward the text event
                    await session.agent._output_queue.put(strands_event)
                    continue

            # Queue tool requests for concurrent execution
            if strands_event.get("toolUse"):
                log_event("tool_queued", name=strands_event["toolUse"].get("name"))
                await session.tool_queue.put(strands_event["toolUse"])
                continue

            # Send output events to Agent for receive() method
            if strands_event.get("audioOutput") or strands_event.get("textOutput"):
                await session.agent._output_queue.put(strands_event)

            # Update Agent conversation history using existing patterns
            if strands_event.get("messageStop"):
                log_event("message_added_to_history")
                session.agent.messages.append(strands_event["messageStop"]["message"])

    except Exception as e:
        log_event("model_events_error", error=str(e))
        traceback.print_exc()
    finally:
        log_flow("model_events", "processor stopped")


async def _process_tool_execution(session: BidirectionalConnection) -> None:
    """Execute tools concurrently with interruption support.

    Background task that manages tool execution without blocking model event
    processing or user interaction. Includes proper task cleanup and cancellation
    handling for interruptions.

    Args:
        session: BidirectionalConnection containing tool queue.
    """
    log_flow("tool_execution", "processor started")
    while session.active:
        try:
            tool_use = await asyncio.wait_for(session.tool_queue.get(), timeout=TOOL_QUEUE_TIMEOUT)
            log_event("tool_execution_started", name=tool_use.get("name"), id=tool_use.get("toolUseId"))

            if not session.active:
                break

            task_id = str(uuid.uuid4())
            task = asyncio.create_task(_execute_tool_with_strands(session, tool_use))
            session.pending_tool_tasks[task_id] = task

            # 🔥 ADD CLEANUP CALLBACK (Nova Sonic pattern)
            def cleanup_task(completed_task, task_id=task_id):
                try:
                    # Remove from pending tasks
                    if task_id in session.pending_tool_tasks:
                        del session.pending_tool_tasks[task_id]

                    # Log completion status
                    if completed_task.cancelled():
                        log_event("tool_task_cleanup_cancelled", task_id=task_id)
                    elif completed_task.exception():
                        log_event("tool_task_cleanup_error", task_id=task_id, error=str(completed_task.exception()))
                    else:
                        log_event("tool_task_cleanup_success", task_id=task_id)
                except Exception as e:
                    log_event("tool_task_cleanup_failed", task_id=task_id, error=str(e))

            task.add_done_callback(cleanup_task)

        except asyncio.TimeoutError:
            if not session.active:
                break
            # 🔥 PERIODIC CLEANUP OF COMPLETED TASKS
            completed_tasks = [task_id for task_id, task in session.pending_tool_tasks.items() if task.done()]
            for task_id in completed_tasks:
                if task_id in session.pending_tool_tasks:
                    del session.pending_tool_tasks[task_id]

            if completed_tasks:
                log_event("periodic_task_cleanup", count=len(completed_tasks))

            continue
        except Exception as e:
            log_event("tool_execution_error", error=str(e))
            if not session.active:
                break

    log_flow("tool_execution", "processor stopped")


def _convert_to_strands_event(provider_event: Dict) -> Dict:
    """Pass-through for events already normalized by provider sessions.

    Providers convert their raw events to standard format before reaching here.
    This just validates and passes through the normalized events.

    Args:
        provider_event: Already normalized event from provider session.

    Returns:
        Dict: The same event, validated and passed through.
    """
    # Basic validation - ensure we have a dict
    if not isinstance(provider_event, dict):
        return {}

    # Pass through - conversion already done by provider session
    return provider_event


async def _execute_tool_with_strands(session: BidirectionalConnection, tool_use: Dict) -> None:
    """Execute tool using Strands infrastructure with interruption support.

    Executes tools using the existing Strands tool system, handles interruption
    during execution, and sends results back to the model provider.

    Args:
        session: BidirectionalConnection for context.
        tool_use: Tool use event to execute.
    """
    tool_name = tool_use.get("name")
    tool_id = tool_use.get("toolUseId")

    try:
        # 🔥 CHECK FOR INTERRUPTION BEFORE STARTING (Nova Sonic pattern)
        if session.interrupted or not session.active:
            log_event("tool_execution_cancelled_before_start", name=tool_name, id=tool_id)
            return

        # Create message structure for existing tool system
        tool_message: Message = {"role": "assistant", "content": [{"toolUse": tool_use}]}

        tool_uses: list[ToolUse] = []
        tool_results: list[ToolResult] = []
        invalid_tool_use_ids: list[str] = []

        # Validate using existing Strands validation
        validate_and_prepare_tools(tool_message, tool_uses, tool_results, invalid_tool_use_ids)

        # Filter valid tool uses
        valid_tool_uses = [tu for tu in tool_uses if tu.get("toolUseId") not in invalid_tool_use_ids]

        if not valid_tool_uses:
            log_event("tool_validation_failed", name=tool_name, id=tool_id)
            return

        # Execute tools directly (simpler approach for bidirectional)
        for tool_use in valid_tool_uses:
            # 🔥 CHECK FOR INTERRUPTION DURING EXECUTION
            if session.interrupted or not session.active:
                log_event("tool_execution_cancelled_during", name=tool_name, id=tool_id)
                return

            tool_func = session.agent.tool_registry.registry.get(tool_use["name"])

            if tool_func:
                try:
                    actual_func = _extract_callable_function(tool_func)

                    # 🔥 WRAP TOOL EXECUTION IN CANCELLATION CHECK
                    # For async tools, we could wrap with asyncio.wait_for with cancellation
                    # For sync tools, we execute directly but check interruption after
                    result = actual_func(**tool_use.get("input", {}))

                    # 🔥 CHECK FOR INTERRUPTION AFTER TOOL EXECUTION
                    if session.interrupted or not session.active:
                        log_event("tool_result_discarded_interruption", name=tool_name, id=tool_id)
                        return

                    tool_result = _create_success_result(tool_use["toolUseId"], result)
                    tool_results.append(tool_result)

                except asyncio.CancelledError:
                    # Tool was cancelled due to interruption
                    log_event("tool_execution_cancelled", name=tool_name, id=tool_id)
                    return
                except Exception as e:
                    # 🔥 CHECK FOR INTERRUPTION EVEN ON ERROR
                    if session.interrupted or not session.active:
                        log_event("tool_error_discarded_interruption", name=tool_name, id=tool_id)
                        return

                    log_event("tool_execution_failed", name=tool_name, error=str(e))
                    tool_result = _create_error_result(tool_use["toolUseId"], str(e))
                    tool_results.append(tool_result)
            else:
                log_event("tool_not_found", name=tool_name)

        # 🔥 FINAL INTERRUPTION CHECK BEFORE SENDING RESULTS
        if session.interrupted or not session.active:
            log_event("tool_results_discarded_interruption", name=tool_name, count=len(tool_results))
            return

        # Send results through provider-specific session
        for result in tool_results:
            await session.model_session.send_tool_result(tool_use.get("toolUseId"), result)

        log_event("tool_execution_completed", name=tool_name, results=len(tool_results))

    except asyncio.CancelledError:
        # Task was cancelled due to interruption - this is expected behavior
        log_event("tool_task_cancelled_gracefully", name=tool_name, id=tool_id)
        raise  # Re-raise to properly handle cancellation
    except Exception as e:
        log_event("tool_execution_error", name=tool_use.get("name"), error=str(e))

        # Only send error if not interrupted
        if not session.interrupted and session.active:
            try:
                await session.model_session.send_tool_error(tool_use.get("toolUseId"), str(e))
            except Exception as send_error:
                log_event("tool_error_send_failed", error=str(send_error))


def _extract_callable_function(tool_func):
    """Extract the callable function from different tool object types."""
    if hasattr(tool_func, "_tool_func"):
        return tool_func._tool_func
    elif hasattr(tool_func, "func"):
        return tool_func.func
    elif callable(tool_func):
        return tool_func
    else:
        raise ValueError(f"Tool function not callable: {type(tool_func).__name__}")


def _create_success_result(tool_use_id: str, result) -> Dict[str, Any]:
    """Create a successful tool result."""
    return {"toolUseId": tool_use_id, "status": "success", "content": [{"text": json.dumps(result)}]}


def _create_error_result(tool_use_id: str, error: str) -> Dict[str, Any]:
    """Create an error tool result."""
    return {"toolUseId": tool_use_id, "status": "error", "content": [{"text": f"Error: {error}"}]}
