"""Agent loop.

The agent loop handles the events received from the model and executes tools when given a tool use request.
"""

import asyncio
import logging
import time
import warnings
from typing import TYPE_CHECKING, Any, AsyncGenerator, cast

from opentelemetry.trace import Span

from ....telemetry.tracer import get_tracer
from ....types._events import ToolInterruptEvent, ToolResultEvent, ToolResultMessageEvent, ToolUseStreamEvent
from ....types.content import Message
from ....types.tools import ToolResult, ToolUse
from ...hooks.events import (
    BidiAfterConnectionRestartEvent,
    BidiAfterInvocationEvent,
    BidiBeforeConnectionRestartEvent,
    BidiBeforeInvocationEvent,
)
from ...hooks.events import (
    BidiInterruptionEvent as BidiInterruptionHookEvent,
)
from .. import _telemetry
from .._async import _TaskPool, stop_all
from ..models import BidiModelTimeoutError
from ..types.events import (
    BidiAudioStreamEvent,
    BidiConnectionCloseEvent,
    BidiConnectionRestartEvent,
    BidiInputEvent,
    BidiInterruptionEvent,
    BidiOutputEvent,
    BidiResponseCompleteEvent,
    BidiResponseStartEvent,
    BidiTextInputEvent,
    BidiTranscriptStreamEvent,
    BidiUsageEvent,
)

if TYPE_CHECKING:
    from .agent import BidiAgent

logger = logging.getLogger(__name__)


class _BidiAgentLoop:
    """Agent loop.

    Attributes:
        _agent: BidiAgent instance to loop.
        _started: Flag if agent loop has started.
        _task_pool: Track active async tasks created in loop.
        _event_queue: Queue model and tool call events for receiver.
        _invocation_state: Optional context to pass to tools during execution.
            This allows passing custom data (user_id, session_id, database connections, etc.)
            that tools can access via their invocation_state parameter.
        _send_gate: Gate the sending of events to the model.
            Blocks when agent is resetting the model connection after timeout.
    """

    def __init__(self, agent: "BidiAgent") -> None:
        """Initialize members of the agent loop.

        Note, before receiving events from the loop, the user must call `start`.

        Args:
            agent: Bidirectional agent to loop over.
        """
        self._agent = agent
        self._started = False
        self._task_pool = _TaskPool()
        self._event_queue: asyncio.Queue
        self._invocation_state: dict[str, Any]

        self._send_gate = asyncio.Event()

        self._tracer = get_tracer()
        self._session_span: Span | None = None
        self._accumulated_input_tokens: int
        self._accumulated_output_tokens: int
        self._accumulated_total_tokens: int
        self._accumulated_cache_read_tokens: int

    async def start(self, invocation_state: dict[str, Any] | None = None) -> None:
        """Start the agent loop.

        The agent model is started as part of this call.

        Args:
            invocation_state: Optional context to pass to tools during execution.
                This allows passing custom data (user_id, session_id, database connections, etc.)
                that tools can access via their invocation_state parameter.

        Raises:
            RuntimeError: If loop already started.
        """
        if self._started:
            raise RuntimeError("loop already started | call stop before starting again")

        logger.debug("agent loop starting")
        await self._agent.hooks.invoke_callbacks_async(BidiBeforeInvocationEvent(agent=self._agent))

        model_id = getattr(self._agent.model, "model_id", None)

        self._session_span = _telemetry.start_session_span(
            self._tracer,
            agent_name=self._agent.name,
            model_id=model_id,
            tools=self._agent.tool_names,
            system_prompt=self._agent.system_prompt,
        )

        connection_span = _telemetry.start_connection_span(
            self._tracer, parent_span=self._session_span, model_id=model_id
        )
        try:
            await self._agent.model.start(
                system_prompt=self._agent.system_prompt,
                tools=self._agent.tool_registry.get_all_tool_specs(),
                messages=self._agent.messages,
            )
        except Exception as error:
            _telemetry.end_connection_span(self._tracer, connection_span, error=error)
            _telemetry.end_session_span(self._tracer, self._session_span, error=error)
            self._session_span = None
            raise
        _telemetry.end_connection_span(self._tracer, connection_span)
        self._accumulated_input_tokens = 0
        self._accumulated_output_tokens = 0
        self._accumulated_total_tokens = 0
        self._accumulated_cache_read_tokens = 0

        self._event_queue = asyncio.Queue(maxsize=1)

        self._task_pool = _TaskPool()
        self._task_pool.create(self._run_model())

        self._invocation_state = invocation_state or {}
        self._send_gate.set()
        self._started = True

    async def stop(self) -> None:
        """Stop the agent loop."""
        logger.debug("agent loop stopping")

        self._started = False
        self._send_gate.clear()
        self._invocation_state = {}

        async def stop_tasks() -> None:
            await self._task_pool.cancel()

        async def stop_model() -> None:
            await self._agent.model.stop()

        try:
            await stop_all(stop_tasks, stop_model)
        finally:
            if self._session_span:
                _telemetry.end_session_span(
                    self._tracer,
                    self._session_span,
                    input_tokens=self._accumulated_input_tokens,
                    output_tokens=self._accumulated_output_tokens,
                    total_tokens=self._accumulated_total_tokens,
                    cache_read_input_tokens=self._accumulated_cache_read_tokens,
                )
                self._session_span = None

            await self._agent.hooks.invoke_callbacks_async(BidiAfterInvocationEvent(agent=self._agent))

    async def send(self, event: BidiInputEvent | ToolResultEvent) -> None:
        """Send model event.

        Additionally, add text input to messages array.

        Args:
            event: User input event or tool result.

        Raises:
            RuntimeError: If start has not been called.
        """
        if not self._started:
            raise RuntimeError("loop not started | call start before sending")

        if not self._send_gate.is_set():
            logger.debug("waiting for model send signal")
            await self._send_gate.wait()

        if isinstance(event, BidiTextInputEvent):
            message: Message = {"role": event.role, "content": [{"text": event.text}]}
            await self._agent._append_messages(message)

        await self._agent.model.send(event)

    async def receive(self) -> AsyncGenerator[BidiOutputEvent, None]:
        """Receive model and tool call events.

        Returns:
            Model and tool call events.

        Raises:
            RuntimeError: If start has not been called.
        """
        if not self._started:
            raise RuntimeError("loop not started | call start before receiving")

        while True:
            event = await self._event_queue.get()
            if isinstance(event, BidiModelTimeoutError):
                logger.debug("model timeout error received")
                yield BidiConnectionRestartEvent(event)
                await self._restart_connection(event)
                continue

            if isinstance(event, Exception):
                raise event

            # Check for graceful shutdown event
            if isinstance(event, BidiConnectionCloseEvent) and event.reason == "user_request":
                yield event
                break

            yield event

    async def _restart_connection(self, timeout_error: BidiModelTimeoutError) -> None:
        """Restart the model connection after timeout.

        Args:
            timeout_error: Timeout error reported by the model.
        """
        logger.debug("resetting model connection")

        self._send_gate.clear()

        # The before-restart hook runs before the restart begins; per the hook contract its
        # exceptions propagate to the caller (receive()), leaving the send gate closed.
        await self._agent.hooks.invoke_callbacks_async(BidiBeforeConnectionRestartEvent(self._agent, timeout_error))

        restart_span = _telemetry.start_restart_span(
            self._tracer, parent_span=self._session_span, error_message=str(timeout_error)
        )

        restart_exception = None
        try:
            await self._agent.model.stop()
            await self._agent.model.start(
                self._agent.system_prompt,
                self._agent.tool_registry.get_all_tool_specs(),
                self._agent.messages,
                **timeout_error.restart_config,
            )
            self._task_pool.create(self._run_model())
        except Exception as exception:
            restart_exception = exception
        finally:
            _telemetry.end_restart_span(self._tracer, restart_span, error=restart_exception)

            await self._agent.hooks.invoke_callbacks_async(
                BidiAfterConnectionRestartEvent(self._agent, restart_exception)
            )

        # A failed restart leaves no running model task and a closed send gate, so surface it
        # to the receive() consumer rather than idling silently. Telemetry and the after-restart
        # hook have already reported the error above.
        if restart_exception is not None:
            raise restart_exception

        self._send_gate.set()

    async def _run_model(self) -> None:
        """Task for running the model.

        Events are streamed through the event queue.
        """
        logger.debug("model task starting")

        response_span: Span | None = None
        response_start_time: float | None = None
        time_to_first_audio_ms: int | None = None
        model_error: Exception | None = None

        try:
            async for event in self._agent.model.receive():
                await self._event_queue.put(event)

                if isinstance(event, BidiResponseStartEvent):
                    if response_span:
                        _telemetry.end_response_span(
                            self._tracer,
                            response_span,
                            stop_reason="interrupted",
                            time_to_first_audio_ms=time_to_first_audio_ms,
                        )
                    response_span = _telemetry.start_response_span(
                        self._tracer, event.response_id, parent_span=self._session_span
                    )
                    response_start_time = time.perf_counter()
                    time_to_first_audio_ms = None

                elif isinstance(event, BidiAudioStreamEvent):
                    if response_start_time is not None and time_to_first_audio_ms is None:
                        time_to_first_audio_ms = int((time.perf_counter() - response_start_time) * 1000)

                elif isinstance(event, BidiResponseCompleteEvent):
                    if response_span:
                        _telemetry.end_response_span(
                            self._tracer,
                            response_span,
                            stop_reason=event.stop_reason,
                            time_to_first_audio_ms=time_to_first_audio_ms,
                        )
                        response_span = None

                elif isinstance(event, BidiTranscriptStreamEvent):
                    if event["is_final"]:
                        message: Message = {"role": event["role"], "content": [{"text": event["text"]}]}
                        await self._agent._append_messages(message)

                elif isinstance(event, ToolUseStreamEvent):
                    tool_use = event["current_tool_use"]
                    self._task_pool.create(self._run_tool(tool_use))

                elif isinstance(event, BidiInterruptionEvent):
                    if self._session_span:
                        _telemetry.add_interruption_event(self._session_span, event["reason"])

                    await self._agent.hooks.invoke_callbacks_async(
                        BidiInterruptionHookEvent(
                            agent=self._agent,
                            reason=event["reason"],
                            interrupted_response_id=event.get("interrupted_response_id"),
                        )
                    )

                elif isinstance(event, BidiUsageEvent):
                    self._accumulated_input_tokens += event.input_tokens
                    self._accumulated_output_tokens += event.output_tokens
                    self._accumulated_total_tokens += event.total_tokens
                    self._accumulated_cache_read_tokens += event.cache_read_input_tokens or 0

        except Exception as error:
            model_error = error
            await self._event_queue.put(error)
        finally:
            if response_span:
                stop_reason = "error" if model_error else "incomplete"
                _telemetry.end_response_span(
                    self._tracer,
                    response_span,
                    stop_reason=stop_reason,
                    time_to_first_audio_ms=time_to_first_audio_ms,
                    error=model_error,
                )
                response_span = None

    async def _run_tool(self, tool_use: ToolUse) -> None:
        """Task for running tool requested by the model using the tool executor.

        Args:
            tool_use: Tool use request from model.
        """
        logger.debug("tool_name=<%s> | tool execution starting", tool_use["name"])

        tool_results: list[ToolResult] = []

        # Ensure request_state exists for tools like strands_tools.stop
        if "request_state" not in self._invocation_state:
            self._invocation_state["request_state"] = {}

        invocation_state: dict[str, Any] = {
            **self._invocation_state,
            "agent": self._agent,
            "model": self._agent.model,
            "messages": self._agent.messages,
            "system_prompt": self._agent.system_prompt,
        }

        tool_call_span = self._tracer.start_tool_call_span(tool_use, parent_span=self._session_span)
        tool_result: ToolResult | None = None
        tool_error: Exception | None = None

        try:
            tool_events = self._agent.tool_executor._stream(
                self._agent,
                tool_use,
                tool_results,
                invocation_state,
                structured_output_context=None,
            )

            async for tool_event in tool_events:
                if isinstance(tool_event, ToolInterruptEvent):
                    self._agent._interrupt_state.deactivate()
                    interrupt_names = [interrupt.name for interrupt in tool_event.interrupts]
                    raise RuntimeError(f"interrupts={interrupt_names} | tool interrupts are not supported in bidi")

                await self._event_queue.put(tool_event)

            # Normal flow for all tools (including stop_conversation)
            tool_result_event = cast(ToolResultEvent, tool_event)
            tool_result = tool_result_event.tool_result

            tool_use_message: Message = {"role": "assistant", "content": [{"toolUse": tool_use}]}
            tool_result_message: Message = {"role": "user", "content": [{"toolResult": tool_result_event.tool_result}]}
            await self._agent._append_messages(tool_use_message, tool_result_message)

            await self._event_queue.put(ToolResultMessageEvent(tool_result_message))

            # Check for stop_event_loop flag (set by strands_tools.stop, stop_conversation, or any custom tool)
            request_state = invocation_state.get("request_state", {})
            should_stop = request_state.get("stop_event_loop", False)

            # Backward compatibility: also check for stop_conversation by name (deprecated)
            if not should_stop and tool_use["name"] == "stop_conversation":
                warnings.warn(
                    "Stopping the event loop by tool name 'stop_conversation' is deprecated. "
                    "Use request_state['stop_event_loop'] = True instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                should_stop = True

            if should_stop:
                logger.info("stop_event_loop=<True> | stopping conversation")
                connection_id = getattr(self._agent.model, "_connection_id", "unknown")
                await self._event_queue.put(
                    BidiConnectionCloseEvent(connection_id=connection_id, reason="user_request")
                )
                return  # Skip sending result to model

            # Send result to model
            await self.send(tool_result_event)

        except Exception as error:
            tool_error = error
            await self._event_queue.put(error)
        finally:
            # Single end site ensures the span is closed even on cancellation.
            self._tracer.end_tool_call_span(tool_call_span, tool_result=tool_result, error=tool_error)
