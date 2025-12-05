"""Agent loop.

The agent loop handles the events received from the model and executes tools when given a tool use request.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, cast

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
from .._async import _TaskPool, stop_all
from ..models import BidiModelTimeoutError
from ..types.events import (
    BidiConnectionCloseEvent,
    BidiConnectionRestartEvent,
    BidiInputEvent,
    BidiInterruptionEvent,
    BidiOutputEvent,
    BidiTextInputEvent,
    BidiTranscriptStreamEvent,
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
            Blocks when agent is reseting the model connection after timeout.
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

        await self._agent.model.start(
            system_prompt=self._agent.system_prompt,
            tools=self._agent.tool_registry.get_all_tool_specs(),
            messages=self._agent.messages,
        )

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
            message: Message = {"role": "user", "content": [{"text": event.text}]}
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
        logger.debug("reseting model connection")

        self._send_gate.clear()

        await self._agent.hooks.invoke_callbacks_async(BidiBeforeConnectionRestartEvent(self._agent, timeout_error))

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
            await self._agent.hooks.invoke_callbacks_async(
                BidiAfterConnectionRestartEvent(self._agent, restart_exception)
            )

        self._send_gate.set()

    async def _run_model(self) -> None:
        """Task for running the model.

        Events are streamed through the event queue.
        """
        logger.debug("model task starting")

        try:
            async for event in self._agent.model.receive():
                await self._event_queue.put(event)

                if isinstance(event, BidiTranscriptStreamEvent):
                    if event["is_final"]:
                        message: Message = {"role": event["role"], "content": [{"text": event["text"]}]}
                        await self._agent._append_messages(message)

                elif isinstance(event, ToolUseStreamEvent):
                    tool_use = event["current_tool_use"]
                    self._task_pool.create(self._run_tool(tool_use))

                elif isinstance(event, BidiInterruptionEvent):
                    await self._agent.hooks.invoke_callbacks_async(
                        BidiInterruptionHookEvent(
                            agent=self._agent,
                            reason=event["reason"],
                            interrupted_response_id=event.get("interrupted_response_id"),
                        )
                    )

        except Exception as error:
            await self._event_queue.put(error)

    async def _run_tool(self, tool_use: ToolUse) -> None:
        """Task for running tool requested by the model using the tool executor.

        Args:
            tool_use: Tool use request from model.
        """
        logger.debug("tool_name=<%s> | tool execution starting", tool_use["name"])

        tool_results: list[ToolResult] = []

        invocation_state: dict[str, Any] = {
            **self._invocation_state,
            "agent": self._agent,
            "model": self._agent.model,
            "messages": self._agent.messages,
            "system_prompt": self._agent.system_prompt,
        }

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

            tool_use_message: Message = {"role": "assistant", "content": [{"toolUse": tool_use}]}
            tool_result_message: Message = {"role": "user", "content": [{"toolResult": tool_result_event.tool_result}]}
            await self._agent._append_messages(tool_use_message, tool_result_message)

            await self._event_queue.put(ToolResultMessageEvent(tool_result_message))

            # Check for stop_conversation before sending to model
            if tool_use["name"] == "stop_conversation":
                logger.info("tool_name=<%s> | conversation stop requested, skipping model send", tool_use["name"])
                connection_id = getattr(self._agent.model, "_connection_id", "unknown")
                await self._event_queue.put(
                    BidiConnectionCloseEvent(connection_id=connection_id, reason="user_request")
                )
                return  # Skip the model send

            # Send result to model (all tools except stop_conversation)
            await self.send(tool_result_event)

        except Exception as error:
            await self._event_queue.put(error)
