"""Direct tool call support.

This module provides the _DirectToolCall and _ToolCaller classes that enable direct tool invocation through the
agent.tool interface, including synchronous execution and streaming methods.

Example:
    ```
    agent = Agent(tools=[my_tool])
    agent.tool.my_tool()
    ```
"""

import asyncio
import contextvars
import json
import logging
import queue
import random
import weakref
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from .._async import run_async
from ..tools.executors._executor import ToolExecutor
from ..types._events import ToolInterruptEvent
from ..types.content import ContentBlock, Message
from ..types.exceptions import ConcurrencyException
from ..types.tools import ToolResult, ToolUse

if TYPE_CHECKING:
    from ..agent import Agent
    from ..experimental.bidi.agent import BidiAgent

logger = logging.getLogger(__name__)

# Sentinel to signal end of stream
_STREAM_END = object()


class _DirectToolCall:
    """Callable wrapper for a single tool that provides streaming methods.

    This class enables three execution modes for direct tool calls:
    1. Synchronous: ``result = agent.tool.my_tool(x=5)``
    2. Sync streaming: ``for event in agent.tool.my_tool.stream(x=5)``
    3. Async streaming: ``async for event in agent.tool.my_tool.stream_async(x=5)``

    Streaming methods do not acquire the invocation lock, do not record to message
    history, and do not apply conversation management. They are designed for
    observability and real-time progress monitoring.
    """

    def __init__(self, agent: "Agent | BidiAgent", tool_name: str) -> None:
        """Initialize direct tool call.

        Args:
            agent: Agent reference that owns the tools.
            tool_name: Name of the tool to execute.
        """
        self._agent_ref = weakref.ref(agent)
        self._tool_name = tool_name

    @property
    def _agent(self) -> "Agent | BidiAgent":
        """Return the agent, raising ReferenceError if it has been garbage collected."""
        agent = self._agent_ref()
        if agent is None:
            raise ReferenceError("Agent has been garbage collected")
        return agent

    def _prepare_tool_use(self, **kwargs: Any) -> tuple[ToolUse, list[ToolResult], dict[str, Any]]:
        """Prepare tool use request, results list, and invocation state.

        Args:
            **kwargs: Tool parameters.

        Returns:
            Tuple of (tool_use, tool_results, invocation_state).

        Raises:
            AttributeError: If tool doesn't exist.
        """
        normalized_name = self._find_normalized_tool_name(self._tool_name)
        tool_id = f"tooluse_{self._tool_name}_{random.randint(100000000, 999999999)}"
        tool_use: ToolUse = {
            "toolUseId": tool_id,
            "name": normalized_name,
            "input": kwargs.copy(),
        }
        return tool_use, [], kwargs

    def __call__(
        self,
        user_message_override: str | None = None,
        record_direct_tool_call: bool | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Synchronous tool execution (existing behavior - backward compatible).

        This method enables the method-style interface (e.g., ``agent.tool.tool_name(param="value")``).
        It matches underscore-separated names to hyphenated tool names (e.g., 'some_thing' matches 'some-thing').

        Args:
            user_message_override: Optional custom message to record.
            record_direct_tool_call: Whether to record in message history.
            **kwargs: Tool parameters.

        Returns:
            ToolResult from execution.

        Raises:
            AttributeError: If tool doesn't exist.
            RuntimeError: If called during interrupt.
            ConcurrencyException: If invocation lock cannot be acquired.
        """
        if self._agent._interrupt_state.activated:
            raise RuntimeError("cannot directly call tool during interrupt")

        if record_direct_tool_call is not None:
            should_record_direct_tool_call = record_direct_tool_call
        else:
            should_record_direct_tool_call = self._agent.record_direct_tool_call

        should_lock = should_record_direct_tool_call

        from ..agent import Agent  # Locally imported to avoid circular reference

        acquired_lock = (
            should_lock and isinstance(self._agent, Agent) and self._agent._invocation_lock.acquire_lock(blocking=False)
        )
        if should_lock and not acquired_lock:
            raise ConcurrencyException(
                "Direct tool call cannot be made while the agent is in the middle of an invocation. "
                "Set record_direct_tool_call=False to allow direct tool calls during agent invocation."
            )

        try:
            tool_use, tool_results, invocation_state = self._prepare_tool_use(**kwargs)

            async def acall() -> ToolResult:
                async for event in ToolExecutor._stream(self._agent, tool_use, tool_results, invocation_state):
                    if isinstance(event, ToolInterruptEvent):
                        self._agent._interrupt_state.deactivate()
                        raise RuntimeError("cannot raise interrupt in direct tool call")

                tool_result = tool_results[0]

                if should_record_direct_tool_call:
                    await self._record_tool_execution(tool_use, tool_result, user_message_override)

                return tool_result

            tool_result = run_async(acall)

            # TODO: https://github.com/strands-agents/sdk-python/issues/1311
            if isinstance(self._agent, Agent):
                self._agent.conversation_manager.apply_management(self._agent)

            return tool_result

        finally:
            if acquired_lock and isinstance(self._agent, Agent):
                self._agent._invocation_lock.release()

    def stream(self, **kwargs: Any) -> Iterator[Any]:
        """Synchronous streaming of tool execution events.

        Bridges async-to-sync streaming using a background thread and queue, yielding
        events in real-time as they are produced by the tool.

        This method does not acquire the invocation lock, does not record to message
        history, and does not apply conversation management.

        Args:
            **kwargs: Tool parameters.

        Yields:
            Tool execution events in real-time.

        Raises:
            AttributeError: If tool doesn't exist.
            RuntimeError: If called during interrupt.
        """
        # Fast-fail before spinning up a thread; stream_async also checks but this avoids unnecessary overhead
        if self._agent._interrupt_state.activated:
            raise RuntimeError("cannot directly call tool during interrupt")

        event_queue: queue.Queue[Any] = queue.Queue()

        async def _produce() -> None:
            try:
                async for event in self.stream_async(**kwargs):
                    event_queue.put(event)
            except BaseException:
                # Re-raise to propagate via future.result(); the sentinel must still be placed
                # on the queue so the main thread unblocks before checking the future
                raise
            finally:
                event_queue.put(_STREAM_END)

        context = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(context.run, asyncio.run, _produce())

            while True:
                item = event_queue.get()
                if item is _STREAM_END:
                    break
                yield item

            # Propagates any exception from the producer thread
            future.result()

    async def stream_async(self, **kwargs: Any) -> AsyncIterator[Any]:
        """Asynchronous streaming of tool execution events.

        Yields events directly from tool execution without recording to message
        history. Designed for observability and real-time progress monitoring.

        This method does not acquire the invocation lock, does not record to message
        history, and does not apply conversation management. It can be used concurrently
        with agent invocations.

        Args:
            **kwargs: Tool parameters.

        Yields:
            Tool execution events from ToolExecutor._stream().

        Raises:
            AttributeError: If tool doesn't exist.
            RuntimeError: If called during interrupt.
        """
        if self._agent._interrupt_state.activated:
            raise RuntimeError("cannot directly call tool during interrupt")

        tool_use, tool_results, invocation_state = self._prepare_tool_use(**kwargs)

        logger.debug("tool_name=<%s>, streaming=<True> | executing tool stream", tool_use["name"])

        async for event in ToolExecutor._stream(self._agent, tool_use, tool_results, invocation_state):
            if isinstance(event, ToolInterruptEvent):
                self._agent._interrupt_state.deactivate()
                raise RuntimeError("cannot raise interrupt in direct tool call")
            yield event

    def _find_normalized_tool_name(self, name: str) -> str:
        """Lookup the tool represented by name, replacing characters with underscores as necessary."""
        tool_registry = self._agent.tool_registry.registry

        if tool_registry.get(name):
            return name

        # If the desired name contains underscores, it might be a placeholder for characters that can't be
        # represented as python identifiers but are valid as tool names, such as dashes. In that case, find
        # all tools that can be represented with the normalized name
        if "_" in name:
            filtered_tools = [
                tool_name for (tool_name, tool) in tool_registry.items() if tool_name.replace("-", "_") == name
            ]

            # The registry itself defends against similar names, so we can just take the first match
            if filtered_tools:
                return filtered_tools[0]

        raise AttributeError(f"Tool '{name}' not found")

    async def _record_tool_execution(
        self,
        tool: ToolUse,
        tool_result: ToolResult,
        user_message_override: str | None,
    ) -> None:
        """Record a tool execution in the message history.

        Creates a sequence of messages that represent the tool execution:

        1. A user message describing the tool call
        2. An assistant message with the tool use
        3. A user message with the tool result
        4. An assistant message acknowledging the tool call

        Args:
            tool: The tool call information.
            tool_result: The result returned by the tool.
            user_message_override: Optional custom message to include.
        """
        # Filter tool input parameters to only include those defined in tool spec
        filtered_input = self._filter_tool_parameters_for_recording(tool["name"], tool["input"])

        # Create user message describing the tool call
        input_parameters = json.dumps(filtered_input, default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>")

        user_msg_content: list[ContentBlock] = [
            {"text": (f"agent.tool.{tool['name']} direct tool call.\nInput parameters: {input_parameters}\n")}
        ]

        # Add override message if provided
        if user_message_override:
            user_msg_content.insert(0, {"text": f"{user_message_override}\n"})

        # Create filtered tool use for message history
        filtered_tool: ToolUse = {
            "toolUseId": tool["toolUseId"],
            "name": tool["name"],
            "input": filtered_input,
        }

        # Create the message sequence
        user_msg: Message = {
            "role": "user",
            "content": user_msg_content,
        }
        tool_use_msg: Message = {
            "role": "assistant",
            "content": [{"toolUse": filtered_tool}],
        }
        tool_result_msg: Message = {
            "role": "user",
            "content": [{"toolResult": tool_result}],
        }
        assistant_msg: Message = {
            "role": "assistant",
            "content": [{"text": f"agent.tool.{tool['name']} was called."}],
        }

        # Add to message history
        await self._agent._append_messages(user_msg, tool_use_msg, tool_result_msg, assistant_msg)

    def _filter_tool_parameters_for_recording(self, tool_name: str, input_params: dict[str, Any]) -> dict[str, Any]:
        """Filter input parameters to only include those defined in the tool specification.

        Args:
            tool_name: Name of the tool to get specification for
            input_params: Original input parameters

        Returns:
            Filtered parameters containing only those defined in tool spec
        """
        all_tools_config = self._agent.tool_registry.get_all_tools_config()
        tool_spec = all_tools_config.get(tool_name)

        if not tool_spec or "inputSchema" not in tool_spec:
            return input_params.copy()

        properties = tool_spec["inputSchema"]["json"]["properties"]
        return {k: v for k, v in input_params.items() if k in properties}


class _ToolCaller:
    """Call tool as a function."""

    def __init__(self, agent: "Agent | BidiAgent") -> None:
        """Initialize instance.

        Args:
            agent: Agent reference that will accept tool results.
        """
        # WARNING: Do not add any other member variables or methods as this could result in a name conflict with
        #          agent tools and thus break their execution.
        self._agent_ref = weakref.ref(agent)

    @property
    def _agent(self) -> "Agent | BidiAgent":
        """Return the agent, raising ReferenceError if it has been garbage collected."""
        agent = self._agent_ref()
        if agent is None:
            raise ReferenceError("Agent has been garbage collected")
        return agent

    def __getattr__(self, name: str) -> _DirectToolCall:
        """Return direct tool call with streaming methods.

        This method enables the tool calling interface by returning a callable
        object that provides both synchronous execution and streaming methods.

        Args:
            name: Tool name.

        Returns:
            Direct tool call instance.
        """
        return _DirectToolCall(self._agent, name)
