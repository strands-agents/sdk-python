"""ToolCaller base class."""

import random
from typing import Any, Callable, Optional

from .._async import run_async
from ..tools.executors._executor import ToolExecutor
from ..types._events import ToolInterruptEvent
from ..types.tools import ToolResult, ToolUse


class _ToolCaller:
    """Provides common tool calling functionality for Agent classes.

    Can be used by both traditional Agent and BidirectionalAgent classes with
    agent-specific customizations.

    Automatically detects agent type and applies appropriate behavior:
        - Traditional agents: Uses conversation_manager.apply_management()
    """

    def __init__(self, agent: Any) -> None:
        """Initialize base tool caller.

        Args:
            agent: Agent instance that will process tool results.
        """
        # WARNING: Do not add other member variables to avoid conflicts with tool names
        self._agent = agent

    def __getattr__(self, name: str) -> Callable[..., Any]:
        """Enable method-style tool calling interface.

        This method enables the method-style interface (e.g., `agent.tool.tool_name(param="value")`).
        It matches underscore-separated names to hyphenated tool names (e.g., 'some_thing' matches 'some-thing').

        Args:
            name: The name of the attribute (tool) being accessed.

        Returns:
            A function that when called will execute the named tool.

        Raises:
            AttributeError: If no tool with the given name exists or if multiple tools match the given name.
        """

        def caller(
            user_message_override: Optional[str] = None,
            record_direct_tool_call: Optional[bool] = None,
            **kwargs: Any,
        ) -> Any:
            """Call a tool directly by name.

            Args:
                user_message_override: Optional custom message to record instead of default.
                record_direct_tool_call: Whether to record direct tool calls in message history.
                **kwargs: Keyword arguments to pass to the tool.

            Returns:
                The result returned by the tool.

            Raises:
                AttributeError: If the tool doesn't exist.
                RuntimeError: If called during an interrupt or if interrupt is raised.
            """
            # Check if agent has interrupt state and if it's activated
            if hasattr(self._agent, "_interrupt_state") and self._agent._interrupt_state.activated:
                raise RuntimeError("cannot directly call tool during interrupt")

            normalized_name = self._find_normalized_tool_name(name)

            # Create unique tool ID and set up the tool request
            tool_id = f"tooluse_{name}_{random.randint(100000000, 999999999)}"
            tool_use: ToolUse = {
                "toolUseId": tool_id,
                "name": normalized_name,
                "input": kwargs.copy(),
            }

            # Execute tool using shared execution pipeline
            tool_result = self._execute_tool_async(tool_use, kwargs, user_message_override, record_direct_tool_call)

            # Apply conversation management if agent supports it (traditional agents)
            if hasattr(self._agent, "conversation_manager"):
                self._agent.conversation_manager.apply_management(self._agent)

            return tool_result

        return caller

    def _find_normalized_tool_name(self, name: str) -> str:
        """Lookup the tool represented by name, replacing characters with underscores as necessary.

        Args:
            name: Tool name to normalize.

        Returns:
            Normalized tool name that exists in registry.

        Raises:
            AttributeError: If tool not found.
        """
        tool_registry = self._agent.tool_registry.registry

        if tool_registry.get(name, None):
            return name

        # Handle underscore placeholder for characters that can't be python identifiers
        if "_" in name:
            filtered_tools = [
                tool_name for (tool_name, tool) in tool_registry.items() if tool_name.replace("-", "_") == name
            ]

            # Registry defends against similar names, so take first match
            if filtered_tools:
                return filtered_tools[0]  # type: ignore

        raise AttributeError(f"Tool '{name}' not found")

    def _execute_tool_async(
        self,
        tool_use: ToolUse,
        invocation_state: dict[str, Any],
        user_message_override: Optional[str],
        record_direct_tool_call: Optional[bool],
    ) -> ToolResult:
        """Execute tool asynchronously using shared Strands pipeline.

        Args:
            tool_use: Tool execution request.
            invocation_state: Execution context.
            user_message_override: Optional message override.
            record_direct_tool_call: Optional recording override.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If interrupt is raised during tool execution.
        """
        tool_results: list[ToolResult] = []

        async def acall() -> ToolResult:
            async for event in ToolExecutor._stream(self._agent, tool_use, tool_results, invocation_state):
                # Check for interrupt events
                if isinstance(event, ToolInterruptEvent):
                    if hasattr(self._agent, "_interrupt_state"):
                        self._agent._interrupt_state.deactivate()
                    raise RuntimeError("cannot raise interrupt in direct tool call")

            tool_result = tool_results[0]

            # Determine if we should record the tool call
            should_record = (
                record_direct_tool_call if record_direct_tool_call is not None else self._agent.record_direct_tool_call
            )

            if should_record:
                # Use agent's async recording method
                await self._agent._record_tool_execution(tool_use, tool_result, user_message_override)

            return tool_result

        return run_async(acall)
