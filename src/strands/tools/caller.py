"""ToolCaller base class."""

import asyncio
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from ..tools.executors._executor import ToolExecutor
from ..types.tools import ToolResult, ToolUse


class _ToolCaller:
    """Provides common tool calling functionality that can be used by both traditional
    Agent and BidirectionalAgent classes with agent-specific customizations.

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
            """
            normalized_name = self._find_normalized_tool_name(name)

            # Create unique tool ID and set up the tool request
            tool_id = f"tooluse_{name}_{random.randint(100000000, 999999999)}"
            tool_use: ToolUse = {
                "toolUseId": tool_id,
                "name": normalized_name,
                "input": kwargs.copy(),
            }

            # Execute tool using shared execution pipeline
            tool_result = self._execute_tool_sync(tool_use, kwargs)

            # Handle tool call recording with agent-specific behavior
            self._handle_tool_call_recording(tool_use, tool_result, user_message_override, record_direct_tool_call)

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
                return filtered_tools[0]

        raise AttributeError(f"Tool '{name}' not found")

    def _execute_tool_sync(self, tool_use: ToolUse, invocation_state: dict[str, Any]) -> ToolResult:
        """Execute tool synchronously using shared Strands pipeline.

        Args:
            tool_use: Tool execution request.
            invocation_state: Execution context.

        Returns:
            Tool execution result.
        """
        tool_results: list[ToolResult] = []

        async def acall() -> ToolResult:
            async for event in ToolExecutor._stream(self._agent, tool_use, tool_results, invocation_state):
                _ = event
            return tool_results[0]

        def tcall() -> ToolResult:
            return asyncio.run(acall())

        with ThreadPoolExecutor() as executor:
            future = executor.submit(tcall)
            return future.result()

    def _handle_tool_call_recording(
        self,
        tool_use: ToolUse,
        tool_result: ToolResult,
        user_message_override: Optional[str],
        record_direct_tool_call: Optional[bool],
    ) -> None:
        """Handle tool call recording with agent-specific behavior.

        Args:
            tool_use: Tool execution information.
            tool_result: Tool result.
            user_message_override: Optional message override.
            record_direct_tool_call: Optional recording override.
        """
        # Determine if we should record the tool call
        should_record = (
            record_direct_tool_call if record_direct_tool_call is not None else self._agent.record_direct_tool_call
        )

        if should_record:
            # Use agent's recording method
            self._agent._record_tool_execution(tool_use, tool_result, user_message_override)

            # Apply conversation management if agent supports it (traditional agents)
            if hasattr(self._agent, "conversation_manager"):
                self._agent.conversation_manager.apply_management(self._agent)
