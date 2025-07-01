"""This module provides handlers for formatting and displaying events from the agent."""

from collections.abc import Callable
from typing import Any

from .formatter import EnhancedToolFormatter


class PrintingCallbackHandler:
    """Handler for streaming text output and tool invocations to stdout."""

    def __init__(self) -> None:
        """Initialize handler."""
        self.tool_count = 0
        self.active_tools = {}  # Track tool_use_id -> tool_name mapping
        self.previous_tool_use = None
        self.tool_formatter = EnhancedToolFormatter()
        self.active_tools = {}  # Track tool_use_id -> tool_name mapping
        self.formatted_tool_names = {}  # Track tool_use_id -> formatted_tool_name mapping

    def __call__(self, **kwargs: Any) -> None:
        """Stream text output and tool invocations to stdout.

        Args:
            **kwargs: Callback event data including:
                - reasoningText (Optional[str]): Reasoning text to print if provided.
                - data (str): Text content to stream.
                - complete (bool): Whether this is the final chunk of a response.
                - current_tool_use (dict): Information about the current tool being used.
                - message (dict): Message containing tool results.
        """
        reasoningText = kwargs.get("reasoningText", False)
        data = kwargs.get("data", "")
        complete = kwargs.get("complete", False)
        current_tool_use = kwargs.get("current_tool_use", {})
        message = kwargs.get("message", {})

        if reasoningText:
            print(reasoningText, end="")

        if data:
            print(data, end="" if not complete else "\n")

        # Handle tool call start (simplified - just track for result matching)
        if current_tool_use and current_tool_use.get("name"):
            tool_name = current_tool_use.get("name", "Unknown tool")
            tool_use_id = current_tool_use.get("toolUseId", "")
            
            # Track this tool for later result matching
            if tool_use_id and self.previous_tool_use != current_tool_use:
                self.previous_tool_use = current_tool_use
                self.tool_count += 1
                self.active_tools[tool_use_id] = tool_name

        # Handle tool results
        if message and message.get("role") == "user" and "content" in message:
            content = message["content"]
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "toolResult" in item:
                        tool_result = item["toolResult"]
                        # Extract tool information from the result
                        tool_use_id = tool_result.get("toolUseId", "")
                        status = tool_result.get("status", "unknown")
                        result_content = tool_result.get("content", [])

                        # Get the tool name from our tracking
                        tool_name = self.active_tools.get(tool_use_id, "Unknown Tool")


                        # Display tool completion result
                        success = status == "success"

                        formatted_result = self.tool_formatter.format_tool_completion(
                            tool_name, success, result_content if success else None,
                            error=result_content if not success else None,
                            formatted_tool_name=tool_name
                        )
                        print(f"{formatted_result}")

                        # Clean up tracking
                        if tool_use_id in self.active_tools:
                            del self.active_tools[tool_use_id]

        if complete and data:
            print("\n")


class CompositeCallbackHandler:
    """Class-based callback handler that combines multiple callback handlers.

    This handler allows multiple callback handlers to be invoked for the same events,
    enabling different processing or output formats for the same stream data.
    """

    def __init__(self, *handlers: Callable) -> None:
        """Initialize handler."""
        self.handlers = handlers

    def __call__(self, **kwargs: Any) -> None:
        """Invoke all handlers in the chain."""
        for handler in self.handlers:
            handler(**kwargs)


def null_callback_handler(**_kwargs: Any) -> None:
    """Callback handler that discards all output.

    Args:
        **_kwargs: Event data (ignored).
    """
    return None
