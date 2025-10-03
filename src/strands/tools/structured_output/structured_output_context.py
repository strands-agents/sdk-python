"""Per-invocation context for structured output management.

This module provides a typed, thread-safe context for managing structured output
state during agent invocations.
"""

import logging
from typing import Dict, Optional, Set, Type

from pydantic import BaseModel

from ...types.tools import ToolChoice, ToolSpec, ToolUse
from .structured_output_tool import StructuredOutputTool

logger = logging.getLogger(__name__)


class StructuredOutputContext:
    """Per-invocation context for structured output execution.

    This class manages all structured output state for a single agent invocation,
    providing thread-safe isolation for concurrent executions. Each invocation
    creates its own context instance, ensuring no shared state between concurrent
    calls.

    This class combines both state management and processing logic for structured
    output, including:
    - Creating and managing structured output tools
    - Managing validated result storage
    - Extracting structured output results from tool executions
    - Managing retry attempts for structured output forcing

    Attributes:
        attempts: Number of structured output forcing attempts (max 3).
        results: Mapping of tool_use_id to validated Pydantic objects.
        structured_output_model: The Pydantic model type for structured output.
        structured_output_tool: The tool instance for this structured output.
        expected_tool_name: Name of the expected structured output tool.
        forced_mode: Whether this is a forced structured output attempt.
        tool_choice: Tool choice configuration for forcing.
        stop_loop: Whether to stop the event loop after extracting structured output.
    """

    def __init__(self, structured_output_model: Optional[Type[BaseModel]] = None):
        """Initialize a new structured output context.

        Args:
            structured_output_model: Optional Pydantic model type for structured output.
        """
        self.MAX_STRUCTURED_OUTPUT_ATTEMPTS: int = 3
        self.attempts: int = 0
        self.results: Dict[str, BaseModel] = {}
        self.structured_output_model: Optional[Type[BaseModel]] = structured_output_model
        self.structured_output_tool: StructuredOutputTool | None = None
        self.forced_mode: bool = False
        self.tool_choice: ToolChoice | None = None
        self.stop_loop: bool = False
        self.expected_tool_name: Optional[str] = None

        if structured_output_model:
            self.structured_output_tool = StructuredOutputTool(structured_output_model)
            self.expected_tool_name = self.structured_output_tool.tool_name

    def store_result(self, tool_use_id: str, result: BaseModel) -> None:
        """Store a validated structured output result.

        Args:
            tool_use_id: Unique identifier for the tool use.
            result: Validated Pydantic model instance.
        """
        self.results[tool_use_id] = result

    def get_result(self, tool_use_id: str) -> BaseModel | None:
        """Retrieve a stored structured output result.

        Args:
            tool_use_id: Unique identifier for the tool use.

        Returns:
            The validated Pydantic model instance, or None if not found.
        """
        return self.results.get(tool_use_id)

    def pop_result(self, tool_use_id: str) -> BaseModel | None:
        """Retrieve and remove a stored structured output result.

        Args:
            tool_use_id: Unique identifier for the tool use.

        Returns:
            The validated Pydantic model instance, or None if not found.
        """
        return self.results.pop(tool_use_id, None)

    def increment_attempts(self) -> int:
        """Increment and return the attempt counter.

        Returns:
            The new attempt count.
        """
        self.attempts += 1
        return self.attempts

    def setup_retry(self) -> None:
        self.increment_attempts()
        self.set_forced_mode()

    def can_retry(self) -> bool:
        """Check if structured output forcing should be retried.

        Returns:
            True if attempts are below the maximum, False otherwise.
        """
        if not self.structured_output_model:
            return False
        return self.attempts <= self.MAX_STRUCTURED_OUTPUT_ATTEMPTS

    def set_forced_mode(self, tool_choice: dict | None = None) -> None:
        """Mark this context as being in forced structured output mode.

        Args:
            tool_choice: Optional tool choice configuration.
        """
        if not self.structured_output_model:
            return
        self.forced_mode = True
        self.tool_choice = tool_choice or {"any": {}}

    def has_structured_output_tool(self, tool_uses: list[ToolUse]) -> bool:
        """Check if any tool uses are for the structured output tool.

        Args:
            tool_uses: List of tool use dictionaries to check.

        Returns:
            True if any tool use matches the expected structured output tool name,
            False if no structured output tool is present or expected.
        """
        if not self.expected_tool_name:
            return False
        return any(tool_use.get("name") == self.expected_tool_name for tool_use in tool_uses)

    def get_tool_spec(self) -> Optional[ToolSpec]:
        """Get the tool specification for structured output.

        Returns:
            Tool specification, or None if no structured output model.
        """
        if self.structured_output_tool:
            return self.structured_output_tool.tool_spec
        return None

    def extract_result(self, tool_uses: list[ToolUse]) -> BaseModel | None:
        """Extract and remove structured output result from stored results.

        This method searches through the provided tool_uses for the structured output tool,
        then extracts its validated result. The result is removed from storage
        during extraction to prevent memory leaks.

        Args:
            tool_uses: List of tool use dictionaries from the current execution cycle.

        Returns:
            The structured output result if found, or None if no result available.
            Results are returned as validated Pydantic BaseModel instances.
        """
        if not self.has_structured_output_tool(tool_uses):
            return None

        for tool_use in tool_uses:
            if tool_use.get("name") == self.expected_tool_name:
                tool_use_id = str(tool_use.get("toolUseId", ""))
                result = self.pop_result(tool_use_id)
                if result is not None:
                    logger.debug(f"Extracted structured output for {tool_use.get('name')}")
                    return result
        return None
