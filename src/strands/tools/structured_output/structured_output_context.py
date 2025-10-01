"""Per-invocation context for structured output management.

This module provides a typed, thread-safe context for managing structured output
state during agent invocations, replacing the use of invocation_state dictionary.
"""

import logging
from typing import Dict, Optional, Set, TYPE_CHECKING

from pydantic import BaseModel

from ...types.tools import ToolChoice, ToolUse

if TYPE_CHECKING:
    from ...output.base import OutputSchema

logger = logging.getLogger(__name__)


class StructuredOutputContext:
    """Per-invocation context for structured output execution.

    This class manages all structured output state for a single agent invocation,
    providing thread-safe isolation for concurrent executions. Each invocation
    creates its own context instance, ensuring no shared state between concurrent
    calls.

    This class combines both state management and processing logic for structured
    output, including:
    - Tracking expected tool names from output schemas
    - Managing validated result storage
    - Extracting structured output results from tool executions
    - Managing retry attempts for structured output forcing

    Attributes:
        attempts: Number of structured output forcing attempts (max 3).
        results: Mapping of tool_use_id to validated Pydantic objects.
        output_schema: The output schema for this invocation.
        expected_tool_names: Set of tool names expected for structured output.
        forced_mode: Whether this is a forced structured output attempt.
        tool_choice: Tool choice configuration for forcing.
    """

    def __init__(self, output_schema: Optional["OutputSchema"] = None):
        """Initialize a new structured output context.

        Args:
            output_schema: Optional output schema for this invocation.
                          If provided, will extract expected tool names.
        """
        self.attempts: int = 0
        self.results: Dict[str, BaseModel] = {}
        self.output_schema: Optional["OutputSchema"] = output_schema
        self.forced_mode: bool = False
        self.tool_choice: Optional[ToolChoice] = None
        self.MAX_STRUCTURED_OUTPUT_ATTEMPTS: int = 3

        self.expected_tool_names: Set[str] = set()
        if output_schema:
            self.expected_tool_names = {spec["name"] for spec in output_schema.tool_specs}

    def store_result(self, tool_use_id: str, result: BaseModel) -> None:
        """Store a validated structured output result.

        Args:
            tool_use_id: Unique identifier for the tool use.
            result: Validated Pydantic model instance.
        """
        self.results[tool_use_id] = result

    def get_result(self, tool_use_id: str) -> Optional[BaseModel]:
        """Retrieve a stored structured output result.

        Args:
            tool_use_id: Unique identifier for the tool use.

        Returns:
            The validated Pydantic model instance, or None if not found.
        """
        return self.results.get(tool_use_id)

    def pop_result(self, tool_use_id: str) -> Optional[BaseModel]:
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

    def can_retry(self) -> bool:
        """Check if structured output forcing should be retried.

        Returns:
            True if attempts are below the maximum, False otherwise.
        """
        return self.attempts <= self.MAX_STRUCTURED_OUTPUT_ATTEMPTS

    def set_forced_mode(self, tool_choice: Optional[Dict] = None) -> None:
        """Mark this context as being in forced structured output mode.

        Args:
            tool_choice: Optional tool choice configuration.
        """
        self.forced_mode = True
        self.tool_choice = tool_choice or {"any": {}}

    def has_structured_output_tools(self, tool_uses: list[ToolUse]) -> bool:
        """Check if any tool uses are for structured output tools.

        Args:
            tool_uses: List of tool use dictionaries to check.

        Returns:
            True if any tool use matches expected structured output tool names,
            False if no structured output tools are present or expected.
        """
        if not self.expected_tool_names:
            return False
        return any(tool_use.get("name") in self.expected_tool_names for tool_use in tool_uses)

    def extract_result(self, tool_uses: list[ToolUse]) -> Optional[BaseModel]:
        """Extract and remove structured output result from stored results.

        This method searches through the provided tool_uses for structured output tools,
        then extracts their validated results. The result is removed from storage
        during extraction to prevent memory leaks.

        Args:
            tool_uses: List of tool use dictionaries from the current execution cycle.

        Returns:
            The first structured output result found, or None if no results available.
            Results are returned as validated Pydantic BaseModel instances.
        """
        if not self.has_structured_output_tools(tool_uses):
            return None

        for tool_use in tool_uses:
            if tool_use.get("name") in self.expected_tool_names:
                tool_use_id = str(tool_use.get("toolUseId", ""))
                result = self.pop_result(tool_use_id)
                if result is not None:
                    logger.debug(f"Extracted structured output for {tool_use.get('name')}")
                    return result
        return None

    def cleanup(self) -> None:
        """Clean up all stored results.

        This method is provided for explicit cleanup if needed, though
        per-invocation contexts will be garbage collected automatically.
        """
        self.results.clear()
