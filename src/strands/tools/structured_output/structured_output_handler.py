"""Structured output handler for managing structured output tool execution and state."""

import logging
from typing import Any, Optional, TYPE_CHECKING

from pydantic import BaseModel

from ...types.tools import ToolUse
from .structured_output_tool import BASE_KEY

if TYPE_CHECKING:
    from ...output.base import OutputSchema

logger = logging.getLogger(__name__)


class StructuredOutputHandler:
    """Handles structured output tool execution and state management.

    This class manages the lifecycle of structured output tools, including:
    - Tracking expected tool names from output schemas
    - Managing state storage using namespaced keys ({BASE_KEY}_{tool_use_id})
    - Extracting and cleaning up structured output results

    State Management Approach:
        Each structured output result is stored in the invocation_state using a key pattern:
        "{BASE_KEY}_{tool_use_id}" where tool_use_id is the unique identifier for
        each tool execution. This prevents collisions and allows for targeted cleanup.

        The handler maintains a set of expected_tool_names based on the output_schema
        to quickly identify which tool uses are structured output tools.
    """

    def __init__(self, output_schema: Optional["OutputSchema"]) -> None:
        """Initialize the structured output handler.

        Args:
            output_schema: The output schema containing type and mode information.
                          If None, no structured output tools are expected.
        """
        self.output_schema: Optional["OutputSchema"] = output_schema
        self.expected_tool_names: set[str] = set()

        if output_schema:
            self.expected_tool_names = {spec["name"] for spec in output_schema.mode.get_tool_specs(output_schema.type)}

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

    def extract_result(self, invocation_state: dict[str, Any], tool_uses: list[ToolUse]) -> Optional[BaseModel]:
        """Extract and remove structured output result from invocation state.

        This method searches through the provided tool_uses for structured output tools,
        then extracts their validated results from the invocation_state. The result is
        removed from state during extraction to prevent memory leaks.

        State Key Pattern:
            Results are stored with keys: "{BASE_KEY}_{tool_use_id}"

        Args:
            invocation_state: Dictionary containing tool execution state and results.
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
                key = f"{BASE_KEY}_{tool_use_id}"
                result = invocation_state.pop(key, None)
                if result is not None:
                    logger.debug(f"Extracted structured output for {tool_use.get('name')}")
                    return result
        return None

    def cleanup_all_state(self, invocation_state: dict[str, Any]) -> None:
        """Clean up all structured output state from invocation state.

        This method removes all structured output entries while preserving other
        state variables. Use this for final cleanup or when resetting state.

        Protected Keys:
            "_structured_output_attempts" - Preserved as it tracks retry counts

        Args:
            invocation_state: Dictionary containing tool execution state to clean.
        """
        keys_to_remove = [
            key for key in invocation_state.keys() if key.startswith(BASE_KEY) and key != "_structured_output_attempts"
        ]
        for key in keys_to_remove:
            del invocation_state[key]
        if keys_to_remove:
            logger.debug(f"Cleaned up {len(keys_to_remove)} structured output entries")
