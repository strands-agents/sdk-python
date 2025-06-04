"""Core tool implementations.

This module provides the base classes for all tool implementations in the SDK, including function-based tools and
Python module-based tools, as well as utilities for validating tool uses and normalizing tool schemas.
"""

import inspect
import logging
import re
from typing import Any, Callable, Dict, Optional, cast

from typing_extensions import Unpack

from ..types.tools import AgentTool, ToolResult, ToolSpec, ToolUse

logger = logging.getLogger(__name__)


class InvalidToolUseNameException(Exception):
    """Exception raised when a tool use has an invalid name."""

    pass


def validate_tool_use(tool: ToolUse) -> None:
    """Validate a tool use request.

    Args:
        tool: The tool use to validate.
    """
    validate_tool_use_name(tool)


def validate_tool_use_name(tool: ToolUse) -> None:
    """Validate the name of a tool use.

    Args:
        tool: The tool use to validate.

    Raises:
        InvalidToolUseNameException: If the tool name is invalid.
    """
    # We need to fix some typing here, because we don't actually expect a ToolUse, but dict[str, Any]
    if "name" not in tool:
        message = "tool name missing"  # type: ignore[unreachable]
        logger.warning(message)
        raise InvalidToolUseNameException(message)

    tool_name = tool["name"]
    tool_name_pattern = r"^[a-zA-Z][a-zA-Z0-9_\-]*$"
    tool_name_max_length = 64
    valid_name_pattern = bool(re.match(tool_name_pattern, tool_name))
    tool_name_len = len(tool_name)

    if not valid_name_pattern:
        message = f"tool_name=<{tool_name}> | invalid tool name pattern"
        logger.warning(message)
        raise InvalidToolUseNameException(message)

    if tool_name_len > tool_name_max_length:
        message = f"tool_name=<{tool_name}>, tool_name_max_length=<{tool_name_max_length}> | invalid tool name length"
        logger.warning(message)
        raise InvalidToolUseNameException(message)


def normalize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a JSON schema to match expectations.

    Args:
        schema: The schema to normalize.

    Returns:
        The normalized schema.
    """
    normalized = {"type": schema.get("type", "object"), "properties": {}}

    # Handle properties
    if "properties" in schema:
        for prop_name, prop_def in schema["properties"].items():
            if isinstance(prop_def, dict):
                normalized_prop = {
                    "type": prop_def.get("type", "string"),
                    "description": prop_def.get("description", f"Property {prop_name}"),
                }

                # Handle enum values correctly
                if "enum" in prop_def:
                    normalized_prop["enum"] = prop_def["enum"]

                # Handle numeric constraints
                if prop_def.get("type") in ["number", "integer"]:
                    if "minimum" in prop_def:
                        normalized_prop["minimum"] = prop_def["minimum"]
                    if "maximum" in prop_def:
                        normalized_prop["maximum"] = prop_def["maximum"]

                normalized["properties"][prop_name] = normalized_prop
            else:
                # Handle non-dict property definitions (like simple strings)
                normalized["properties"][prop_name] = {
                    "type": "string",
                    "description": f"Property {prop_name}",
                }

    # Required fields
    if "required" in schema:
        normalized["required"] = schema["required"]
    else:
        normalized["required"] = []

    return normalized


def normalize_tool_spec(tool_spec: ToolSpec) -> ToolSpec:
    """Normalize a complete tool specification by transforming its inputSchema.

    Args:
        tool_spec: The tool specification to normalize.

    Returns:
        The normalized tool specification.
    """
    normalized = tool_spec.copy()

    # Handle inputSchema
    if "inputSchema" in normalized:
        if isinstance(normalized["inputSchema"], dict):
            if "json" in normalized["inputSchema"]:
                # Schema is already in correct format, just normalize inner schema
                normalized["inputSchema"]["json"] = normalize_schema(normalized["inputSchema"]["json"])
            else:
                # Convert direct schema to proper format
                normalized["inputSchema"] = {"json": normalize_schema(normalized["inputSchema"])}

    return normalized


class FunctionTool(AgentTool):
    """Tool implementation for function-based tools created with @tool.

    This class adapts Python functions decorated with @tool to the AgentTool interface.
    """

    def __init__(self, func: Callable[[ToolUse, Unpack[Any]], ToolResult], tool_name: Optional[str] = None) -> None:
        """Initialize a function-based tool.

        Args:
            func: The decorated function.
            tool_name: Optional tool name (defaults to function name).

        Raises:
            ValueError: If func is not decorated with @tool.
        """
        super().__init__()

        self._func = func

        # Get TOOL_SPEC from the decorated function
        if hasattr(func, "TOOL_SPEC") and isinstance(func.TOOL_SPEC, dict):
            self._tool_spec = cast(ToolSpec, func.TOOL_SPEC)
            # Use name from tool spec if available, otherwise use function name or passed tool_name
            name = self._tool_spec.get("name", tool_name or func.__name__)
            if isinstance(name, str):
                self._name = name
            else:
                raise ValueError(f"Tool name must be a string, got {type(name)}")
        else:
            raise ValueError(f"Function {func.__name__} is not decorated with @tool")

    @property
    def tool_name(self) -> str:
        """Get the name of the tool.

        Returns:
            The name of the tool.
        """
        return self._name

    @property
    def tool_spec(self) -> ToolSpec:
        """Get the tool specification for this function-based tool.

        Returns:
            The tool specification.
        """
        return self._tool_spec

    @property
    def tool_type(self) -> str:
        """Get the type of the tool.

        Returns:
            The string "function" indicating this is a function-based tool.
        """
        return "function"

    @property
    def supports_hot_reload(self) -> bool:
        """Check if this tool supports automatic reloading when modified.

        Returns:
            Always true for function-based tools.
        """
        return True

    def invoke(self, tool: ToolUse, *args: Any, **kwargs: Any) -> ToolResult:
        """Execute the function with the given tool use request.

        Args:
            tool: The tool use request containing the tool name, ID, and input parameters.
            *args: Additional positional arguments to pass to the function.
            **kwargs: Additional keyword arguments to pass to the function.

        Returns:
            A ToolResult containing the status and content from the function execution.
        """
        # Make sure to pass through all kwargs, including 'agent' if provided
        try:
            # Check if the function accepts agent as a keyword argument
            sig = inspect.signature(self._func)
            if "agent" in sig.parameters:
                # Pass agent if function accepts it
                return self._func(tool, **kwargs)
            else:
                # Skip passing agent if function doesn't accept it
                filtered_kwargs = {k: v for k, v in kwargs.items() if k != "agent"}
                return self._func(tool, **filtered_kwargs)
        except Exception as e:
            return {
                "toolUseId": tool.get("toolUseId", "unknown"),
                "status": "error",
                "content": [{"text": f"Error executing function: {str(e)}"}],
            }

    @property
    def original_function(self) -> Callable:
        """Get the original function (without wrapper).

        Returns:
            Undecorated function.
        """
        if hasattr(self._func, "original_function"):
            return cast(Callable, self._func.original_function)
        return self._func

    def get_display_properties(self) -> dict[str, str]:
        """Get properties to display in UI representations.

        Returns:
            Function properties (e.g., function name).
        """
        properties = super().get_display_properties()
        properties["Function"] = self.original_function.__name__
        return properties


class PythonAgentTool(AgentTool):
    """Tool implementation for Python-based tools.

    This class handles tools implemented as Python functions, providing a simple interface for executing Python code
    as SDK tools.
    """

    _callback: Callable[[ToolUse, Any, dict[str, Any]], ToolResult]
    _tool_name: str
    _tool_spec: ToolSpec

    def __init__(
        self, tool_name: str, tool_spec: ToolSpec, callback: Callable[[ToolUse, Any, dict[str, Any]], ToolResult]
    ) -> None:
        """Initialize a Python-based tool.

        Args:
            tool_name: Unique identifier for the tool.
            tool_spec: Tool specification defining parameters and behavior.
            callback: Python function to execute when the tool is invoked.
        """
        super().__init__()

        self._tool_name = tool_name
        self._tool_spec = tool_spec
        self._callback = callback

    @property
    def tool_name(self) -> str:
        """Get the name of the tool.

        Returns:
            The name of the tool.
        """
        return self._tool_name

    @property
    def tool_spec(self) -> ToolSpec:
        """Get the tool specification for this Python-based tool.

        Returns:
            The tool specification.
        """
        return self._tool_spec

    @property
    def tool_type(self) -> str:
        """Identifies this as a Python-based tool implementation.

        Returns:
            "python".
        """
        return "python"

    def invoke(self, tool: ToolUse, *args: Any, **kwargs: dict[str, Any]) -> ToolResult:
        """Execute the Python function with the given tool use request.

        Args:
            tool: The tool use request.
            *args: Additional positional arguments to pass to the underlying callback function.
            **kwargs: Additional keyword arguments to pass to the underlying callback function.

        Returns:
            A ToolResult containing the status and content from the callback execution.
        """
        return self._callback(tool, *args, **kwargs)
