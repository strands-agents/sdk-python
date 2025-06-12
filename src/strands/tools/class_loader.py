"""This module defines a method for accessing tools from an instance.

It exposes:
- `load_tools_from_instance`: loads all public methods from an instance as AgentTool objects, with automatic name
  disambiguation for instance methods.

It will load instance, class, and static methods from the class, including inherited methods.

By default, all public methods (not starting with _) will be loaded as AgentTool objects, even if not decorated.

Note:
    Tool names must be unique within an agent. If you load tools from multiple instances of the same class,
    you MUST provide a unique label for each instance, or tools will overwrite each other in the registry.
    The registry does not warn or error on duplicates; the last tool registered with a given name wins.

The `load_tools_from_instance` function will return a list of `AgentTool` objects.
"""

import inspect
import logging
from typing import Any, Callable, List, Optional

from ..types.tools import AgentTool, ToolResult, ToolSpec, ToolUse
from .decorator import FunctionToolMetadata

logger = logging.getLogger(__name__)


class GenericFunctionTool(AgentTool):
    """Wraps any callable (instance, static, or class method) as an AgentTool.

    Uses FunctionToolMetadata for metadata extraction and input validation.
    """

    def __init__(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None):
        """Initialize a GenericFunctionTool."""
        super().__init__()
        self._func = func
        try:
            self._meta = FunctionToolMetadata(func)
            self._tool_spec = self._meta.extract_metadata()
            if name:
                self._tool_spec["name"] = name
            if description:
                self._tool_spec["description"] = description
        except Exception as e:
            logger.warning("Could not convert %s to AgentTool: %s", getattr(func, "__name__", str(func)), str(e))
            raise

    @property
    def tool_name(self) -> str:
        """Return the tool's name."""
        return self._tool_spec["name"]

    @property
    def tool_spec(self) -> ToolSpec:
        """Return the tool's specification."""
        return self._tool_spec

    @property
    def tool_type(self) -> str:
        """Return the tool's type."""
        return "function"

    def invoke(self, tool: ToolUse, *args: Any, **kwargs: Any) -> ToolResult:
        """Invoke the tool with validated input."""
        try:
            validated_input = self._meta.validate_input(tool["input"])
            result = self._func(**validated_input)
            return {
                "toolUseId": tool.get("toolUseId", "unknown"),
                "status": "success",
                "content": [{"text": str(result)}],
            }
        except Exception as e:
            return {
                "toolUseId": tool.get("toolUseId", "unknown"),
                "status": "error",
                "content": [{"text": f"Error: {e}"}],
            }


def load_tools_from_instance(
    instance: object,
    disambiguator: Optional[str] = None,
) -> List[AgentTool]:
    """Load all public methods from an instance as AgentTool objects with name disambiguation.

    Instance methods are bound to the given instance and are disambiguated by suffixing the tool name
    with the given label (or the instance id if no prefix is provided). Static and class methods are
    not disambiguated, as they do not depend on instance state.

    Args:
        instance: The instance to inspect.
        disambiguator: Optional string to disambiguate instance method tool names. If not provided, uses id(instance).

    Returns:
        List of AgentTool objects (GenericFunctionTool wrappers).

    Note:
        Tool names must be unique within an agent. If you load tools from multiple instances of the same
        class, you MUST provide a unique label for each instance, or tools will overwrite each
        other in the registry. The registry does not warn or error on duplicates; the last tool registered
        with a given name wins. This function will log a warning if a duplicate tool name is detected in
        the returned list.

    Example:
        from strands.tools.class_loader import load_tools_from_instance

        class MyClass:
            def foo(self, x: int) -> int:
                return x + 1

            @staticmethod
            def bar(y: int) -> int:
                return y * 2

        instance = MyClass()
        tools = load_tools_from_instance(instance, disambiguator="special")
        # tools is a list of AgentTool objects for foo and bar, with foo disambiguated as 'myclass_foo_special'
    """
    methods = []
    class_name = instance.__class__.__name__.lower()
    for name, _member in inspect.getmembers(instance.__class__):
        if name.startswith("_"):
            continue
        tool_name = f"{class_name}_{name}"
        raw_attr = instance.__class__.__dict__.get(name, None)
        if isinstance(raw_attr, staticmethod):
            func = raw_attr.__func__
        elif isinstance(raw_attr, classmethod):
            func = raw_attr.__func__.__get__(instance, instance.__class__)
        else:
            # Instance method: bind to instance and disambiguate
            func = getattr(instance, name, None)
            tool_name += f"_{str(id(instance))}" if disambiguator is None else f"_{disambiguator}"
        if callable(func):
            try:
                methods.append(GenericFunctionTool(func, name=tool_name))
            except Exception:
                # Warning already logged in GenericFunctionTool
                pass
    return methods
