"""Tool loading utilities."""

import importlib
import logging
import os
import sys
import warnings
from importlib.machinery import ModuleSpec
from pathlib import Path
from posixpath import expanduser
from types import ModuleType
from typing import List, cast

from ..types.tools import AgentTool
from .decorator import DecoratedFunctionTool
from .tools import PythonAgentTool

logger = logging.getLogger(__name__)

_TOOL_MODULE_PREFIX = "_strands_tool_"


def load_tool_from_string(tool_string: str) -> List[AgentTool]:
    """Load tools follows strands supported input string formats.

    This function can load a tool based on a string in the following ways:
    1. Local file path to a module based tool: `./path/to/module/tool.py`
    2. Module import path
      2.1. Path to a module based tool: `strands_tools.file_read`
      2.2. Path to a module with multiple AgentTool instances (@tool decorated): `tests.fixtures.say_tool`
      2.3. Path to a module and a specific function: `tests.fixtures.say_tool:say`
    """
    # Case 1: Local file path to a tool
    # Ex: ./path/to/my_cool_tool.py
    tool_path = expanduser(tool_string)
    if os.path.exists(tool_path):
        return load_tools_from_file_path(tool_path)

    # Case 2: Module import path
    # Ex: test.fixtures.say_tool:say (Load specific @tool decorated function)
    # Ex: strands_tools.file_read (Load all @tool decorated functions, or module tool)
    return load_tools_from_module_path(tool_string)


def load_tools_from_file_path(tool_path: str) -> List[AgentTool]:
    """Load module from specified path, and then load tools from that module.

    This function attempts to load the passed in path as a python module, and if it succeeds,
    then it tries to import strands tool(s) from that module.
    """
    abs_path = str(Path(tool_path).resolve())
    logger.debug("tool_path=<%s> | loading python tool from path", abs_path)

    # Load the module by spec

    # Using this to determine the module name
    # ./path/to/my_cool_tool.py -> my_cool_tool
    module_name = os.path.basename(tool_path).split(".")[0]

    # This function imports a module based on its path, and gives it the provided name

    spec: ModuleSpec = cast(ModuleSpec, importlib.util.spec_from_file_location(module_name, abs_path))
    if not spec:
        raise ImportError(f"Could not create spec for {module_name}")
    if not spec.loader:
        raise ImportError(f"No loader available for {module_name}")

    module = importlib.util.module_from_spec(spec)
    # Load, or re-load, the module
    sys.modules[f"{_TOOL_MODULE_PREFIX}{module_name}"] = module
    # Execute the module to run any top level code
    spec.loader.exec_module(module)

    return load_tools_from_module(module, module_name)


def load_tools_from_module_path(module_tool_path: str) -> list[AgentTool]:
    """Load strands tool from a module path.

    Example module paths:
    my.module.path
    my.module.path:tool_name
    """
    if ":" in module_tool_path:
        module_path, tool_func_name = module_tool_path.split(":")
    else:
        module_path, tool_func_name = (module_tool_path, None)

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise AttributeError(f'Tool string: "{module_tool_path}" is not a valid tool string.') from e

    # If a ':' is present in the string, then its a targeted function in a module
    if tool_func_name:
        if hasattr(module, tool_func_name):
            target_tool = getattr(module, tool_func_name)
            if isinstance(target_tool, DecoratedFunctionTool):
                return [target_tool]

        raise AttributeError(f"Tool {tool_func_name} not found in module {module_path}")

    # Else, try to import all of the @tool decorated tools, or the module based tool
    module_name = module_path.split(".")[-1]
    return load_tools_from_module(module, module_name)


def load_tools_from_module(module: ModuleType, module_name: str) -> list[AgentTool]:
    """Load tools from a module.

    First checks if the passed in module has instances of DecoratedToolFunction classes as atributes to the module.
    If so, then it returns them as a list of tools. If not, then it attempts to load the module as a module based tool.
    """
    logger.debug("tool_name=<%s>, module=<%s> | loading tools from module", module_name, module_name)

    # Try and see if any of the attributes in the module are function-based tools decorated with @tool
    # This means that there may be more than one tool available in this module, so we load them all

    function_tools: List[AgentTool] = []
    # Function tools will appear as attributes in the module
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        # Check if the module attribute is a DecoratedFunctiontool
        if isinstance(attr, DecoratedFunctionTool):
            logger.debug("tool_name=<%s>, module=<%s> | found function-based tool in module", attr_name, module_name)
            function_tools.append(cast(AgentTool, attr))

    if function_tools:
        return function_tools

    # Finally, if no DecoratedFunctionTools are found in the module, fall back
    # to module based tools, and search for TOOL_SPEC + function
    module_tool_name = module_name
    tool_spec = getattr(module, "TOOL_SPEC", None)
    if not tool_spec:
        raise AttributeError(
            f"The module {module_tool_name} is not a valid module for loading tools."
            "This module must contain @tool decorated function(s), or must be a module based tool."
        )

    # If this is a module based tool, the module should have a function with the same name as the module itself
    if not hasattr(module, module_tool_name):
        raise AttributeError(f"Module-based tool {module_tool_name} missing function {module_tool_name}")

    tool_func = getattr(module, module_tool_name)
    if not callable(tool_func):
        raise TypeError(f"Tool {module_tool_name} function is not callable")

    return [PythonAgentTool(module_tool_name, tool_spec, tool_func)]


class ToolLoader:
    """Handles loading of tools from different sources."""

    @staticmethod
    def load_python_tools(tool_path: str, tool_name: str) -> List[AgentTool]:
        """DEPRECATED: Load a Python tool module and return all discovered function-based tools as a list.

        This method always returns a list of AgentTool (possibly length 1). It is the
        canonical API for retrieving multiple tools from a single Python file.
        """
        warnings.warn(
            "ToolLoader.load_python_tool is deprecated and will be removed in Strands SDK 2.0. "
            "Use the `load_tools_from_string` or `load_tools_from_module` methods instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            # Support module:function style (e.g. package.module:function)
            if not os.path.exists(tool_path) and ":" in tool_path:
                module_path, function_name = tool_path.rsplit(":", 1)
                logger.debug("tool_name=<%s>, module_path=<%s> | importing tool from path", function_name, module_path)

                try:
                    module = __import__(module_path, fromlist=["*"])
                except ImportError as e:
                    raise ImportError(f"Failed to import module {module_path}: {str(e)}") from e

                if not hasattr(module, function_name):
                    raise AttributeError(f"Module {module_path} has no function named {function_name}")

                func = getattr(module, function_name)
                if isinstance(func, DecoratedFunctionTool):
                    logger.debug(
                        "tool_name=<%s>, module_path=<%s> | found function-based tool", function_name, module_path
                    )
                    return [cast(AgentTool, func)]
                else:
                    raise ValueError(
                        f"Function {function_name} in {module_path} is not a valid tool (missing @tool decorator)"
                    )

            # Normal file-based tool loading
            abs_path = str(Path(tool_path).resolve())
            logger.debug("tool_path=<%s> | loading python tool from path", abs_path)

            # Load the module by spec
            spec = importlib.util.spec_from_file_location(tool_name, abs_path)
            if not spec:
                raise ImportError(f"Could not create spec for {tool_name}")
            if not spec.loader:
                raise ImportError(f"No loader available for {tool_name}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[f"{_TOOL_MODULE_PREFIX}{tool_name}"] = module
            spec.loader.exec_module(module)

            # Collect function-based tools decorated with @tool
            function_tools: List[AgentTool] = []
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, DecoratedFunctionTool):
                    logger.debug(
                        "tool_name=<%s>, tool_path=<%s> | found function-based tool in path", attr_name, tool_path
                    )
                    function_tools.append(cast(AgentTool, attr))

            if function_tools:
                return function_tools

            # Fall back to module-level TOOL_SPEC + function
            tool_spec = getattr(module, "TOOL_SPEC", None)
            if not tool_spec:
                raise AttributeError(
                    f"Tool {tool_name} missing TOOL_SPEC (neither at module level nor as a decorated function)"
                )

            tool_func_name = tool_name
            if not hasattr(module, tool_func_name):
                raise AttributeError(f"Tool {tool_name} missing function {tool_func_name}")

            tool_func = getattr(module, tool_func_name)
            if not callable(tool_func):
                raise TypeError(f"Tool {tool_name} function is not callable")

            return [PythonAgentTool(tool_name, tool_spec, tool_func)]

        except Exception:
            logger.exception("tool_name=<%s>, sys_path=<%s> | failed to load python tool(s)", tool_name, sys.path)
            raise

    @staticmethod
    def load_python_tool(tool_path: str, tool_name: str) -> AgentTool:
        """DEPRECATED: Load a Python tool module and return a single AgentTool for backwards compatibility.

        Use `load_python_tools` to retrieve all tools defined in a .py file (returns a list).
        This function will emit a `DeprecationWarning` and return the first discovered tool.
        """
        warnings.warn(
            "ToolLoader.load_python_tool is deprecated and will be removed in Strands SDK 2.0. "
            "Use the `load_tools_from_string` or `load_tools_from_module` methods instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        tools = ToolLoader.load_python_tools(tool_path, tool_name)
        if not tools:
            raise RuntimeError(f"No tools found in {tool_path} for {tool_name}")
        return tools[0]

    @classmethod
    def load_tool(cls, tool_path: str, tool_name: str) -> AgentTool:
        """DEPRECATED: Load a single tool based on its file extension for backwards compatibility.

        Use `load_tools` to retrieve all tools defined in a file (returns a list).
        This function will emit a `DeprecationWarning` and return the first discovered tool.
        """
        warnings.warn(
            "ToolLoader.load_tool is deprecated and will be removed in Strands SDK 2.0. "
            "Use the `load_tools_from_string` or `load_tools_from_module` methods instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        tools = ToolLoader.load_tools(tool_path, tool_name)
        if not tools:
            raise RuntimeError(f"No tools found in {tool_path} for {tool_name}")

        return tools[0]

    @classmethod
    def load_tools(cls, tool_path: str, tool_name: str) -> list[AgentTool]:
        """DEPRECATED: Load tools from a file based on its file extension.

        Args:
            tool_path: Path to the tool file.
            tool_name: Name of the tool.

        Returns:
            A single Tool instance.

        Raises:
            FileNotFoundError: If the tool file does not exist.
            ValueError: If the tool file has an unsupported extension.
            Exception: For other errors during tool loading.
        """
        warnings.warn(
            "ToolLoader.load_tools is deprecated and will be removed in Strands SDK 2.0. "
            "Use the `load_tools_from_string` or `load_tools_from_module` methods instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        ext = Path(tool_path).suffix.lower()
        abs_path = str(Path(tool_path).resolve())

        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Tool file not found: {abs_path}")

        try:
            if ext == ".py":
                return cls.load_python_tools(abs_path, tool_name)
            else:
                raise ValueError(f"Unsupported tool file type: {ext}")
        except Exception:
            logger.exception(
                "tool_name=<%s>, tool_path=<%s>, tool_ext=<%s>, cwd=<%s> | failed to load tool",
                tool_name,
                abs_path,
                ext,
                os.getcwd(),
            )
            raise
