"""Tool loading utilities."""

import importlib
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import List, cast

from ..types.tools import AgentTool
from .decorator import DecoratedFunctionTool
from .tools import PythonAgentTool

logger = logging.getLogger(__name__)


class ToolLoader:
    """Handles loading of tools from different sources."""

    @staticmethod
    def load_python_tools(tool_path: str, tool_name: str) -> List[AgentTool]:
        """Load a Python tool module and return all discovered function-based tools as a list.

        This method always returns a list of AgentTool (possibly length 1). It is the
        canonical API for retrieving multiple tools from a single Python file.
        """
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
            sys.modules[tool_name] = module
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
            "Use ToolLoader.load_python_tools(...) which always returns a list of AgentTool.",
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
            "Use ToolLoader.load_tools(...) which always returns a list of AgentTool.",
            DeprecationWarning,
            stacklevel=2,
        )

        tools = ToolLoader.load_tools(tool_path, tool_name)
        if not tools:
            raise RuntimeError(f"No tools found in {tool_path} for {tool_name}")

        return tools[0]

    @classmethod
    def load_tools(cls, tool_path: str, tool_name: str) -> list[AgentTool]:
        """Load tools from a file based on its file extension.

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
