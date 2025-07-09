"""Tool loading utilities."""

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import cast

from ..types.tools import AgentTool
from .decorator import DecoratedFunctionTool
from .tools import PythonAgentTool

logger = logging.getLogger(__name__)


class ToolLoader:
    """Handles loading of tools from different sources."""

    @staticmethod
    def load_python_tool(tool_path: str, tool_name: str) -> AgentTool:
        """Load a Python tool module.

        Args:
            tool_path: Path to the Python tool file.
            tool_name: Name of the tool.

        Returns:
            Tool instance.

        Raises:
            AttributeError: If required attributes are missing from the tool module.
            ImportError: If there are issues importing the tool module.
            TypeError: If the tool function is not callable.
            ValueError: If function in module is not a valid tool.
            Exception: For other errors during tool loading.
        """
        try:
            # Check if tool_path is in the format "package.module:function"; but keep in mind windows whose file path
            # could have a colon so also ensure that it's not a file
            if not os.path.exists(tool_path) and ":" in tool_path:
                module_path, function_name = tool_path.rsplit(":", 1)
                logger.debug("tool_name=<%s>, module_path=<%s> | importing tool from path", function_name, module_path)

                try:
                    # Import the module
                    module = __import__(module_path, fromlist=["*"])

                    # Get the function
                    if not hasattr(module, function_name):
                        raise AttributeError(f"Module {module_path} has no function named {function_name}")

                    func = getattr(module, function_name)

                    if isinstance(func, DecoratedFunctionTool):
                        logger.debug(
                            "tool_name=<%s>, module_path=<%s> | found function-based tool", function_name, module_path
                        )
                        # mypy has problems converting between DecoratedFunctionTool <-> AgentTool
                        return cast(AgentTool, func)
                    else:
                        raise ValueError(
                            f"Function {function_name} in {module_path} is not a valid tool (missing @tool decorator)"
                        )

                except ImportError as e:
                    raise ImportError(f"Failed to import module {module_path}: {str(e)}") from e

            # Normal file-based tool loading
            abs_path = str(Path(tool_path).resolve())

            logger.debug("tool_path=<%s> | loading python tool from path", abs_path)

            # First load the module to get TOOL_SPEC and check for Lambda deployment
            spec = importlib.util.spec_from_file_location(tool_name, abs_path)
            if not spec:
                raise ImportError(f"Could not create spec for {tool_name}")
            if not spec.loader:
                raise ImportError(f"No loader available for {tool_name}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[tool_name] = module
            spec.loader.exec_module(module)

            # First, check for function-based tools with @tool decorator
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, DecoratedFunctionTool):
                    logger.debug(
                        "tool_name=<%s>, tool_path=<%s> | found function-based tool in path", attr_name, tool_path
                    )
                    # mypy has problems converting between DecoratedFunctionTool <-> AgentTool
                    return cast(AgentTool, attr)

            # If no function-based tools found, fall back to traditional module-level tool
            tool_spec = getattr(module, "TOOL_SPEC", None)
            if not tool_spec:
                raise AttributeError(
                    f"Tool {tool_name} missing TOOL_SPEC (neither at module level nor as a decorated function)"
                )

            # Standard local tool loading
            tool_func_name = tool_name
            if not hasattr(module, tool_func_name):
                raise AttributeError(f"Tool {tool_name} missing function {tool_func_name}")

            tool_func = getattr(module, tool_func_name)
            if not callable(tool_func):
                raise TypeError(f"Tool {tool_name} function is not callable")

            return PythonAgentTool(tool_name, tool_spec, tool_func)

        except Exception:
            logger.exception("tool_name=<%s>, sys_path=<%s> | failed to load python tool", tool_name, sys.path)
            raise

    @classmethod
    def load_tool(cls, tool_path: str, tool_name: str) -> AgentTool:
        """Load a tool based on its file extension.

        Args:
            tool_path: Path to the tool file.
            tool_name: Name of the tool.

        Returns:
            Tool instance.

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
                return cls.load_python_tool(abs_path, tool_name)
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
