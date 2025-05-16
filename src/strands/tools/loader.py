"""Tool loading utilities."""

import importlib
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..types.tools import AgentTool
from .tools import FunctionTool, PythonAgentTool

logger = logging.getLogger(__name__)


def load_function_tool(func: Any) -> Optional[FunctionTool]:
    """Load a function as a tool if it's decorated with @tool.

    Args:
        func: The function to load.

    Returns:
        FunctionTool if successful, None otherwise.
    """
    if not inspect.isfunction(func):
        return None

    if not hasattr(func, "TOOL_SPEC"):
        return None

    try:
        return FunctionTool(func)
    except Exception as e:
        logger.warning("tool_name=<%s> | failed to load function tool | %s", func.__name__, e)
        return None


def scan_module_for_tools(module: Any) -> List[FunctionTool]:
    """Scan a module for function-based tools.

    Args:
        module: The module to scan.

    Returns:
        List of FunctionTool instances found in the module.
    """
    tools = []

    for name, obj in inspect.getmembers(module):
        # Check if this is a function with TOOL_SPEC attached
        if inspect.isfunction(obj) and hasattr(obj, "TOOL_SPEC"):
            # Create a function tool with correct name
            try:
                tool = FunctionTool(obj)
                tools.append(tool)
            except Exception as e:
                logger.warning("tool_name=<%s> | failed to create function tool | %s", name, e)

    return tools


def scan_directory_for_tools(directory: Path) -> Dict[str, FunctionTool]:
    """Scan a directory for Python modules containing function-based tools.

    Args:
        directory: The directory to scan.

    Returns:
        Dictionary mapping tool names to FunctionTool instances.
    """
    tools: Dict[str, FunctionTool] = {}

    if not directory.exists() or not directory.is_dir():
        return tools

    for file_path in directory.glob("*.py"):
        if file_path.name.startswith("_"):
            continue

        try:
            # Dynamically import the module
            module_name = file_path.stem
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if not spec or not spec.loader:
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find tools in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if hasattr(attr, "TOOL_SPEC") and callable(attr):
                    tool = load_function_tool(attr)
                    if tool:
                        # Use the tool's name from tool_name property (which includes custom names)
                        tools[tool.tool_name] = tool

        except Exception as e:
            logger.warning("tool_path=<%s> | failed to load tools under path | %s", file_path, e)

    return tools


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

                    # Check if the function has a TOOL_SPEC (from @tool decorator)
                    if inspect.isfunction(func) and hasattr(func, "TOOL_SPEC"):
                        logger.debug(
                            "tool_name=<%s>, module_path=<%s> | found function-based tool", function_name, module_path
                        )
                        return FunctionTool(func)
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
                # Check if this is a function with TOOL_SPEC attached (from @tool decorator)
                if inspect.isfunction(attr) and hasattr(attr, "TOOL_SPEC"):
                    logger.debug(
                        "tool_name=<%s>, tool_path=<%s> | found function-based tool in path", attr_name, tool_path
                    )
                    # Return as FunctionTool
                    return FunctionTool(attr)

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

            return PythonAgentTool(tool_name, tool_spec, callback=tool_func)

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
