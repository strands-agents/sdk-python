"""Tool registry.

This module provides the central registry for all tools available to the agent, including discovery, validation, and
invocation capabilities.
"""

import inspect
import logging
import os
import sys
from importlib import import_module, util
from os.path import expanduser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from typing_extensions import TypedDict, cast

from strands.tools.decorator import DecoratedFunctionTool

from ..types.tools import AgentTool, ToolSpec
from .tools import PythonAgentTool, normalize_schema, normalize_tool_spec

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for all tools available to the agent.

    This class manages tool registration, validation, discovery, and invocation.
    """

    def __init__(self) -> None:
        """Initialize the tool registry."""
        self.registry: Dict[str, AgentTool] = {}
        self.dynamic_tools: Dict[str, AgentTool] = {}
        self.tool_config: Optional[Dict[str, Any]] = None

    def process_tools(self, tools: List[Any]) -> List[str]:
        """Process tools list that can contain tool names, paths, imported modules, or functions.

        Args:
            tools: List of tool specifications.
                Can be:

                - String tool names (e.g., "calculator")
                - File paths (e.g., "/path/to/tool.py")
                - Imported Python modules (e.g., a module object)
                - Functions decorated with @tool
                - Dictionaries with name/path keys
                - Instance of an AgentTool

        Returns:
            List of tool names that were processed.
        """
        tool_names = []

        def add_tool(tool: Any) -> None:
            # Case 1: String file path
            if isinstance(tool, str):
                # Extract tool name from path
                tool_name = os.path.basename(tool).split(".")[0]
                self.load_tool_from_filepath(tool_name=tool_name, tool_path=tool)
                tool_names.append(tool_name)

            # Case 2: Dictionary with name and path
            elif isinstance(tool, dict) and "name" in tool and "path" in tool:
                self.load_tool_from_filepath(tool_name=tool["name"], tool_path=tool["path"])
                tool_names.append(tool["name"])

            # Case 3: Dictionary with path only
            elif isinstance(tool, dict) and "path" in tool:
                tool_name = os.path.basename(tool["path"]).split(".")[0]
                self.load_tool_from_filepath(tool_name=tool_name, tool_path=tool["path"])
                tool_names.append(tool_name)

            # Case 4: Imported Python module
            elif hasattr(tool, "__file__") and inspect.ismodule(tool):
                # Get the module file path
                module_path = tool.__file__
                # Extract the tool name from the module name
                tool_name = tool.__name__.split(".")[-1]

                # Check for TOOL_SPEC in module to validate it's a Strands tool
                if hasattr(tool, "TOOL_SPEC") and hasattr(tool, tool_name) and module_path:
                    self.load_tool_from_filepath(tool_name=tool_name, tool_path=module_path)
                    tool_names.append(tool_name)
                else:
                    function_tools = self._scan_module_for_tools(tool)
                    for function_tool in function_tools:
                        self.register_tool(function_tool)
                        tool_names.append(function_tool.tool_name)

                    if not function_tools:
                        logger.warning("tool_name=<%s>, module_path=<%s> | invalid agent tool", tool_name, module_path)

            # Case 5: AgentTools (which also covers @tool)
            elif isinstance(tool, AgentTool):
                self.register_tool(tool)
                tool_names.append(tool.tool_name)
            # Case 6: Nested iterable (list, tuple, etc.) - add each sub-tool
            elif isinstance(tool, Iterable) and not isinstance(tool, (str, bytes, bytearray)):
                for t in tool:
                    add_tool(t)
            else:
                logger.warning("tool=<%s> | unrecognized tool specification", tool)

        for a_tool in tools:
            add_tool(a_tool)

        return tool_names

    def load_tool_from_filepath(self, tool_name: str, tool_path: str) -> None:
        """Load a tool from a file path.

        Args:
            tool_name: Name of the tool.
            tool_path: Path to the tool file.

        Raises:
            FileNotFoundError: If the tool file is not found.
            ValueError: If the tool cannot be loaded.
        """
        from .loader import ToolLoader

        try:
            tool_path = expanduser(tool_path)
            if not os.path.exists(tool_path):
                raise FileNotFoundError(f"Tool file not found: {tool_path}")

            loaded_tools = ToolLoader.load_tools(tool_path, tool_name)
            for t in loaded_tools:
                t.mark_dynamic()
                # Because we're explicitly registering the tool we don't need an allowlist
                self.register_tool(t)
        except Exception as e:
            exception_str = str(e)
            logger.exception("tool_name=<%s> | failed to load tool", tool_name)
            raise ValueError(f"Failed to load tool {tool_name}: {exception_str}") from e

    def get_all_tools_config(self) -> Dict[str, Any]:
        """Dynamically generate tool configuration by combining built-in and dynamic tools.

        Returns:
            Dictionary containing all tool configurations.
        """
        tool_config = {}
        logger.debug("getting tool configurations")

        # Add all registered tools
        for tool_name, tool in self.registry.items():
            # Make a deep copy to avoid modifying the original
            spec = tool.tool_spec.copy()
            try:
                # Normalize the schema before validation
                spec = normalize_tool_spec(spec)
                self.validate_tool_spec(spec)
                tool_config[tool_name] = spec
                logger.debug("tool_name=<%s> | loaded tool config", tool_name)
            except ValueError as e:
                logger.warning("tool_name=<%s> | spec validation failed | %s", tool_name, e)

        # Add any dynamic tools
        for tool_name, tool in self.dynamic_tools.items():
            if tool_name not in tool_config:
                # Make a deep copy to avoid modifying the original
                spec = tool.tool_spec.copy()
                try:
                    # Normalize the schema before validation
                    spec = normalize_tool_spec(spec)
                    self.validate_tool_spec(spec)
                    tool_config[tool_name] = spec
                    logger.debug("tool_name=<%s> | loaded dynamic tool config", tool_name)
                except ValueError as e:
                    logger.warning("tool_name=<%s> | dynamic tool spec validation failed | %s", tool_name, e)

        logger.debug("tool_count=<%s> | tools configured", len(tool_config))
        return tool_config

    # mypy has problems converting between DecoratedFunctionTool <-> AgentTool
    def register_tool(self, tool: AgentTool) -> None:
        """Register a tool function with the given name.

        Args:
            tool: The tool to register.
        """
        logger.debug(
            "tool_name=<%s>, tool_type=<%s>, is_dynamic=<%s> | registering tool",
            tool.tool_name,
            tool.tool_type,
            tool.is_dynamic,
        )

        # Check duplicate tool name, throw on duplicate tool names except if hot_reloading is enabled
        if tool.tool_name in self.registry and not tool.supports_hot_reload:
            raise ValueError(
                f"Tool name '{tool.tool_name}' already exists. Cannot register tools with exact same name."
            )

        # Check for normalized name conflicts (- vs _)
        if self.registry.get(tool.tool_name) is None:
            normalized_name = tool.tool_name.replace("-", "_")

            matching_tools = [
                tool_name
                for (tool_name, tool) in self.registry.items()
                if tool_name.replace("-", "_") == normalized_name
            ]

            if matching_tools:
                raise ValueError(
                    f"Tool name '{tool.tool_name}' already exists as '{matching_tools[0]}'."
                    " Cannot add a duplicate tool which differs by a '-' or '_'"
                )

        # Register in main registry
        self.registry[tool.tool_name] = tool

        # Register in dynamic tools if applicable
        if tool.is_dynamic:
            self.dynamic_tools[tool.tool_name] = tool

            if not tool.supports_hot_reload:
                logger.debug("tool_name=<%s>, tool_type=<%s> | skipping hot reloading", tool.tool_name, tool.tool_type)
                return

            logger.debug(
                "tool_name=<%s>, tool_registry=<%s>, dynamic_tools=<%s> | tool registered",
                tool.tool_name,
                list(self.registry.keys()),
                list(self.dynamic_tools.keys()),
            )

    def get_tools_dirs(self) -> List[Path]:
        """Get all tool directory paths.

        Returns:
            A list of Path objects for current working directory's "./tools/".
        """
        # Current working directory's tools directory
        cwd_tools_dir = Path.cwd() / "tools"

        # Return all directories that exist
        tool_dirs = []
        for directory in [cwd_tools_dir]:
            if directory.exists() and directory.is_dir():
                tool_dirs.append(directory)
                logger.debug("tools_dir=<%s> | found tools directory", directory)
            else:
                logger.debug("tools_dir=<%s> | tools directory not found", directory)

        return tool_dirs

    def discover_tool_modules(self) -> Dict[str, Path]:
        """Discover available tool modules in all tools directories.

        Returns:
            Dictionary mapping tool names to their full paths.
        """
        tool_modules = {}
        tools_dirs = self.get_tools_dirs()

        for tools_dir in tools_dirs:
            logger.debug("tools_dir=<%s> | scanning", tools_dir)

            # Find Python tools
            for extension in ["*.py"]:
                for item in tools_dir.glob(extension):
                    if item.is_file() and not item.name.startswith("__"):
                        module_name = item.stem
                        # If tool already exists, newer paths take precedence
                        if module_name in tool_modules:
                            logger.debug("tools_dir=<%s>, module_name=<%s> | tool overridden", tools_dir, module_name)
                        tool_modules[module_name] = item

        logger.debug("tool_modules=<%s> | discovered", list(tool_modules.keys()))
        return tool_modules

    def reload_tool(self, tool_name: str) -> None:
        """Reload a specific tool module.

        Args:
            tool_name: Name of the tool to reload.

        Raises:
            FileNotFoundError: If the tool file cannot be found.
            ImportError: If there are issues importing the tool module.
            ValueError: If the tool specification is invalid or required components are missing.
            Exception: For other errors during tool reloading.
        """
        try:
            # Check for tool file
            logger.debug("tool_name=<%s> | searching directories for tool", tool_name)
            tools_dirs = self.get_tools_dirs()
            tool_path = None

            # Search for the tool file in all tool directories
            for tools_dir in tools_dirs:
                temp_path = tools_dir / f"{tool_name}.py"
                if temp_path.exists():
                    tool_path = temp_path
                    break

            if not tool_path:
                raise FileNotFoundError(f"No tool file found for: {tool_name}")

            logger.debug("tool_name=<%s> | reloading tool", tool_name)

            # Add tool directory to path temporarily
            tool_dir = str(tool_path.parent)
            sys.path.insert(0, tool_dir)
            try:
                # Load the module directly using spec
                spec = util.spec_from_file_location(tool_name, str(tool_path))
                if spec is None:
                    raise ImportError(f"Could not load spec for {tool_name}")

                module = util.module_from_spec(spec)
                sys.modules[tool_name] = module

                if spec.loader is None:
                    raise ImportError(f"Could not load {tool_name}")

                spec.loader.exec_module(module)

            finally:
                # Remove the temporary path
                sys.path.remove(tool_dir)

            # Look for function-based tools first
            try:
                function_tools = self._scan_module_for_tools(module)

                if function_tools:
                    for function_tool in function_tools:
                        # Register the function-based tool
                        self.register_tool(function_tool)

                        # Update tool configuration if available
                        if self.tool_config is not None:
                            self._update_tool_config(self.tool_config, {"spec": function_tool.tool_spec})

                    logger.debug("tool_name=<%s> | successfully reloaded function-based tool from module", tool_name)
                    return
            except ImportError:
                logger.debug("function tool loader not available | falling back to traditional tools")

            # Fall back to traditional module-level tools
            if not hasattr(module, "TOOL_SPEC"):
                raise ValueError(
                    f"Tool {tool_name} is missing TOOL_SPEC (neither at module level nor as a decorated function)"
                )

            expected_func_name = tool_name
            if not hasattr(module, expected_func_name):
                raise ValueError(f"Tool {tool_name} is missing {expected_func_name} function")

            tool_function = getattr(module, expected_func_name)
            if not callable(tool_function):
                raise ValueError(f"Tool {tool_name} function is not callable")

            # Validate tool spec
            self.validate_tool_spec(module.TOOL_SPEC)

            new_tool = PythonAgentTool(tool_name, module.TOOL_SPEC, tool_function)

            # Register the tool
            self.register_tool(new_tool)

            # Update tool configuration if available
            if self.tool_config is not None:
                self._update_tool_config(self.tool_config, {"spec": module.TOOL_SPEC})
            logger.debug("tool_name=<%s> | successfully reloaded tool", tool_name)

        except Exception:
            logger.exception("tool_name=<%s> | failed to reload tool", tool_name)
            raise

    def initialize_tools(self, load_tools_from_directory: bool = False) -> None:
        """Initialize all tools by discovering and loading them dynamically from all tool directories.

        Args:
            load_tools_from_directory: Whether to reload tools if changes are made at runtime.
        """
        self.tool_config = None

        # Then discover and load other tools
        tool_modules = self.discover_tool_modules()
        successful_loads = 0
        total_tools = len(tool_modules)
        tool_import_errors = {}

        # Process Python tools
        for tool_name, tool_path in tool_modules.items():
            if tool_name in ["__init__"]:
                continue

            if not load_tools_from_directory:
                continue

            try:
                # Add directory to path temporarily
                tool_dir = str(tool_path.parent)
                sys.path.insert(0, tool_dir)
                try:
                    module = import_module(tool_name)
                finally:
                    if tool_dir in sys.path:
                        sys.path.remove(tool_dir)

                # Process Python tool
                if tool_path.suffix == ".py":
                    # Check for decorated function tools first
                    try:
                        function_tools = self._scan_module_for_tools(module)

                        if function_tools:
                            for function_tool in function_tools:
                                self.register_tool(function_tool)
                                successful_loads += 1
                        else:
                            # Fall back to traditional tools
                            # Check for expected tool function
                            expected_func_name = tool_name
                            if hasattr(module, expected_func_name):
                                tool_function = getattr(module, expected_func_name)
                                if not callable(tool_function):
                                    logger.warning(
                                        "tool_name=<%s> | tool function exists but is not callable", tool_name
                                    )
                                    continue

                                # Validate tool spec before registering
                                if not hasattr(module, "TOOL_SPEC"):
                                    logger.warning("tool_name=<%s> | tool is missing TOOL_SPEC | skipping", tool_name)
                                    continue

                                try:
                                    self.validate_tool_spec(module.TOOL_SPEC)
                                except ValueError as e:
                                    logger.warning("tool_name=<%s> | tool spec validation failed | %s", tool_name, e)
                                    continue

                                tool_spec = module.TOOL_SPEC
                                tool = PythonAgentTool(tool_name, tool_spec, tool_function)
                                self.register_tool(tool)
                                successful_loads += 1

                            else:
                                logger.warning("tool_name=<%s> | tool function missing", tool_name)
                    except ImportError:
                        # Function tool loader not available, fall back to traditional tools
                        # Check for expected tool function
                        expected_func_name = tool_name
                        if hasattr(module, expected_func_name):
                            tool_function = getattr(module, expected_func_name)
                            if not callable(tool_function):
                                logger.warning("tool_name=<%s> | tool function exists but is not callable", tool_name)
                                continue

                            # Validate tool spec before registering
                            if not hasattr(module, "TOOL_SPEC"):
                                logger.warning("tool_name=<%s> | tool is missing TOOL_SPEC | skipping", tool_name)
                                continue

                            try:
                                self.validate_tool_spec(module.TOOL_SPEC)
                            except ValueError as e:
                                logger.warning("tool_name=<%s> | tool spec validation failed | %s", tool_name, e)
                                continue

                            tool_spec = module.TOOL_SPEC
                            tool = PythonAgentTool(tool_name, tool_spec, tool_function)
                            self.register_tool(tool)
                            successful_loads += 1

                        else:
                            logger.warning("tool_name=<%s> | tool function missing", tool_name)

            except Exception as e:
                logger.warning("tool_name=<%s> | failed to load tool | %s", tool_name, e)
                tool_import_errors[tool_name] = str(e)

        # Log summary
        logger.debug("tool_count=<%d>, success_count=<%d> | finished loading tools", total_tools, successful_loads)
        if tool_import_errors:
            for tool_name, error in tool_import_errors.items():
                logger.debug("tool_name=<%s> | import error | %s", tool_name, error)

    def get_all_tool_specs(self) -> list[ToolSpec]:
        """Get all the tool specs for all tools in this registry..

        Returns:
            A list of ToolSpecs.
        """
        all_tools = self.get_all_tools_config()
        tools: List[ToolSpec] = [tool_spec for tool_spec in all_tools.values()]
        return tools

    def validate_tool_spec(self, tool_spec: ToolSpec) -> None:
        """Validate tool specification against required schema.

        Args:
            tool_spec: Tool specification to validate.

        Raises:
            ValueError: If the specification is invalid.
        """
        required_fields = ["name", "description"]
        missing_fields = [field for field in required_fields if field not in tool_spec]
        if missing_fields:
            raise ValueError(f"Missing required fields in tool spec: {', '.join(missing_fields)}")

        if "json" not in tool_spec["inputSchema"]:
            # Convert direct schema to proper format
            json_schema = normalize_schema(tool_spec["inputSchema"])
            tool_spec["inputSchema"] = {"json": json_schema}
            return

        # Validate json schema fields
        json_schema = tool_spec["inputSchema"]["json"]

        # Ensure schema has required fields
        if "type" not in json_schema:
            json_schema["type"] = "object"
        if "properties" not in json_schema:
            json_schema["properties"] = {}
        if "required" not in json_schema:
            json_schema["required"] = []

        # Validate property definitions
        for prop_name, prop_def in json_schema.get("properties", {}).items():
            if not isinstance(prop_def, dict):
                json_schema["properties"][prop_name] = {
                    "type": "string",
                    "description": f"Property {prop_name}",
                }
                continue

            # It is expected that type and description are already included in referenced $def.
            if "$ref" in prop_def:
                continue

            if "type" not in prop_def:
                prop_def["type"] = "string"
            if "description" not in prop_def:
                prop_def["description"] = f"Property {prop_name}"

    class NewToolDict(TypedDict):
        """Dictionary type for adding or updating a tool in the configuration.

        Attributes:
            spec: The tool specification that defines the tool's interface and behavior.
        """

        spec: ToolSpec

    def _update_tool_config(self, tool_config: Dict[str, Any], new_tool: NewToolDict) -> None:
        """Update tool configuration with a new tool.

        Args:
            tool_config: The current tool configuration dictionary.
            new_tool: The new tool to add/update.

        Raises:
            ValueError: If the new tool spec is invalid.
        """
        if not new_tool.get("spec"):
            raise ValueError("Invalid tool format - missing spec")

        # Validate tool spec before updating
        try:
            self.validate_tool_spec(new_tool["spec"])
        except ValueError as e:
            raise ValueError(f"Tool specification validation failed: {str(e)}") from e

        new_tool_name = new_tool["spec"]["name"]
        existing_tool_idx = None

        # Find if tool already exists
        for idx, tool_entry in enumerate(tool_config["tools"]):
            if tool_entry["toolSpec"]["name"] == new_tool_name:
                existing_tool_idx = idx
                break

        # Update existing tool or add new one
        new_tool_entry = {"toolSpec": new_tool["spec"]}
        if existing_tool_idx is not None:
            tool_config["tools"][existing_tool_idx] = new_tool_entry
            logger.debug("tool_name=<%s> | updated existing tool", new_tool_name)
        else:
            tool_config["tools"].append(new_tool_entry)
            logger.debug("tool_name=<%s> | added new tool", new_tool_name)

    def _scan_module_for_tools(self, module: Any) -> List[AgentTool]:
        """Scan a module for function-based tools.

        Args:
            module: The module to scan.

        Returns:
            List of FunctionTool instances found in the module.
        """
        tools: List[AgentTool] = []

        for name, obj in inspect.getmembers(module):
            if isinstance(obj, DecoratedFunctionTool):
                # Create a function tool with correct name
                try:
                    # Cast as AgentTool for mypy
                    tools.append(cast(AgentTool, obj))
                except Exception as e:
                    logger.warning("tool_name=<%s> | failed to create function tool | %s", name, e)

        return tools
