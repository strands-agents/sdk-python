"""Tool registry for agent tools.

This module provides a central registry for managing agent tools, including discovery,
validation, registration, and invocation capabilities.
"""

import inspect
import logging
import os
import sys
from importlib import util
from os.path import expanduser
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from typing_extensions import cast

from strands.tools.decorator import DecoratedFunctionTool

from ..types.tools import AgentTool, ToolSpec
from .tools import PythonAgentTool, normalize_schema, normalize_tool_spec

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Agent Tool CRUD Interface.

    Provides methods for creating, reading, updating, and deleting tools in the registry.
    """

    def __init__(self, load_tools_from_directory: bool = False) -> None:
        """Initialize the ToolRegistry.

        Args:
            load_tools_from_directory: Whether to load tools from the tools directory.
                If True, tools will be loaded from the standard tool directories.
        """
        self.agent_tools: Dict[str, AgentTool] = {}
        self.tool_directories = self._find_tool_directories()
        self._tool_config: Dict[str, Any] = {"tools": []}
        self._hot_reload_tools: Dict[str, AgentTool] = {}

        if load_tools_from_directory:
            self._load_directory_tools()

    def create_tool(self, tool: Union[str, dict[str, str], Any]) -> str:
        """Create a single new tool in the registry.

        Args:
            tool: A tool to create. Can be:
                - String tool name (e.g., "calculator")
                - File path (e.g., "/path/to/tool.py")
                - Imported Python module (e.g., a module object)
                - Function decorated with @tool
                - Dictionary with name/path keys
                - Instance of an AgentTool

        Returns:
            The name of the created tool.

        Raises:
            ValueError: If tool validation fails or tool already exists.
            FileNotFoundError: If specified tool file doesn't exist.
        """
        if tool is None:
            raise ValueError("Tool cannot be None")

        # Extract the tool name to check if it already exists
        tool_name = self._extract_tool_name(tool)
        if not tool_name:
            raise ValueError("Could not extract tool name from provided tool")

        # Check if the tool already exists before making any changes
        if tool_name in self.agent_tools:
            raise ValueError(f"Tool '{tool_name}' already exists")

        # Process the tool - this will validate and register the tool
        created_tool_name = self._process_tool(tool, validate_unique=True)

        if not created_tool_name:
            raise ValueError(f"Failed to create tool '{tool_name}'")

        # Update tool configuration
        if created_tool_name in self.agent_tools:
            tool_spec = self.agent_tools[created_tool_name].tool_spec
            self._update_tool_config(self._tool_config, tool_spec)

        return created_tool_name

    def read_tool(self, tool_name: str) -> ToolSpec:
        """Read a single tool specification.

        Args:
            tool_name: The name of the tool to retrieve.

        Returns:
            The tool specification as a ToolSpec object.

        Raises:
            ValueError: If the tool doesn't exist or validation fails.
        """
        if tool_name not in self.agent_tools:
            raise ValueError(f"Tool '{tool_name}' not found in registry")

        tool = self.agent_tools[tool_name]
        spec = tool.tool_spec.copy()

        try:
            spec = normalize_tool_spec(spec)
            self._validate_tool_spec(spec)
            return spec
        except ValueError as e:
            logger.warning("tool_name=<%s> | spec validation failed | %s", tool_name, e)
            raise ValueError(f"Tool '{tool_name}' spec validation failed: {str(e)}") from e

    def update_tool(self, tool: Union[str, dict[str, str], Any]) -> str:
        """Update a single existing tool in the registry.

        This method updates an existing tool based on its ID/name. It does not add new tools
        or remove existing ones - only updates the specification and implementation of a tool
        that already exists in the registry.

        Args:
            tool: The tool to update. Can be:
                - String tool name (e.g., "calculator")
                - File path (e.g., "/path/to/tool.py")
                - Imported Python module (e.g., a module object)
                - Function decorated with @tool
                - Dictionary with name/path keys
                - Instance of an AgentTool

        Returns:
            The name of the updated tool.

        Raises:
            ValueError: If tool validation fails or if the tool is not found in the registry.
            FileNotFoundError: If specified tool file doesn't exist.
        """
        if tool is None:
            raise ValueError("Tool cannot be None")

        # Extract the tool name to check if it exists
        tool_name = self._extract_tool_name(tool)
        if not tool_name:
            raise ValueError("Could not extract tool name from provided tool")

        # Check if the tool exists before making any changes
        if tool_name not in self.agent_tools:
            raise ValueError(f"Tool '{tool_name}' not found in registry")

        # Make a backup of the current tool in case the update fails
        original_tool = self.agent_tools.get(tool_name)

        try:
            # Update the tool
            self._update_single_tool(tool, validate_unique=False)
            return tool_name
        except Exception as e:
            # Restore the original tool if the update fails
            if original_tool:
                self.agent_tools[tool_name] = original_tool
            logger.warning("tool_name=<%s> | failed to update tool | %s", tool_name, e)
            raise ValueError(f"Failed to update tool '{tool_name}': {str(e)}") from e

    def delete_tool(self, tool_name: str) -> str:
        """Delete a single tool from the registry.

        This method removes a specified tool from both the main registry and dynamic tools.

        Args:
            tool_name: The name of the tool to delete.

        Returns:
            The name of the deleted tool.

        Raises:
            ValueError: If the specified tool doesn't exist.
        """
        if tool_name is None:
            raise ValueError("Tool name cannot be None")

        # Check if the tool exists before attempting to delete it
        if tool_name not in self.agent_tools:
            raise ValueError(f"Tool '{tool_name}' not found in registry")

        # Make a backup of the tool in case we need to restore it
        original_tool = self.agent_tools.get(tool_name)
        original_hot_reload = self._hot_reload_tools.get(tool_name)

        # Find and remove the tool from the tool config
        original_tool_config_entry = None
        for idx, tool_entry in enumerate(self._tool_config.get("tools", [])):
            if tool_entry.get("toolSpec", {}).get("name") == tool_name:
                original_tool_config_entry = (idx, tool_entry)
                break

        try:
            # Delete the tool from the main registry
            del self.agent_tools[tool_name]
            logger.debug("tool_name=<%s> | deleted from main registry", tool_name)

            # Delete from hot reload tools if present
            if tool_name in self._hot_reload_tools:
                del self._hot_reload_tools[tool_name]
                logger.debug("tool_name=<%s> | deleted from dynamic tools", tool_name)

            # Remove from tool config if present
            if original_tool_config_entry:
                idx, _ = original_tool_config_entry
                self._tool_config["tools"].pop(idx)
                logger.debug("tool_name=<%s> | removed from tool config", tool_name)

            return tool_name
        except Exception as e:
            # Restore the tool if deletion fails
            if original_tool:
                self.agent_tools[tool_name] = original_tool
            if original_hot_reload:
                self._hot_reload_tools[tool_name] = original_hot_reload

            # Restore tool config entry if it was removed
            if original_tool_config_entry:
                idx, entry = original_tool_config_entry
                if idx < len(self._tool_config["tools"]):
                    self._tool_config["tools"].insert(idx, entry)
                else:
                    self._tool_config["tools"].append(entry)

            logger.warning("tool_name=<%s> | failed to delete tool | %s", tool_name, e)
            raise ValueError(f"Failed to delete tool '{tool_name}': {str(e)}") from e

    def list_tools(self) -> Dict[str, ToolSpec]:
        """List all tool specifications.

        Returns:
            A dictionary mapping tool names to their specifications.
        """
        tool_specs = {}
        for tool_name, tool in self.agent_tools.items():
            spec = tool.tool_spec.copy()
            try:
                spec = normalize_tool_spec(spec)
                self._validate_tool_spec(spec)
                tool_specs[tool_name] = spec
            except ValueError as e:
                logger.warning("tool_name=<%s> | spec validation failed | %s", tool_name, e)
        return tool_specs

    def _extract_tool_name(self, tool: Any) -> Optional[str]:
        """Extract tool name from various tool specification formats.

        Args:
            tool: Tool specification in various formats.

        Returns:
            Tool name if extractable, None otherwise.
        """
        if isinstance(tool, str):
            return os.path.basename(tool).split(".")[0]
        elif isinstance(tool, dict) and "name" in tool:
            return str(tool["name"])
        elif isinstance(tool, dict) and "path" in tool:
            return os.path.basename(str(tool["path"])).split(".")[0]
        elif hasattr(tool, "__file__") and inspect.ismodule(tool):
            return tool.__name__.split(".")[-1]
        elif isinstance(tool, AgentTool):
            return tool.tool_name
        return None

    def _update_single_tool(self, tool: Any, validate_unique: bool = False) -> None:
        """Update a single tool specification.

        Args:
            tool: The tool to update in various formats (string, dict, AgentTool).
            validate_unique: Whether to check if the tool already exists before updating.

        Raises:
            ValueError: If tool validation fails or tool already exists (when validate_unique is True).
            FileNotFoundError: If specified tool file doesn't exist.
        """
        if isinstance(tool, str):
            tool_name = os.path.basename(tool).split(".")[0]
            if validate_unique and tool_name in self.agent_tools:
                raise ValueError(f"Tool '{tool_name}' already exists")
            self._load_tool_from_filepath(tool_name=tool_name, tool_path=tool)
        elif isinstance(tool, dict) and "name" in tool and "path" in tool:
            if validate_unique and tool["name"] in self.agent_tools:
                raise ValueError(f"Tool '{tool['name']}' already exists")
            self._load_tool_from_filepath(tool_name=tool["name"], tool_path=tool["path"])
        elif isinstance(tool, dict) and "path" in tool:
            tool_name = os.path.basename(tool["path"]).split(".")[0]
            if validate_unique and tool_name in self.agent_tools:
                raise ValueError(f"Tool '{tool_name}' already exists")
            self._load_tool_from_filepath(tool_name=tool_name, tool_path=tool["path"])
        elif isinstance(tool, AgentTool):
            if validate_unique and tool.tool_name in self.agent_tools:
                raise ValueError(f"Tool '{tool.tool_name}' already exists")
            self._register_tool(tool)

    def _update_tools_from_directory(self) -> List[str]:
        """Update existing tools from directory discovery.

        Returns:
            List of successfully updated tool names.

        Raises:
            ValueError: If any tool fails to update.
        """
        updated_tools = []
        tool_modules = self._discover_tool_modules()
        tools_to_update = []

        # First, identify all tools that need to be updated
        for tool_name, tool_path in tool_modules.items():
            if tool_name in ["__init__"]:
                continue

            # Only include tools that already exist in registry
            if tool_name in self.agent_tools:
                tools_to_update.append((tool_name, tool_path))
            else:
                logger.debug("tool_name=<%s> | tool not in registry, skipping directory update", tool_name)

        # Then update all tools, failing atomically if any update fails
        for tool_name, _ in tools_to_update:
            try:
                # Reload the existing tool
                self._reload_tool(tool_name)
                updated_tools.append(tool_name)
            except Exception as e:
                logger.warning("tool_name=<%s> | failed to update from directory | %s", tool_name, e)
                # Fail atomically - if any tool fails to update, the entire operation fails
                raise ValueError(f"Failed to update tool '{tool_name}' from directory: {str(e)}") from e

        return updated_tools

    def _reload_tool(self, tool_name: str) -> None:
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
            tools_dirs = self.tool_directories
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
                        self._register_tool(function_tool)

                        # Update tool configuration
                        self._update_tool_config(self._tool_config, function_tool.tool_spec)

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
            self._validate_tool_spec(module.TOOL_SPEC)

            new_tool = PythonAgentTool(tool_name, module.TOOL_SPEC, tool_function)

            # Register the tool
            self._register_tool(new_tool)

            # Update tool configuration
            self._update_tool_config(self._tool_config, module.TOOL_SPEC)
            logger.debug("tool_name=<%s> | successfully reloaded tool", tool_name)

        except Exception:
            logger.exception("tool_name=<%s> | failed to reload tool", tool_name)
            raise

    def _find_tool_directories(self) -> List[Path]:
        """Find all tool directory paths.

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

    def _load_tool_from_filepath(self, tool_name: str, tool_path: str) -> None:
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

            loaded_tool = ToolLoader.load_tool(tool_path, tool_name)
            loaded_tool.mark_dynamic()

            # Because we're explicitly registering the tool we don't need an allowlist
            self._register_tool(loaded_tool)
        except Exception as e:
            exception_str = str(e)
            logger.exception("tool_name=<%s> | failed to load tool", tool_name)
            raise ValueError(f"Failed to load tool {tool_name}: {exception_str}") from e

    def _discover_tool_modules(self) -> Dict[str, Path]:
        """Discover available tool modules in all tools directories.

        Returns:
            Dictionary mapping tool names to their full paths.
        """
        tool_modules = {}
        tools_dirs = self.tool_directories

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

    def _register_tool(self, tool: AgentTool) -> None:
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

        if self.agent_tools.get(tool.tool_name) is None:
            normalized_name = tool.tool_name.replace("-", "_")

            matching_tools = [
                tool_name
                for (tool_name, tool) in self.agent_tools.items()
                if tool_name.replace("-", "_") == normalized_name
            ]

            if matching_tools:
                raise ValueError(
                    f"Tool name '{tool.tool_name}' already exists as '{matching_tools[0]}'."
                    " Cannot add a duplicate tool which differs by a '-' or '_'"
                )

        self.agent_tools[tool.tool_name] = tool

        if tool.is_dynamic:
            self._hot_reload_tools[tool.tool_name] = tool

            if not tool.supports_hot_reload:
                logger.debug("tool_name=<%s>, tool_type=<%s> | skipping hot reloading", tool.tool_name, tool.tool_type)
                return

            logger.debug(
                "tool_name=<%s>, tool_registry=<%s>, dynamic_tools=<%s> | tool registered",
                tool.tool_name,
                list(self.agent_tools.keys()),
                list(self._hot_reload_tools.keys()),
            )

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

    def _update_tool_config(self, tool_config: Dict[str, Any], new_tool_spec: ToolSpec) -> None:
        """Update tool configuration with a new tool.

        Args:
            tool_config: The current tool configuration dictionary.
            new_tool_spec: The new tool spec to add/update.

        Raises:
            ValueError: If the new tool spec is invalid.
        """
        # Validate tool spec before updating
        try:
            self._validate_tool_spec(new_tool_spec)
        except ValueError as e:
            raise ValueError(f"Tool specification validation failed: {str(e)}") from e

        new_tool_name = new_tool_spec["name"]
        existing_tool_idx = None

        # Find if tool already exists
        for idx, tool_entry in enumerate(tool_config["tools"]):
            if tool_entry["toolSpec"]["name"] == new_tool_name:
                existing_tool_idx = idx
                break

        # Update existing tool or add new one
        new_tool_entry = {"toolSpec": new_tool_spec}
        if existing_tool_idx is not None:  # This check is necessary as existing_tool_idx could be 0 (valid index)
            tool_config["tools"][existing_tool_idx] = new_tool_entry
            logger.debug("tool_name=<%s> | updated existing tool", new_tool_name)
        else:
            tool_config["tools"].append(new_tool_entry)
            logger.debug("tool_name=<%s> | added new tool", new_tool_name)

    def _validate_tool_spec(self, tool_spec: ToolSpec) -> None:
        """Validate tool specification against required schema.

        Args:
            tool_spec: Tool specification to validate.

        Raises:
            ValueError: If the specification is invalid.
        """
        if tool_spec is None:
            raise ValueError("Tool spec cannot be None")
        if not isinstance(tool_spec, dict):
            raise ValueError("Tool spec must be a dictionary")

        required_fields = ["name", "description"]
        missing_fields = [field for field in required_fields if field not in tool_spec]
        if missing_fields:
            raise ValueError(f"Missing required fields in tool spec: {', '.join(missing_fields)}")

        # Validate field types and values
        name = tool_spec.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool name must be a non-empty string")

        description = tool_spec.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Tool description must be a non-empty string")

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

    def _process_tool(self, tool: Union[str, dict[str, str], Any], validate_unique: bool = False) -> str:
        """Process a single tool specification in various formats.

        This private method handles the actual processing of different tool formats
        and is called by the create_tool() and update_tool() methods.

        Args:
            tool: A tool specification. Can be:
                - String file path (e.g., "/path/to/tool.py")
                - Imported Python module (e.g., a module object)
                - Function decorated with @tool
                - Dictionary with name/path keys (e.g., {"name": "calc", "path": "/path/calc.py"})
                - Instance of an AgentTool
            validate_unique: Whether to validate that the tool doesn't already exist in the registry.
                If True, raises ValueError when a tool with the same name already exists.

        Returns:
            The name of the processed tool.

        Raises:
            ValueError: If tool validation fails or the tool already exists (when validate_unique is True).
            FileNotFoundError: If the specified tool file doesn't exist.
        """
        tool_name = None

        # Case 1: String file path
        if isinstance(tool, str):
            # Extract tool name from path
            tool_name = os.path.basename(tool).split(".")[0]
            if validate_unique and tool_name in self.agent_tools:
                raise ValueError(f"Tool '{tool_name}' already exists")
            if not os.path.exists(tool):
                raise FileNotFoundError(f"Tool file not found: {tool}")
            self._load_tool_from_filepath(tool_name=tool_name, tool_path=str(tool))

        # Case 2: Dictionary with name and path
        elif isinstance(tool, dict) and "name" in tool and "path" in tool:
            tool_name = str(tool["name"])
            if validate_unique and tool_name in self.agent_tools:
                raise ValueError(f"Tool '{tool_name}' already exists")
            if not os.path.exists(tool["path"]):
                raise FileNotFoundError(f"Tool file not found: {tool['path']}")
            self._load_tool_from_filepath(tool_name=tool_name, tool_path=str(tool["path"]))

        # Case 3: Dictionary with path only
        elif isinstance(tool, dict) and "path" in tool:
            tool_name = os.path.basename(tool["path"]).split(".")[0]
            if validate_unique and tool_name in self.agent_tools:
                raise ValueError(f"Tool '{tool_name}' already exists")
            if not os.path.exists(tool["path"]):
                raise FileNotFoundError(f"Tool file not found: {tool['path']}")
            self._load_tool_from_filepath(tool_name=tool_name, tool_path=str(tool["path"]))

        # Case 4: Imported Python module
        elif hasattr(tool, "__file__") and inspect.ismodule(tool):
            # Get the module file path
            module_path = tool.__file__
            # Extract the tool name from the module name
            tool_name = tool.__name__.split(".")[-1]

            # Check for TOOL_SPEC in module to validate it's a Strands tool
            if hasattr(tool, "TOOL_SPEC") and hasattr(tool, tool_name) and module_path:
                if validate_unique and tool_name in self.agent_tools:
                    raise ValueError(f"Tool '{tool_name}' already exists")
                self._load_tool_from_filepath(tool_name=tool_name, tool_path=str(module_path))
            else:
                function_tools = self._scan_module_for_tools(tool)
                if function_tools:
                    # Just use the first function tool found
                    function_tool = function_tools[0]
                    tool_name = function_tool.tool_name
                    if validate_unique and tool_name in self.agent_tools:
                        raise ValueError(f"Tool '{tool_name}' already exists")
                    self._register_tool(function_tool)
                else:
                    logger.warning("tool_name=<%s>, module_path=<%s> | invalid agent tool", tool_name, module_path)
                    raise ValueError(f"Module '{tool_name}' does not contain valid tools")

        # Case 5: AgentTools (which also covers @tool)
        elif isinstance(tool, AgentTool):
            tool_name = tool.tool_name
            if validate_unique and tool_name in self.agent_tools:
                raise ValueError(f"Tool '{tool_name}' already exists")
            self._register_tool(tool)
        else:
            raise ValueError(f"Unrecognized tool specification: {tool}")

        if not tool_name:
            raise ValueError("Could not extract tool name from provided tool")

        return tool_name

    def _load_directory_tools(self) -> List[str]:
        """Load tools from standard tool directories.

        Discovers and loads tools from the standard tool directories.
        Automatically called during initialization if load_tools_from_directory=True.

        Returns:
            List of successfully loaded tool names.
        """
        tool_names = []
        self._tool_config = {"tools": []}
        tool_modules = self._discover_tool_modules()
        successful_loads = 0
        total_tools = len(tool_modules)
        tool_import_errors = {}

        # Process Python tools from directory
        for tool_name, tool_path in tool_modules.items():
            if tool_name in ["__init__"]:
                continue

            try:
                # Try to create the tool from the path
                try:
                    created_tool_name = self.create_tool(str(tool_path))
                    tool_names.append(created_tool_name)
                    successful_loads += 1
                    logger.debug("tool_name=<%s> | successfully loaded tool from directory", created_tool_name)
                except ValueError as e:
                    # If the tool already exists, log and continue
                    if "already exists" in str(e):
                        logger.debug("tool_name=<%s> | tool already exists, skipping", tool_name)
                    else:
                        # For other validation errors, log and continue with other tools
                        logger.warning("tool_name=<%s> | validation error | %s", tool_name, e)
                        tool_import_errors[tool_name] = str(e)
                except FileNotFoundError as e:
                    logger.warning("tool_name=<%s> | file not found | %s", tool_name, e)
                    tool_import_errors[tool_name] = str(e)
            except Exception as e:
                logger.warning("tool_name=<%s> | failed to load tool | %s", tool_name, e)
                tool_import_errors[tool_name] = str(e)

        # Log summary
        logger.debug("tool_count=<%d>, success_count=<%d> | finished loading tools", total_tools, successful_loads)
        if tool_import_errors:
            for error_tool_name, error in tool_import_errors.items():
                logger.debug("tool_name=<%s> | import error | %s", error_tool_name, error)

        return tool_names
