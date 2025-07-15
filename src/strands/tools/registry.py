"""Central registry for managing agent tools with CRUD operations.

This module provides a unified interface for tool discovery, validation, registration,
and invocation. The ToolRegistry class offers create_tool(), read_tool(), update_tool(),
delete_tool(), and list_tools() methods for complete tool lifecycle management.

Tools can be loaded from various sources including file paths, Python modules,
decorated functions, or AgentTool instances. The registry handles validation,
normalization, and hot-reloading of tools automatically.
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
from .loader import ToolLoader

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Contains the collection of tools available to the agent.

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

        # Tool has been successfully created and registered

        return created_tool_name

    def read_tool(self, tool_name: str) -> ToolSpec:
        """Read a single tool specification.

        Args:
            tool_name: The name of the tool to retrieve.

        Returns:
            The tool specification as a ToolSpec object.

        Raises:
            ValueError: If the tool doesn't exist.
        """
        if tool_name not in self.agent_tools:
            raise ValueError(f"Tool '{tool_name}' not found in registry")

        tool = self.agent_tools[tool_name]
        # Return a copy of the spec to prevent modification of the original
        return tool.tool_spec.copy()

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
        original_tool = self.agent_tools[tool_name]

        try:
            # Use the same processing logic as create_tool but without uniqueness check
            self._process_tool(tool, validate_unique=False)
            return tool_name
        except Exception as e:
            # Restore the original tool if the update fails
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

        # No need to track tool config entries anymore

        try:
            # Delete the tool from the main registry
            del self.agent_tools[tool_name]
            logger.debug("tool_name=<%s> | deleted from main registry", tool_name)

            # No need to delete from hot reload tools anymore
            logger.debug("tool_name=<%s> | deleted from registry", tool_name)

            # No need to remove from tool config anymore

            return tool_name
        except Exception as e:
            # Restore the tool if deletion fails
            if original_tool:
                self.agent_tools[tool_name] = original_tool

            # No need to restore tool config entries anymore

            logger.warning("tool_name=<%s> | failed to delete tool | %s", tool_name, e)
            raise ValueError(f"Failed to delete tool '{tool_name}': {str(e)}") from e

    def list_tools(self) -> Dict[str, ToolSpec]:
        """List all tool specifications.

        Returns:
            A dictionary mapping tool names to their specifications.
        """
        tool_specs = {}
        for tool_name, tool in self.agent_tools.items():
            tool_specs[tool_name] = tool.tool_spec.copy()
        return tool_specs

    def _extract_tool_name(self, tool: Any) -> Optional[str]:
        """Extract tool name from various tool specification formats.

        Args:
            tool: Tool specification in various formats.

        Returns:
            Tool name if extractable, None otherwise.
        """
        # Handle string paths
        if isinstance(tool, str):
            return os.path.basename(tool).split(".")[0]
            
        # Handle dictionaries with name or path
        if isinstance(tool, dict):
            if "name" in tool:
                return str(tool["name"])
            if "path" in tool:
                return os.path.basename(str(tool["path"])).split(".")[0]
                
        # Handle Python modules
        if hasattr(tool, "__file__") and inspect.ismodule(tool):
            return tool.__name__.split(".")[-1]
            
        # Handle AgentTool instances
        if isinstance(tool, AgentTool):
            return tool.tool_name
            
        return None

    def _find_tool_directories(self) -> List[Path]:
        """Find all tool directory paths.

        Returns:
            A list of Path objects for current working directory's "./tools/".
        """
        # Current working directory's tools directory
        cwd_tools_dir = Path.cwd() / "tools"
        
        # Check if directory exists
        if cwd_tools_dir.exists() and cwd_tools_dir.is_dir():
            logger.debug("tools_dir=<%s> | found tools directory", cwd_tools_dir)
            return [cwd_tools_dir]
        else:
            logger.debug("tools_dir=<%s> | tools directory not found", cwd_tools_dir)
            return []

    def _discover_tool_modules(self) -> Dict[str, Path]:
        """Discover available tool modules in all tools directories.

        Returns:
            Dictionary mapping tool names to their full paths.
        """
        tool_modules = {}
        
        # If no tool directories found, return empty dict
        if not self.tool_directories:
            return tool_modules
            
        for tools_dir in self.tool_directories:
            logger.debug("tools_dir=<%s> | scanning", tools_dir)

            # Find Python tools (simplified to just look for .py files)
            for item in tools_dir.glob("*.py"):
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
            if not tool.supports_hot_reload:
                logger.debug("tool_name=<%s>, tool_type=<%s> | skipping hot reloading", tool.tool_name, tool.tool_type)
                return

            logger.debug(
                "tool_name=<%s>, tool_registry=<%s> | dynamic tool registered",
                tool.tool_name,
                list(self.agent_tools.keys()),
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
        # Extract tool name using the existing method
        tool_name = self._extract_tool_name(tool)
        if not tool_name:
            raise ValueError("Could not extract tool name from provided tool")
            
        # Check uniqueness if required
        if validate_unique and tool_name in self.agent_tools:
            raise ValueError(f"Tool '{tool_name}' already exists")

        # Process the tool based on its type
        try:
            # Handle file path based tools
            if isinstance(tool, str) or (isinstance(tool, dict) and "path" in tool):
                path = tool if isinstance(tool, str) else tool["path"]
                path = expanduser(str(path))
                
                # Check if file exists
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Tool file not found: {path}")
                    
                # Load and register the tool
                loaded_tool = ToolLoader.load_tool(path, tool_name)
                loaded_tool.mark_dynamic()
                self._register_tool(loaded_tool)
                
            # Handle Python module
            elif hasattr(tool, "__file__") and inspect.ismodule(tool):
                module_path = tool.__file__
                
                # Check for TOOL_SPEC in module
                if hasattr(tool, "TOOL_SPEC") and hasattr(tool, tool_name) and module_path:
                    # Load as file path
                    path = expanduser(str(module_path))
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"Module file not found: {path}")
                        
                    loaded_tool = ToolLoader.load_tool(path, tool_name)
                    loaded_tool.mark_dynamic()
                    self._register_tool(loaded_tool)
                else:
                    # Look for decorated function tools
                    function_tools = self._scan_module_for_tools(tool)
                    if function_tools:
                        # Use the first function tool found
                        function_tool = function_tools[0]
                        tool_name = function_tool.tool_name
                        self._register_tool(function_tool)
                    else:
                        logger.warning("tool_name=<%s>, module_path=<%s> | invalid agent tool", 
                                      tool_name, module_path)
                        raise ValueError(f"Module '{tool_name}' does not contain valid tools")
                        
            # Handle AgentTool instances
            elif isinstance(tool, AgentTool):
                self._register_tool(tool)
                
            else:
                raise ValueError(f"Unrecognized tool specification: {type(tool)}")
                
        except Exception as e:
            if not isinstance(e, (ValueError, FileNotFoundError)):
                logger.exception("tool_name=<%s> | failed to process tool", tool_name)
                raise ValueError(f"Failed to process tool {tool_name}: {str(e)}") from e
            raise
            
        return tool_name

    def _load_directory_tools(self) -> List[str]:
        """Load tools from standard tool directories.

        Discovers and loads tools from the standard tool directories.
        Automatically called during initialization if load_tools_from_directory=True.

        Returns:
            List of successfully loaded tool names.
        """
        tool_names = []
        tool_modules = self._discover_tool_modules()
        successful_loads = 0
        total_tools = len(tool_modules)
        
        # Skip __init__ files
        valid_modules = {name: path for name, path in tool_modules.items() if name != "__init__"}
        
        # Process Python tools from directory
        for tool_name, tool_path in valid_modules.items():
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
                    # For other validation errors, log and continue
                    logger.warning("tool_name=<%s> | validation error | %s", tool_name, e)
            except Exception as e:
                # Log any other errors and continue with other tools
                logger.warning("tool_name=<%s> | failed to load tool | %s", tool_name, e)

        # Log summary
        logger.debug("tool_count=<%d>, success_count=<%d> | finished loading tools", total_tools, successful_loads)
        
        return tool_names
