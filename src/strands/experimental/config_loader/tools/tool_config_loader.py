"""Tool configuration loader for Strands Agents.

This module provides the ToolConfigLoader class that enables loading AgentTool instances
via string identifiers, supporting both @tool decorated functions and traditional tools.
It also supports loading Agents as tools through dictionary configurations.
"""

import importlib
import inspect
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from strands.agent.agent import Agent
    from strands.multiagent.graph import Graph
    from strands.multiagent.swarm import Swarm

    from ..agent.agent_config_loader import AgentConfigLoader
    from ..graph.graph_config_loader import GraphConfigLoader
    from ..swarm.swarm_config_loader import SwarmConfigLoader

from strands.tools.decorator import DecoratedFunctionTool
from strands.tools.registry import ToolRegistry
from strands.types.tools import AgentTool, ToolSpec, ToolUse

logger = logging.getLogger(__name__)


class ModuleFunctionTool(AgentTool):
    """Wrapper for module-based tools that follow the TOOL_SPEC pattern.

    This class wraps regular functions that follow the tool pattern:
    - Function signature: (tool: ToolUse, **kwargs) -> ToolResult
    - Companion TOOL_SPEC dictionary defining the tool specification
    - No @tool decorator required

    This enables loading tools from packages like strands_tools that use
    the module-based tool pattern instead of decorators.
    """

    def __init__(self, func: Callable, tool_spec: ToolSpec, module_name: str):
        """Initialize the ModuleFunctionTool wrapper.

        Args:
            func: The tool function to wrap.
            tool_spec: Tool specification dictionary.
            module_name: Name of the module containing the tool.
        """
        super().__init__()
        self._func = func
        self._tool_spec = tool_spec
        self._module_name = module_name
        self._tool_name = tool_spec.get("name", func.__name__)

    @property
    def tool_name(self) -> str:
        """The unique name of the tool used for identification and invocation."""
        return self._tool_name

    @property
    def tool_spec(self) -> ToolSpec:
        """Tool specification that describes its functionality and parameters."""
        return self._tool_spec

    @property
    def tool_type(self) -> str:
        """The type of the tool implementation."""
        return "module_function"

    async def stream(
        self, tool_use: ToolUse, invocation_state: Dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        """Stream tool events and return the final result.

        Args:
            tool_use: The tool use request containing tool ID and parameters.
            invocation_state: Context for the tool invocation, including agent state.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Tool execution events and final result.
        """
        try:
            logger.debug("module_tool=<%s> | executing function from module: %s", self._tool_name, self._module_name)

            # Call the module function with the tool_use and kwargs
            if inspect.iscoroutinefunction(self._func):
                result = await self._func(tool_use, **kwargs)
            else:
                result = self._func(tool_use, **kwargs)

            # Ensure result is a ToolResult
            if not isinstance(result, dict) or "status" not in result:
                # Wrap simple return values in ToolResult format
                result = {"status": "success", "content": [{"text": str(result)}]}

            logger.debug("module_tool=<%s> | execution completed successfully", self._tool_name)
            yield result

        except Exception as e:
            error_msg = f"Error executing module tool '{self._tool_name}': {str(e)}"
            logger.error("module_tool=<%s> | execution failed: %s", self._tool_name, e)

            yield {"status": "error", "content": [{"text": error_msg}]}


class SwarmAsToolWrapper(AgentTool):
    """Wrapper that allows a Swarm to be used as a tool.

    This class wraps a Swarm instance and exposes it as an AgentTool,
    enabling swarms to be used as tools within other agents.
    """

    def __init__(
        self,
        swarm: "Swarm",
        tool_name: str,
        description: Optional[str] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        prompt: Optional[str] = None,
        entry_agent: Optional[str] = None,
    ):
        """Initialize the SwarmAsToolWrapper.

        Args:
            swarm: The Swarm instance to wrap as a tool.
            tool_name: The name to use for this tool.
            description: Optional description of what this swarm tool does.
            input_schema: Optional JSON Schema defining the expected input parameters.
            prompt: Optional prompt template to send to the swarm. Can contain {arg_name} placeholders.
            entry_agent: Optional specific agent name to start with.
        """
        super().__init__()
        self._swarm = swarm
        self._tool_name = tool_name
        self._description = description or f"Swarm tool: {tool_name}"
        self._input_schema = self._normalize_input_schema(input_schema or {})
        self._prompt = prompt
        self._entry_agent = entry_agent

    def _normalize_input_schema(self, input_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize input_schema to a consistent JSONSchema format."""
        # Handle empty schema
        if not input_schema:
            return {"type": "object", "properties": {}, "required": []}

        # Validate JSONSchema format
        if not isinstance(input_schema, dict):
            raise ValueError(f"input_schema must be a dictionary, got: {type(input_schema)}")

        # Ensure required JSONSchema fields
        if "type" not in input_schema:
            input_schema["type"] = "object"

        if input_schema["type"] != "object":
            raise ValueError(f"input_schema type must be 'object', got: {input_schema['type']}")

        if "properties" not in input_schema:
            input_schema["properties"] = {}

        if "required" not in input_schema:
            input_schema["required"] = []

        return input_schema

    def _extract_parameter_defaults(self) -> Dict[str, Any]:
        """Extract default values from the input schema for parameter substitution."""
        defaults = {}
        properties = self._input_schema.get("properties", {})

        for param_name, param_spec in properties.items():
            if "default" in param_spec:
                defaults[param_name] = param_spec["default"]

        return defaults

    @property
    def tool_name(self) -> str:
        """The unique name of the tool used for identification and invocation."""
        return self._tool_name

    @property
    def tool_spec(self) -> ToolSpec:
        """Tool specification that describes its functionality and parameters."""
        # Use the normalized input schema directly
        input_schema = self._input_schema.copy()

        # If no prompt template is provided and no query parameter exists, add default query parameter
        if self._prompt is None and "query" not in input_schema.get("properties", {}):
            if "properties" not in input_schema:
                input_schema["properties"] = {}
            if "required" not in input_schema:
                input_schema["required"] = []

            input_schema["properties"]["query"] = {
                "type": "string",
                "description": "The query or input to send to the swarm",
            }
            input_schema["required"].append("query")

        return {
            "name": self._tool_name,
            "description": self._description,
            "inputSchema": input_schema,
        }

    @property
    def tool_type(self) -> str:
        """The type of the tool implementation."""
        return "swarm"

    def _substitute_args(self, text: str, substitutions: Dict[str, Any]) -> str:
        """Substitute template variables in text using {arg_name} format."""
        try:
            return text.format(**substitutions)
        except KeyError as e:
            logger.warning("swarm_tool=<%s> | template substitution failed for variable: %s", self._tool_name, e)
            return text
        except Exception as e:
            logger.warning("swarm_tool=<%s> | template substitution error: %s", self._tool_name, e)
            return text

    async def stream(
        self, tool_use: ToolUse, invocation_state: Dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        """Stream tool events and return the final result."""
        try:
            # Extract the input parameters
            tool_input = tool_use.get("input", {})

            # Prepare substitution values using defaults from input schema
            substitutions = self._extract_parameter_defaults()

            # Override with values from tool input
            properties = self._input_schema.get("properties", {})
            for param_name in properties.keys():
                if param_name in tool_input:
                    substitutions[param_name] = tool_input[param_name]

            # Determine the prompt to send to the swarm
            if self._prompt is not None:
                # Use the configured prompt template with substitutions
                prompt = self._substitute_args(self._prompt, substitutions)
                logger.debug("swarm_tool=<%s> | using prompt template: %s", self._tool_name, prompt)
            else:
                # Fall back to query parameter with substitutions
                query = tool_input.get("query", "")
                prompt = self._substitute_args(query, substitutions) if substitutions else query
                logger.debug("swarm_tool=<%s> | using query with substitutions: %s", self._tool_name, prompt)

            # Call the wrapped swarm
            if self._entry_agent:
                # Start with specific agent if specified
                response = self._swarm(prompt, entry_agent=self._entry_agent)
            else:
                # Use default entry behavior
                response = self._swarm(prompt)

            # Yield the final result
            yield {
                "content": [{"text": str(response)}],
                "status": "success",
                "toolUseId": tool_use.get("toolUseId", ""),
            }

        except Exception as e:
            logger.error("swarm_tool=<%s> | execution failed: %s", self._tool_name, e)
            yield {
                "content": [{"text": f"Error in swarm tool {self._tool_name}: {str(e)}"}],
                "status": "error",
                "toolUseId": tool_use.get("toolUseId", ""),
            }


class GraphAsToolWrapper(AgentTool):
    """Wrapper that allows a Graph to be used as a tool.

    This class wraps a Graph instance and exposes it as an AgentTool,
    enabling graphs to be used as tools within other agents.
    """

    def __init__(
        self,
        graph: "Graph",
        tool_name: str,
        description: Optional[str] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        prompt: Optional[str] = None,
        entry_point: Optional[str] = None,
    ):
        """Initialize the GraphAsToolWrapper.

        Args:
            graph: The Graph instance to wrap as a tool.
            tool_name: The name to use for this tool.
            description: Optional description of what this graph tool does.
            input_schema: Optional JSON Schema defining the expected input parameters.
            prompt: Optional prompt template to send to the graph. Can contain {arg_name} placeholders.
            entry_point: Optional specific entry point node to start with.
        """
        super().__init__()
        self._graph = graph
        self._tool_name = tool_name
        self._description = description or f"Graph tool: {tool_name}"
        self._input_schema = self._normalize_input_schema(input_schema or {})
        self._prompt = prompt
        self._entry_point = entry_point

    def _normalize_input_schema(self, input_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize input_schema to a consistent JSONSchema format."""
        # Handle empty schema
        if not input_schema:
            return {"type": "object", "properties": {}, "required": []}

        # Validate JSONSchema format
        if not isinstance(input_schema, dict):
            raise ValueError(f"input_schema must be a dictionary, got: {type(input_schema)}")

        # Ensure required JSONSchema fields
        if "type" not in input_schema:
            input_schema["type"] = "object"

        if input_schema["type"] != "object":
            raise ValueError(f"input_schema type must be 'object', got: {input_schema['type']}")

        if "properties" not in input_schema:
            input_schema["properties"] = {}

        if "required" not in input_schema:
            input_schema["required"] = []

        return input_schema

    def _extract_parameter_defaults(self) -> Dict[str, Any]:
        """Extract default values from the input schema for parameter substitution."""
        defaults = {}
        properties = self._input_schema.get("properties", {})

        for param_name, param_spec in properties.items():
            if "default" in param_spec:
                defaults[param_name] = param_spec["default"]

        return defaults

    @property
    def tool_name(self) -> str:
        """The unique name of the tool used for identification and invocation."""
        return self._tool_name

    @property
    def tool_spec(self) -> ToolSpec:
        """Tool specification that describes its functionality and parameters."""
        # Use the normalized input schema directly
        input_schema = self._input_schema.copy()

        # If no prompt template is provided and no query parameter exists, add default query parameter
        if self._prompt is None and "query" not in input_schema.get("properties", {}):
            if "properties" not in input_schema:
                input_schema["properties"] = {}
            if "required" not in input_schema:
                input_schema["required"] = []

            input_schema["properties"]["query"] = {
                "type": "string",
                "description": "The query or input to send to the graph",
            }
            input_schema["required"].append("query")

        return {
            "name": self._tool_name,
            "description": self._description,
            "inputSchema": input_schema,
        }

    @property
    def tool_type(self) -> str:
        """The type of the tool implementation."""
        return "graph"

    def _substitute_args(self, text: str, substitutions: Dict[str, Any]) -> str:
        """Substitute template variables in text using {arg_name} format."""
        try:
            return text.format(**substitutions)
        except KeyError as e:
            logger.warning("graph_tool=<%s> | template substitution failed for variable: %s", self._tool_name, e)
            return text
        except Exception as e:
            logger.warning("graph_tool=<%s> | template substitution error: %s", self._tool_name, e)
            return text

    async def stream(
        self, tool_use: ToolUse, invocation_state: Dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        """Stream tool events and return the final result."""
        try:
            # Extract the input parameters
            tool_input = tool_use.get("input", {})

            # Prepare substitution values using defaults from input schema
            substitutions = self._extract_parameter_defaults()

            # Override with values from tool input
            properties = self._input_schema.get("properties", {})
            for param_name in properties.keys():
                if param_name in tool_input:
                    substitutions[param_name] = tool_input[param_name]

            # Determine the prompt to send to the graph
            if self._prompt is not None:
                # Use the configured prompt template with substitutions
                prompt = self._substitute_args(self._prompt, substitutions)
                logger.debug("graph_tool=<%s> | using prompt template: %s", self._tool_name, prompt)
            else:
                # Fall back to query parameter with substitutions
                query = tool_input.get("query", "")
                prompt = self._substitute_args(query, substitutions) if substitutions else query
                logger.debug("graph_tool=<%s> | using query with substitutions: %s", self._tool_name, prompt)

            # Call the wrapped graph
            if self._entry_point:
                # Start with specific entry point if specified
                response = self._graph(prompt, entry_point=self._entry_point)
            else:
                # Use default entry behavior
                response = self._graph(prompt)

            # Yield the final result
            yield {
                "content": [{"text": str(response)}],
                "status": "success",
                "toolUseId": tool_use.get("toolUseId", ""),
            }

        except Exception as e:
            logger.error("graph_tool=<%s> | execution failed: %s", self._tool_name, e)
            yield {
                "content": [{"text": f"Error in graph tool {self._tool_name}: {str(e)}"}],
                "status": "error",
                "toolUseId": tool_use.get("toolUseId", ""),
            }


class AgentAsToolWrapper(AgentTool):
    """Wrapper that allows an Agent to be used as a tool.

    This class wraps an Agent instance and exposes it as an AgentTool,
    enabling agents to be used as tools within other agents.
    """

    def __init__(
        self,
        agent: "Agent",
        tool_name: str,
        description: Optional[str] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        prompt: Optional[str] = None,
    ):
        """Initialize the AgentAsToolWrapper.

        Args:
            agent: The Agent instance to wrap as a tool.
            tool_name: The name to use for this tool.
            description: Optional description of what this agent tool does.
            input_schema: Optional JSON Schema defining the expected input parameters.
                         Should follow the JSONSchema format used in ToolSpec:
                         {
                             "type": "object",
                             "properties": {
                                 "arg_name": {
                                     "type": "string",
                                     "description": "Argument description"
                                 }
                             },
                             "required": ["arg_name"]
                         }
            prompt: Optional prompt template to send to the agent. Can contain {arg_name} placeholders
                   that will be replaced with argument values. If not provided, uses the query directly.
        """
        super().__init__()
        self._agent = agent
        self._tool_name = tool_name
        self._description = description or f"Agent tool: {tool_name}"
        self._input_schema = self._normalize_input_schema(input_schema or {})
        self._prompt = prompt

    def _normalize_input_schema(self, input_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize input_schema to a consistent JSONSchema format.

        Args:
            input_schema: Input schema dictionary in JSONSchema format:
                         {
                             "type": "object",
                             "properties": {
                                 "param_name": {
                                     "type": "string",
                                     "description": "Parameter description"
                                 }
                             },
                             "required": ["param_name"]
                         }

        Returns:
            Normalized JSONSchema dict with required fields filled in.

        Raises:
            ValueError: If input_schema has invalid format.
        """
        # Handle empty schema
        if not input_schema:
            return {"type": "object", "properties": {}, "required": []}

        # Validate JSONSchema format
        if not isinstance(input_schema, dict):
            raise ValueError(f"input_schema must be a dictionary, got: {type(input_schema)}")

        # Ensure required JSONSchema fields
        if "type" not in input_schema:
            input_schema["type"] = "object"

        if input_schema["type"] != "object":
            raise ValueError(f"input_schema type must be 'object', got: {input_schema['type']}")

        if "properties" not in input_schema:
            input_schema["properties"] = {}

        if "required" not in input_schema:
            input_schema["required"] = []

        return input_schema

    def _extract_parameter_defaults(self) -> Dict[str, Any]:
        """Extract default values from the input schema for parameter substitution."""
        defaults = {}
        properties = self._input_schema.get("properties", {})

        for param_name, param_spec in properties.items():
            if "default" in param_spec:
                defaults[param_name] = param_spec["default"]

        return defaults

    @property
    def tool_name(self) -> str:
        """The unique name of the tool used for identification and invocation."""
        return self._tool_name

    @property
    def tool_spec(self) -> ToolSpec:
        """Tool specification that describes its functionality and parameters."""
        # Use the normalized input schema directly
        input_schema = self._input_schema.copy()

        # If no prompt template is provided and no query parameter exists, add default query parameter
        if self._prompt is None and "query" not in input_schema.get("properties", {}):
            if "properties" not in input_schema:
                input_schema["properties"] = {}
            if "required" not in input_schema:
                input_schema["required"] = []

            input_schema["properties"]["query"] = {
                "type": "string",
                "description": "The query or input to send to the agent",
            }
            input_schema["required"].append("query")

        return {
            "name": self._tool_name,
            "description": self._description,
            "inputSchema": input_schema,
        }

    @property
    def tool_type(self) -> str:
        """The type of the tool implementation."""
        return "agent"

    def _substitute_args(self, text: str, substitutions: Dict[str, Any]) -> str:
        """Substitute template variables in text using {arg_name} format.

        Args:
            text: Text containing template variables like {arg1}, {arg2}
            substitutions: Dictionary of variable names to values

        Returns:
            Text with variables substituted
        """
        try:
            return text.format(**substitutions)
        except KeyError as e:
            logger.warning("agent_tool=<%s> | template substitution failed for variable: %s", self._tool_name, e)
            return text
        except Exception as e:
            logger.warning("agent_tool=<%s> | template substitution error: %s", self._tool_name, e)
            return text

    async def stream(
        self, tool_use: ToolUse, invocation_state: Dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        """Stream tool events and return the final result.

        Args:
            tool_use: The tool use request containing tool ID and parameters.
            invocation_state: Context for the tool invocation, including agent state.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Tool execution events and final result.
        """
        try:
            # Extract the input parameters
            tool_input = tool_use.get("input", {})

            # Prepare substitution values using defaults from input schema
            substitutions = self._extract_parameter_defaults()

            # Override with values from tool input
            properties = self._input_schema.get("properties", {})
            for param_name in properties.keys():
                if param_name in tool_input:
                    substitutions[param_name] = tool_input[param_name]

            # Determine the prompt to send to the agent
            if self._prompt is not None:
                # Use the configured prompt template with substitutions
                prompt = self._substitute_args(self._prompt, substitutions)
                logger.debug("agent_tool=<%s> | using prompt template: %s", self._tool_name, prompt)
            else:
                # Fall back to query parameter with substitutions
                query = tool_input.get("query", "")
                prompt = self._substitute_args(query, substitutions) if substitutions else query
                logger.debug("agent_tool=<%s> | using query with substitutions: %s", self._tool_name, prompt)

            # Call the wrapped agent
            response = self._agent(prompt)

            # Yield the final result
            yield {
                "content": [{"text": str(response)}],
                "status": "success",
                "toolUseId": tool_use.get("toolUseId", ""),
            }

        except Exception as e:
            logger.error("agent_tool=<%s> | execution failed: %s", self._tool_name, e)
            yield {
                "content": [{"text": f"Error in agent tool {self._tool_name}: {str(e)}"}],
                "status": "error",
                "toolUseId": tool_use.get("toolUseId", ""),
            }


class ToolConfigLoader:
    """Loads Strands AgentTool instances via string identifiers or multi-agent configurations.

    This class provides functionality to load tools decorated with the @tool decorator
    by their string identifier, and also supports loading Agents, Swarms, and Graphs as tools
    through dictionary configurations using convention-based type detection.

    The loader supports multiple resolution strategies:
    1. Direct function name lookup in modules
    2. Class name lookup for tool classes
    3. Registry-based lookup for registered tools
    4. Module path-based loading
    5. Multi-agent configuration loading (Agent, Swarm, Graph)

    Convention-based type detection:
    - Presence of "swarm" key → Swarm tool
    - Presence of "graph" key → Graph tool
    - Presence of "agent" key → Agent tool
    - Presence of "module" key → Legacy module-based tool
    """

    def __init__(self, registry: Optional[ToolRegistry] = None):
        """Initialize the ToolConfigLoader.

        Args:
            registry: Optional ToolRegistry instance to use for tool lookup.
                     If not provided, a new registry will be created.
        """
        self._registry = registry or ToolRegistry()
        self._agent_config_loader: Optional["AgentConfigLoader"] = None
        self._swarm_config_loader: Optional["SwarmConfigLoader"] = None
        self._graph_config_loader: Optional["GraphConfigLoader"] = None

    def _get_agent_config_loader(self) -> "AgentConfigLoader":
        """Get or create an AgentConfigLoader instance.

        This method implements lazy loading to avoid circular imports.

        Returns:
            AgentConfigLoader instance.
        """
        if self._agent_config_loader is None:
            # Import here to avoid circular imports
            from ..agent.agent_config_loader import AgentConfigLoader

            self._agent_config_loader = AgentConfigLoader(tool_config_loader=self)
        return self._agent_config_loader

    def _get_swarm_config_loader(self) -> "SwarmConfigLoader":
        """Get or create a SwarmConfigLoader instance.

        This method implements lazy loading to avoid circular imports.

        Returns:
            SwarmConfigLoader instance.
        """
        if self._swarm_config_loader is None:
            # Import here to avoid circular imports
            from ..swarm.swarm_config_loader import SwarmConfigLoader

            self._swarm_config_loader = SwarmConfigLoader(agent_config_loader=self._get_agent_config_loader())
        return self._swarm_config_loader

    def _get_graph_config_loader(self) -> "GraphConfigLoader":
        """Get or create a GraphConfigLoader instance.

        This method implements lazy loading to avoid circular imports.

        Returns:
            GraphConfigLoader instance.
        """
        if self._graph_config_loader is None:
            # Import here to avoid circular imports
            from ..graph.graph_config_loader import GraphConfigLoader

            self._graph_config_loader = GraphConfigLoader(
                agent_loader=self._get_agent_config_loader(), swarm_loader=self._get_swarm_config_loader()
            )
        return self._graph_config_loader

    def _determine_config_type(self, config: Dict[str, Any]) -> str:
        """Determine configuration type from dictionary structure using convention over configuration.

        Detection priority order:
        1. Presence of "swarm" field → "swarm"
        2. Presence of "graph" field → "graph"
        3. Presence of "agent" field → "agent"
        4. Presence of "module" field → "legacy_tool"
        5. Default → "agent" (backward compatibility)

        Args:
            config: Configuration dictionary to analyze.

        Returns:
            String indicating the detected configuration type.
        """
        if "swarm" in config:
            return "swarm"
        elif "graph" in config:
            return "graph"
        elif "agent" in config:
            return "agent"
        elif "module" in config:
            return "legacy_tool"
        else:
            # Default to agent for backward compatibility
            return "agent"

    def load_tool(self, tool: Union[str, Dict[str, Any]], module_path: Optional[str] = None) -> AgentTool:
        """Load a tool by its string identifier or configuration.

        Args:
            tool: Tool specification. Can be:
                 - String identifier for the tool (function name, class name, fully qualified name)
                 - Dictionary containing configuration (type auto-detected):
                   * {"name": "...", "agent": {...}} → Agent tool
                   * {"name": "...", "swarm": {...}} → Swarm tool
                   * {"name": "...", "graph": {...}} → Graph tool
                   * {"name": "...", "module": "..."} → Legacy tool format
            module_path: Optional path to the module containing the tool.
                        If not provided, will attempt to resolve from identifier.
                        Only used when tool is a string.

        Returns:
            AgentTool instance for the specified identifier or configuration.

        Raises:
            ValueError: If the tool cannot be found or loaded.
            ImportError: If the module cannot be imported.
        """
        # Handle dictionary configuration
        if isinstance(tool, dict):
            return self._load_config_tool(tool)

        # Handle string identifier (existing functionality)
        return self._load_string_tool(tool, module_path)

    def _load_config_tool(self, tool_config: Dict[str, Any]) -> AgentTool:
        """Load a tool from dictionary configuration using convention-based type detection.

        Args:
            tool_config: Dictionary containing tool configuration.

        Returns:
            AgentTool instance for the specified configuration.

        Raises:
            ValueError: If required configuration is missing or invalid.
        """
        # Determine configuration type using convention
        config_type = self._determine_config_type(tool_config)

        # Dispatch to appropriate loader
        if config_type == "swarm":
            return self._load_swarm_as_tool(tool_config)
        elif config_type == "graph":
            return self._load_graph_as_tool(tool_config)
        elif config_type == "agent":
            return self._load_agent_as_tool(tool_config)
        elif config_type == "legacy_tool":
            return self._load_legacy_tool(tool_config)
        else:
            raise ValueError(f"Unknown configuration type: {config_type}")

    def _load_swarm_as_tool(self, tool_config: Dict[str, Any]) -> AgentTool:
        """Load a Swarm as a tool from dictionary configuration.

        Args:
            tool_config: Dictionary containing swarm tool configuration.
                        Expected format:
                        {
                            "name": "tool_name",
                            "description": "Tool description",
                            "input_schema": {...},
                            "prompt": "Prompt template with {arg_name} substitution",
                            "entry_agent": "agent_name",  # optional
                            "swarm": {
                                "max_handoffs": 10,
                                "agents": [...]
                            }
                        }

        Returns:
            SwarmAsToolWrapper instance wrapping the configured swarm.

        Raises:
            ValueError: If required configuration is missing.
        """
        # Extract tool metadata
        tool_name = tool_config.get("name")
        if not tool_name:
            raise ValueError("Swarm tool configuration must include 'name' field")

        description = tool_config.get("description")
        input_schema = tool_config.get("input_schema", {})
        prompt = tool_config.get("prompt")
        entry_agent = tool_config.get("entry_agent")

        # Extract swarm configuration
        swarm_config = tool_config.get("swarm")
        if not swarm_config:
            raise ValueError("Swarm tool configuration must include 'swarm' field")

        try:
            # Load the swarm using SwarmConfigLoader
            # Wrap the swarm config in the required top-level 'swarm' key
            swarm_loader = self._get_swarm_config_loader()
            wrapped_swarm_config = {"swarm": swarm_config}
            swarm = swarm_loader.load_swarm(wrapped_swarm_config)

            # Wrap the swarm as a tool
            swarm_tool = SwarmAsToolWrapper(
                swarm=swarm,
                tool_name=tool_name,
                description=description,
                input_schema=input_schema,
                prompt=prompt,
                entry_agent=entry_agent,
            )

            return swarm_tool

        except Exception as e:
            logger.error("swarm_tool=<%s> | failed to load: %s", tool_name, e)
            raise ValueError(f"Failed to load swarm tool '{tool_name}': {str(e)}") from e

    def _transform_graph_config(
        self, graph_config: Dict[str, Any], entry_point: Optional[str] = None
    ) -> Dict[str, Any]:
        """Transform graph configuration from tool format to GraphConfigLoader format.

        This method converts between the tool configuration format (which uses 'name' for nodes
        and 'from'/'to' for edges) and the GraphConfigLoader format (which uses 'node_id' for nodes
        and 'from_node'/'to_node' for edges).

        Args:
            graph_config: Graph configuration in tool format.
            entry_point: Optional entry point from tool configuration.

        Returns:
            Graph configuration in GraphConfigLoader format.
        """
        config_copy = graph_config.copy()

        # Transform nodes: convert 'name' to 'node_id' and add 'type' field
        if "nodes" in config_copy:
            transformed_nodes = []
            for node in config_copy["nodes"]:
                if isinstance(node, dict):
                    transformed_node = node.copy()

                    # Convert 'name' to 'node_id'
                    if "name" in transformed_node:
                        transformed_node["node_id"] = transformed_node.pop("name")

                    # Add 'type' field based on what's present in the node
                    if "agent" in transformed_node:
                        transformed_node["type"] = "agent"
                        # Move agent config to 'config' field
                        transformed_node["config"] = transformed_node.pop("agent")
                    elif "swarm" in transformed_node:
                        transformed_node["type"] = "swarm"
                        # Move swarm config to 'config' field
                        transformed_node["config"] = transformed_node.pop("swarm")
                    elif "graph" in transformed_node:
                        transformed_node["type"] = "graph"
                        # Move graph config to 'config' field
                        transformed_node["config"] = transformed_node.pop("graph")
                    else:
                        # Default to agent type
                        transformed_node["type"] = "agent"

                    transformed_nodes.append(transformed_node)
                else:
                    # Handle string node references
                    transformed_nodes.append({"node_id": str(node), "type": "agent"})

            config_copy["nodes"] = transformed_nodes

        # Transform edges: convert 'from'/'to' to 'from_node'/'to_node'
        if "edges" in config_copy:
            transformed_edges = []
            for edge in config_copy["edges"]:
                if isinstance(edge, dict):
                    transformed_edge = edge.copy()

                    # Convert 'from' to 'from_node'
                    if "from" in transformed_edge:
                        transformed_edge["from_node"] = transformed_edge.pop("from")

                    # Convert 'to' to 'to_node'
                    if "to" in transformed_edge:
                        transformed_edge["to_node"] = transformed_edge.pop("to")

                    # Transform condition if present
                    if "condition" in transformed_edge and isinstance(transformed_edge["condition"], dict):
                        condition = transformed_edge["condition"]
                        if condition.get("type") == "always":
                            # Convert "always" condition to expression that always returns True
                            transformed_edge["condition"] = {
                                "type": "expression",
                                "expression": "True",
                                "description": condition.get("description", "Always proceed"),
                            }

                    transformed_edges.append(transformed_edge)
                else:
                    transformed_edges.append(edge)

            config_copy["edges"] = transformed_edges

        # Handle entry points
        if "entry_point" in config_copy and "entry_points" not in config_copy:
            # Convert singular entry_point to plural entry_points list
            entry_point_value = config_copy.pop("entry_point")
            config_copy["entry_points"] = [entry_point_value]
        elif entry_point and "entry_points" not in config_copy:
            # Use entry_point from tool configuration
            config_copy["entry_points"] = [entry_point]
        elif "entry_points" not in config_copy:
            # If no entry points specified, use the first node as default
            nodes = config_copy.get("nodes", [])
            if nodes:
                first_node = nodes[0]
                if isinstance(first_node, dict):
                    first_node_id = first_node.get("node_id") or first_node.get("name")
                else:
                    first_node_id = str(first_node)
                config_copy["entry_points"] = [first_node_id]
            else:
                raise ValueError("Graph configuration must have at least one node to determine entry point")

        return config_copy

    def _load_graph_as_tool(self, tool_config: Dict[str, Any]) -> AgentTool:
        """Load a Graph as a tool from dictionary configuration.

        Args:
            tool_config: Dictionary containing graph tool configuration.
                        Expected format:
                        {
                            "name": "tool_name",
                            "description": "Tool description",
                            "input_schema": {...},
                            "prompt": "Prompt template with {arg_name} substitution",
                            "entry_point": "node_id",  # optional
                            "graph": {
                                "nodes": [...],
                                "edges": [...],
                                "entry_points": [...]
                            }
                        }

        Returns:
            GraphAsToolWrapper instance wrapping the configured graph.

        Raises:
            ValueError: If required configuration is missing.
        """
        # Extract tool metadata
        tool_name = tool_config.get("name")
        if not tool_name:
            raise ValueError("Graph tool configuration must include 'name' field")

        description = tool_config.get("description")
        input_schema = tool_config.get("input_schema", {})
        prompt = tool_config.get("prompt")
        entry_point = tool_config.get("entry_point")

        # Extract graph configuration
        graph_config = tool_config.get("graph")
        if not graph_config:
            raise ValueError("Graph tool configuration must include 'graph' field")

        try:
            # Load the graph using GraphConfigLoader
            graph_loader = self._get_graph_config_loader()

            # Transform the graph configuration to match GraphConfigLoader expectations
            graph_config_copy = self._transform_graph_config(graph_config, entry_point)

            # Wrap the graph config in the required top-level 'graph' key
            wrapped_graph_config = {"graph": graph_config_copy}
            graph = graph_loader.load_graph(wrapped_graph_config)

            # Wrap the graph as a tool
            graph_tool = GraphAsToolWrapper(
                graph=graph,
                tool_name=tool_name,
                description=description,
                input_schema=input_schema,
                prompt=prompt,
                entry_point=entry_point,
            )

            return graph_tool

        except Exception as e:
            logger.error("graph_tool=<%s> | failed to load: %s", tool_name, e)
            raise ValueError(f"Failed to load graph tool '{tool_name}': {str(e)}") from e

    def _load_legacy_tool(self, tool_config: Dict[str, Any]) -> AgentTool:
        """Load a legacy tool from dictionary configuration.

        Args:
            tool_config: Dictionary containing legacy tool configuration.
                        Expected format: {"name": "tool_name", "module": "module_path"}

        Returns:
            AgentTool instance for the legacy tool.
        """
        name = tool_config.get("name")
        module_path = tool_config.get("module")
        if not name:
            raise ValueError("Legacy tool configuration must include 'name' field")
        if not module_path:
            raise ValueError("Legacy tool configuration must include 'module' field")

        return self._load_string_tool(name, module_path)

    def _load_agent_as_tool(self, tool_config: Dict[str, Any]) -> AgentTool:
        """Load an Agent as a tool from dictionary configuration.

        Args:
            tool_config: Dictionary containing agent configuration and tool metadata.
                        Expected format:
                        {
                            "name": "tool_name",
                            "description": "Tool description",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "arg_name": {
                                        "type": "string",
                                        "description": "Argument description"
                                    }
                                },
                                "required": ["arg_name"]
                            },
                            "agent": {
                                "model": "model_id",
                                "system_prompt": "System prompt for the agent",
                                "prompt": "Prompt template with {arg_name} substitution",  # optional
                                "tools": [...]
                            }
                        }

        Returns:
            AgentAsToolWrapper instance wrapping the configured agent.

        Raises:
            ValueError: If required configuration is missing.
        """
        # Extract tool metadata
        tool_name = tool_config.get("name")
        if not tool_name:
            raise ValueError("Agent tool configuration must include 'name' field")

        description = tool_config.get("description")
        input_schema = tool_config.get("input_schema", {})

        # Extract agent configuration
        agent_config = tool_config.get("agent")
        if not agent_config:
            raise ValueError("Agent tool configuration must include 'agent' field")

        # Extract prompt template from agent config
        prompt = agent_config.get("prompt")

        try:
            # Load the agent using AgentConfigLoader
            # Wrap the agent config in the required top-level 'agent' key
            agent_loader = self._get_agent_config_loader()
            wrapped_agent_config = {"agent": agent_config}
            agent = agent_loader.load_agent(wrapped_agent_config)

            # Wrap the agent as a tool
            agent_tool = AgentAsToolWrapper(
                agent=agent, tool_name=tool_name, description=description, input_schema=input_schema, prompt=prompt
            )

            return agent_tool

        except Exception as e:
            logger.error("agent_tool=<%s> | failed to load: %s", tool_name, e)
            raise ValueError(f"Failed to load agent tool '{tool_name}': {str(e)}") from e

    def _load_string_tool(self, identifier: str, module_path: Optional[str] = None) -> AgentTool:
        """Load a tool by its string identifier (existing functionality).

        Args:
            identifier: String identifier for the tool.
            module_path: Optional path to the module containing the tool.

        Returns:
            AgentTool instance for the specified identifier.
        """
        tool = None

        # Strategy 1: Check registry for already loaded tools
        if identifier in self._registry.registry:
            tool = self._registry.registry[identifier]
            logger.debug("tool_identifier=<%s> | found in registry", identifier)

        # Strategy 2: Try to load from module path
        elif module_path:
            tool = self._load_from_module_path(identifier, module_path)

        # Strategy 3: Try to resolve fully qualified name
        elif "." in identifier:
            module_name, tool_name = identifier.rsplit(".", 1)
            tool = self._load_from_module_name(tool_name, module_name)

        # Strategy 4: Search in common locations
        else:
            tool = self._search_for_tool(identifier)

        if tool is None:
            raise ValueError(f"Tool '{identifier}' not found")

        return tool

    def load_tools(self, identifiers: List[Union[str, Dict[str, Any]]]) -> List[AgentTool]:
        """Load multiple tools by their identifiers.

        Args:
            identifiers: List of tool identifiers. Each can be:
                        - String identifier
                        - Dict with tool configuration (agent, swarm, graph, or legacy)

        Returns:
            List of AgentTool instances.

        Raises:
            ValueError: If any tool cannot be found or loaded.
        """
        tools = []

        for item in identifiers:
            if isinstance(item, str):
                tool = self.load_tool(item)
            elif isinstance(item, dict):
                # Use convention-based detection for all dictionary configurations
                tool = self.load_tool(item)
            else:
                raise ValueError(f"Invalid tool specification: {item}")

            tools.append(tool)

        return tools

    def get_available_tools(self, module_path: Optional[str] = None) -> List[str]:
        """Get list of available tool identifiers.

        Args:
            module_path: Optional path to scan for tools. If not provided,
                        returns tools from the registry.

        Returns:
            List of available tool identifiers.
        """
        if module_path:
            return self._scan_module_for_tool_names(module_path)
        else:
            return list(self._registry.registry.keys())

    def _load_from_module_path(self, identifier: str, module_path: str) -> Optional[AgentTool]:
        """Load a tool from a specific module path.

        Args:
            identifier: Tool identifier to find in the module.
            module_path: Path to the module file.

        Returns:
            AgentTool instance if found, None otherwise.
        """
        try:
            module = self._import_module_from_path(module_path)
            return self._extract_tool_from_module(identifier, module)
        except Exception as e:
            logger.warning("module_path=<%s>, identifier=<%s> | failed to load | %s", module_path, identifier, e)
            return None

    def _load_from_module_name(self, tool_name: str, module_name: str) -> Optional[AgentTool]:
        """Load a tool from a module by name.

        Args:
            tool_name: Name of the tool to find.
            module_name: Name of the module to import.

        Returns:
            AgentTool instance if found, None otherwise.
        """
        try:
            # First try to import the module directly
            module = importlib.import_module(module_name)

            # Try to find the tool directly
            tool = self._extract_tool_from_module(tool_name, module)
            if tool:
                return tool

            # Special case: if tool_name matches the last part of module_name,
            # try to find a tool with the same name in the module
            # This handles cases like 'strands_tools.file_write' where we want 'file_write' tool
            module_basename = module_name.split(".")[-1]
            if tool_name == module_basename:
                # Try to find a tool function with the same name as the module
                tool = self._extract_tool_from_module(tool_name, module)
                if tool:
                    return tool

                # Also check if there's a TOOL_SPEC that matches
                if hasattr(module, "TOOL_SPEC"):
                    spec = module.TOOL_SPEC
                    if isinstance(spec, dict) and spec.get("name") == tool_name:
                        if hasattr(module, tool_name):
                            func = getattr(module, tool_name)
                            if callable(func) and self._is_tool_function(func):
                                return ModuleFunctionTool(func, spec, module_name)  # type: ignore[arg-type]

            # If we didn't find the tool in the main module, try as a submodule
            # This handles cases like 'strands_tools.file_write' where file_write is a submodule
            try:
                full_module_name = f"{module_name}.{tool_name}"
                submodule = importlib.import_module(full_module_name)

                # Look for a tool with the same name as the submodule
                tool = self._extract_tool_from_module(tool_name, submodule)
                if tool:
                    return tool

            except ImportError:
                # Submodule doesn't exist, that's okay
                pass

            return None

        except ImportError:
            # If direct import fails, try importing as a submodule only
            try:
                full_module_name = f"{module_name}.{tool_name}"
                submodule = importlib.import_module(full_module_name)

                # Look for a tool with the same name as the submodule
                tool = self._extract_tool_from_module(tool_name, submodule)
                if tool:
                    return tool

                return None

            except ImportError:
                logger.warning(
                    "module_name=<%s>, tool_name=<%s> | neither direct nor submodule import succeeded",
                    module_name,
                    tool_name,
                )
                return None

        except Exception as e:
            logger.warning("module_name=<%s>, tool_name=<%s> | failed to load | %s", module_name, tool_name, e)
            return None

    def _search_for_tool(self, identifier: str) -> Optional[AgentTool]:
        """Search for a tool in common locations.

        Args:
            identifier: Tool identifier to search for.

        Returns:
            AgentTool instance if found, None otherwise.
        """
        # Search in tools directory
        tools_dir = Path.cwd() / "tools"
        if tools_dir.exists():
            for py_file in tools_dir.glob("*.py"):
                if py_file.stem == identifier or py_file.stem == "__init__":
                    tool = self._load_from_module_path(identifier, str(py_file))
                    if tool:
                        return tool

        # Search in current working directory
        cwd = Path.cwd()
        for py_file in cwd.glob("*.py"):
            if py_file.stem == identifier:
                tool = self._load_from_module_path(identifier, str(py_file))
                if tool:
                    return tool

        return None

    def _import_module_from_path(self, module_path: str) -> Any:
        """Import a module from a file path.

        Args:
            module_path: Path to the Python module file.

        Returns:
            Imported module object.

        Raises:
            ImportError: If the module cannot be imported.
        """
        path = Path(module_path)
        if not path.exists():
            raise ImportError(f"Module file not found: {module_path}")

        # Import the module
        spec = importlib.util.spec_from_file_location(path.stem, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec from: {module_path}")

        module = importlib.util.module_from_spec(spec)

        # Add to sys.modules temporarily to handle relative imports
        sys.modules[path.stem] = module
        try:
            spec.loader.exec_module(module)
        finally:
            # Clean up sys.modules if it wasn't there before
            if path.stem in sys.modules and sys.modules[path.stem] is module:
                del sys.modules[path.stem]

        return module

    def _extract_tool_from_module(self, identifier: str, module: Any) -> Optional[AgentTool]:
        """Extract a tool from a module by identifier.

        Args:
            identifier: Tool identifier to find.
            module: Module object to search in.

        Returns:
            AgentTool instance if found, None otherwise.
        """
        # Strategy 1: Look for @tool decorated functions
        for name, obj in inspect.getmembers(module):
            if isinstance(obj, DecoratedFunctionTool):
                if name == identifier or obj.tool_name == identifier:
                    return obj

        # Strategy 2: Look for AgentTool subclasses
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if name == identifier and issubclass(obj, AgentTool) and obj is not AgentTool:
                try:
                    return obj()
                except Exception as e:
                    logger.warning("class_name=<%s> | failed to instantiate | %s", name, e)

        # Strategy 3: Look for functions that might be tools
        if hasattr(module, identifier):
            obj = getattr(module, identifier)
            if isinstance(obj, AgentTool):
                return obj

        # Strategy 4: Look for module-based tools with TOOL_SPEC pattern
        tool_func = None
        tool_spec = None

        # Check if the identifier matches a function in the module
        if hasattr(module, identifier):
            potential_func = getattr(module, identifier)
            if callable(potential_func) and self._is_tool_function(potential_func):
                tool_func = potential_func

        # Look for corresponding TOOL_SPEC
        if tool_func and hasattr(module, "TOOL_SPEC"):
            spec = module.TOOL_SPEC
            if isinstance(spec, dict) and spec.get("name") == identifier:
                tool_spec = spec

        # Create wrapper if both function and spec found
        if tool_func and tool_spec:
            logger.debug("module_tool=<%s> | found module-based tool in %s", identifier, module.__name__)
            return ModuleFunctionTool(tool_func, tool_spec, module.__name__)  # type: ignore[arg-type]

        return None

    def _is_tool_function(self, func: Callable) -> bool:
        """Check if a function matches the tool function signature pattern.

        Expected pattern: (tool: ToolUse, **kwargs: Any) -> ToolResult

        Args:
            func: Function to check.

        Returns:
            True if function matches tool pattern, False otherwise.
        """
        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.values())

            # Must have at least one parameter
            if len(params) < 1:
                return False

            first_param = params[0]

            # Check if first parameter could be ToolUse
            # Look for common parameter names or type annotations
            if (
                first_param.name in ["tool", "tool_use"]
                or "ToolUse" in str(first_param.annotation)
                or "tool_use" in str(first_param.annotation).lower()
            ):
                return True

            # Check if function has **kwargs (common pattern for tool functions)
            has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
            if has_var_keyword and len(params) >= 1:
                return True

            return False
        except Exception as e:
            logger.debug("function=<%s> | signature check failed: %s", getattr(func, "__name__", "unknown"), e)
            return False

    def _scan_module_for_tools(self, module: Any) -> List[Tuple[str, str]]:
        """Scan module for all available tools (decorated and module-based).

        Args:
            module: Module object to scan.

        Returns:
            List of tuples (tool_name, tool_type).
        """
        tools = []

        # Find @tool decorated functions
        for _name, obj in inspect.getmembers(module):
            if isinstance(obj, DecoratedFunctionTool):
                tools.append((obj.tool_name, "decorated"))

        # Find AgentTool subclasses
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, AgentTool) and obj is not AgentTool:
                tools.append((name, "class"))

        # Scan for module-based tools
        if hasattr(module, "TOOL_SPEC"):
            tool_spec = module.TOOL_SPEC
            if isinstance(tool_spec, dict) and "name" in tool_spec:
                tool_name = tool_spec["name"]
                if hasattr(module, tool_name):
                    func = getattr(module, tool_name)
                    if callable(func) and self._is_tool_function(func):
                        tools.append((tool_name, "module_function"))

        return tools

    def _scan_module_for_tool_names(self, module_path: str) -> List[str]:
        """Scan a module for available tool names.

        Args:
            module_path: Path to the module to scan.

        Returns:
            List of tool names found in the module.
        """
        try:
            module = self._import_module_from_path(module_path)
            tools = self._scan_module_for_tools(module)
            return [tool_name for tool_name, tool_type in tools]

        except Exception as e:
            logger.warning("module_path=<%s> | failed to scan | %s", module_path, e)
            return []
