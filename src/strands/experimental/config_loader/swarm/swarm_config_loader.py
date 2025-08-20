"""Swarm configuration loader for Strands Agents.

This module provides the SwarmConfigLoader class that enables creating Swarm instances
from YAML/dictionary configurations, supporting serialization and deserialization of Swarm
configurations for persistence and dynamic loading scenarios.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..agent.agent_config_loader import AgentConfigLoader

from strands.agent.agent import Agent
from strands.multiagent.swarm import Swarm

logger = logging.getLogger(__name__)


class SwarmConfigLoader:
    """Loads and serializes Strands Swarm instances via YAML/dictionary configurations.

    This class provides functionality to create Swarm instances from YAML/dictionary
    configurations and serialize existing Swarm instances to dictionaries for
    persistence and configuration management.

    The loader supports:
    1. Loading swarms from YAML/dictionary configurations
    2. Serializing swarms to YAML-compatible dictionary configurations
    3. Agent loading via AgentConfigLoader integration
    4. Caching for performance optimization
    5. Configuration validation and error handling
    """

    def __init__(self, agent_config_loader: Optional["AgentConfigLoader"] = None):
        """Initialize the SwarmConfigLoader.

        Args:
            agent_config_loader: Optional AgentConfigLoader instance for loading agents.
                                If not provided, will be imported and created when needed.
        """
        self._agent_config_loader = agent_config_loader

    def _get_agent_config_loader(self) -> "AgentConfigLoader":
        """Get or create an AgentConfigLoader instance.

        This method implements lazy loading to avoid circular imports.

        Returns:
            AgentConfigLoader instance.
        """
        if self._agent_config_loader is None:
            # Import here to avoid circular imports
            from ..agent.agent_config_loader import AgentConfigLoader

            self._agent_config_loader = AgentConfigLoader()
        return self._agent_config_loader

    def load_swarm(self, config: Dict[str, Any]) -> Swarm:
        """Load a Swarm from configuration dictionary.

        Args:
            config: Dictionary containing swarm configuration with top-level 'swarm' key.

        Returns:
            Swarm instance configured according to the provided dictionary.

        Raises:
            ValueError: If required configuration is missing or invalid.
            ImportError: If specified models or tools cannot be imported.
        """
        # Validate top-level structure
        if "swarm" not in config:
            raise ValueError("Configuration must contain a top-level 'swarm' key")

        swarm_config = config["swarm"]
        if not isinstance(swarm_config, dict):
            raise ValueError("The 'swarm' configuration must be a dictionary")

        # Validate configuration structure
        self._validate_config(swarm_config)

        # Extract agents configuration
        agents_config = swarm_config.get("agents", [])
        if not agents_config:
            raise ValueError("Swarm configuration must include 'agents' field with at least one agent")

        # Load agents using AgentConfigLoader
        agents = self.load_agents(agents_config)

        # Extract swarm parameters
        swarm_params = self._extract_swarm_parameters(swarm_config)

        # Create swarm
        swarm = Swarm(nodes=agents, **swarm_params)

        return swarm

    def serialize_swarm(self, swarm: Swarm) -> Dict[str, Any]:
        """Serialize a Swarm instance to YAML-compatible dictionary configuration.

        Args:
            swarm: Swarm instance to serialize.

        Returns:
            Dictionary containing the swarm's configuration with top-level 'swarm' key.
        """
        swarm_config = {}

        # Serialize swarm parameters (only include non-default values)
        if swarm.max_handoffs != 20:
            swarm_config["max_handoffs"] = swarm.max_handoffs
        if swarm.max_iterations != 20:
            swarm_config["max_iterations"] = swarm.max_iterations
        if swarm.execution_timeout != 900.0:
            swarm_config["execution_timeout"] = swarm.execution_timeout
        if swarm.node_timeout != 300.0:
            swarm_config["node_timeout"] = swarm.node_timeout
        if swarm.repetitive_handoff_detection_window != 0:
            swarm_config["repetitive_handoff_detection_window"] = swarm.repetitive_handoff_detection_window
        if swarm.repetitive_handoff_min_unique_agents != 0:
            swarm_config["repetitive_handoff_min_unique_agents"] = swarm.repetitive_handoff_min_unique_agents

        # Serialize agents
        agents_config = []
        agent_loader = self._get_agent_config_loader()

        for _node_id, swarm_node in swarm.nodes.items():
            agent = swarm_node.executor

            # Create a temporary copy of the agent without swarm coordination tools
            # to avoid conflicts when the swarm is recreated
            temp_agent = self._create_clean_agent_copy(agent)

            agent_config = agent_loader.serialize_agent(temp_agent)
            agents_config.append(agent_config)

        swarm_config["agents"] = agents_config

        return {"swarm": swarm_config}

    def _create_clean_agent_copy(self, agent: Agent) -> Agent:
        """Create a copy of an agent without swarm coordination tools.

        Args:
            agent: Original agent with potentially injected swarm tools.

        Returns:
            Agent copy without swarm coordination tools.
        """
        # List of swarm coordination tool names to exclude
        swarm_tool_names = {"handoff_to_agent"}

        # Get the original tools (excluding swarm coordination tools)
        original_tools = []
        if hasattr(agent, "tool_registry") and agent.tool_registry:
            for tool_name, tool in agent.tool_registry.registry.items():
                if tool_name not in swarm_tool_names:
                    original_tools.append(tool)

        # Extract hooks as a list if they exist
        hooks_list = None
        if hasattr(agent, "hooks") and agent.hooks:
            # HookRegistry has a hooks attribute that contains the actual hooks
            if hasattr(agent.hooks, "hooks"):
                hooks_list = list(agent.hooks.hooks)

        # Create a new agent with the same configuration but without swarm tools
        clean_agent = Agent(
            model=agent.model,
            messages=agent.messages,
            tools=original_tools,  # type: ignore[arg-type]
            system_prompt=agent.system_prompt,
            callback_handler=agent.callback_handler,
            conversation_manager=agent.conversation_manager,
            record_direct_tool_call=agent.record_direct_tool_call,
            load_tools_from_directory=agent.load_tools_from_directory,
            trace_attributes=agent.trace_attributes,
            agent_id=agent.agent_id,
            name=agent.name,
            description=agent.description,
            state=agent.state,
            hooks=hooks_list,
            session_manager=getattr(agent, "_session_manager", None),
        )

        return clean_agent

    def load_agents(self, agents_config: List[Dict[str, Any]]) -> List[Agent]:
        """Load agents using AgentConfigLoader from YAML agent configurations.

        Args:
            agents_config: List of agent configuration dictionaries.

        Returns:
            List of Agent instances.
        """
        if not agents_config:
            raise ValueError("Agents configuration cannot be empty")

        agents = []
        agent_loader = self._get_agent_config_loader()

        for i, agent_config in enumerate(agents_config):
            if not isinstance(agent_config, dict):
                raise ValueError(f"Agent configuration at index {i} must be a dictionary")

            # Validate required fields
            if "name" not in agent_config:
                raise ValueError(f"Agent configuration at index {i} must include 'name' field")
            if "model" not in agent_config:
                raise ValueError(f"Agent configuration at index {i} must include 'model' field")

            agent_name = agent_config["name"]

            try:
                # Wrap the agent config in the required top-level 'agent' key
                wrapped_agent_config = {"agent": agent_config}
                agent = agent_loader.load_agent(wrapped_agent_config)
                agents.append(agent)
                logger.debug("agent_name=<%s> | loaded agent for swarm", agent_name)
            except Exception as e:
                logger.error("agent_name=<%s> | failed to load agent: %s", agent_name, e)
                raise ValueError(f"Failed to load agent '{agent_name}': {str(e)}") from e

        return agents

    def _extract_swarm_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract swarm-specific parameters from YAML configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            Dictionary containing swarm constructor parameters.
        """
        params = {}

        # Extract parameters with defaults matching Swarm constructor
        if "max_handoffs" in config:
            max_handoffs = config["max_handoffs"]
            if not isinstance(max_handoffs, int) or max_handoffs < 1:
                raise ValueError("max_handoffs must be a positive integer")
            params["max_handoffs"] = max_handoffs

        if "max_iterations" in config:
            max_iterations = config["max_iterations"]
            if not isinstance(max_iterations, int) or max_iterations < 1:
                raise ValueError("max_iterations must be a positive integer")
            params["max_iterations"] = max_iterations

        if "execution_timeout" in config:
            execution_timeout = config["execution_timeout"]
            if not isinstance(execution_timeout, (int, float)) or execution_timeout <= 0:
                raise ValueError("execution_timeout must be a positive number")
            params["execution_timeout"] = int(execution_timeout)

        if "node_timeout" in config:
            node_timeout = config["node_timeout"]
            if not isinstance(node_timeout, (int, float)) or node_timeout <= 0:
                raise ValueError("node_timeout must be a positive number")
            params["node_timeout"] = int(node_timeout)

        if "repetitive_handoff_detection_window" in config:
            window = config["repetitive_handoff_detection_window"]
            if not isinstance(window, int) or window < 0:
                raise ValueError("repetitive_handoff_detection_window must be a non-negative integer")
            params["repetitive_handoff_detection_window"] = window

        if "repetitive_handoff_min_unique_agents" in config:
            min_agents = config["repetitive_handoff_min_unique_agents"]
            if not isinstance(min_agents, int) or min_agents < 0:
                raise ValueError("repetitive_handoff_min_unique_agents must be a non-negative integer")
            params["repetitive_handoff_min_unique_agents"] = min_agents

        return params

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate YAML swarm configuration structure.

        Args:
            config: Configuration dictionary to validate.

        Raises:
            ValueError: If configuration is invalid.
        """
        if not isinstance(config, dict):
            raise ValueError(f"Swarm configuration must be a dictionary, got: {type(config)}")

        # Check for required fields
        if "agents" not in config:
            raise ValueError("Swarm configuration must include 'agents' field")

        agents_config = config["agents"]
        if not isinstance(agents_config, list):
            raise ValueError("'agents' field must be a list")

        if not agents_config:
            raise ValueError("'agents' list cannot be empty")

        # Validate each agent configuration is a dictionary
        for i, agent_config in enumerate(agents_config):
            if not isinstance(agent_config, dict):
                raise ValueError(f"Agent configuration at index {i} must be a dictionary")

        # Validate parameter types if present
        for param_name in ["max_handoffs", "max_iterations"]:
            if param_name in config:
                value = config[param_name]
                if not isinstance(value, int):
                    raise ValueError(f"{param_name} must be an integer, got: {type(value)}")

        for param_name in ["execution_timeout", "node_timeout"]:
            if param_name in config:
                value = config[param_name]
                if not isinstance(value, (int, float)):
                    raise ValueError(f"{param_name} must be a number, got: {type(value)}")
