"""Agent configuration loader for Strands Agents.

This module provides the AgentConfigLoader class that enables creating Agent instances
from dictionary configurations, supporting serialization and deserialization of Agent
configurations for persistence and dynamic loading scenarios.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from ..tools.tool_config_loader import ToolConfigLoader

from pydantic import BaseModel

from strands.agent.agent import Agent
from strands.agent.conversation_manager import ConversationManager, SlidingWindowConversationManager
from strands.agent.state import AgentState
from strands.handlers.callback_handler import PrintingCallbackHandler, null_callback_handler
from strands.hooks import HookProvider
from strands.models.bedrock import BedrockModel
from strands.models.model import Model
from strands.session.session_manager import SessionManager
from strands.types.content import Messages

from .schema_registry import SchemaRegistry

logger = logging.getLogger(__name__)


class AgentConfigLoader:
    """Loads and serializes Strands Agent instances via dictionary configurations.

    This class provides functionality to create Agent instances from dictionary
    configurations and serialize existing Agent instances to dictionaries for
    persistence and configuration management.

    The loader supports:
    1. Loading agents from dictionary configurations
    2. Serializing agents to dictionary configurations
    3. Tool loading via ToolConfigLoader integration
    4. Model configuration and instantiation
    5. State and session management
    6. Structured output schema configuration and management
    """

    def __init__(self, tool_config_loader: Optional["ToolConfigLoader"] = None):
        """Initialize the AgentConfigLoader.

        Args:
            tool_config_loader: Optional ToolConfigLoader instance for loading tools.
                               If not provided, will be imported and created when needed.
        """
        self._tool_config_loader = tool_config_loader
        self.schema_registry = SchemaRegistry()
        self._global_schemas_loaded = False
        self._structured_output_defaults: Dict[str, Any] = {}

    def _get_tool_config_loader(self) -> "ToolConfigLoader":
        """Get or create a ToolConfigLoader instance.

        This method implements lazy loading to avoid circular imports.

        Returns:
            ToolConfigLoader instance.
        """
        if self._tool_config_loader is None:
            # Import here to avoid circular imports
            from ..tools.tool_config_loader import ToolConfigLoader

            self._tool_config_loader = ToolConfigLoader()
        return self._tool_config_loader

    def load_agent(self, config: Dict[str, Any]) -> Agent:
        """Load an Agent from a dictionary configuration.

        Args:
            config: Dictionary containing agent configuration with top-level 'agent' key.

        Returns:
            Agent instance configured according to the provided dictionary.

        Raises:
            ValueError: If required configuration is missing or invalid.
            ImportError: If specified models or tools cannot be imported.
        """
        # Validate top-level structure
        if "agent" not in config:
            raise ValueError("Configuration must contain a top-level 'agent' key")

        agent_config = config["agent"]
        if not isinstance(agent_config, dict):
            raise ValueError("The 'agent' configuration must be a dictionary")

        # Load global schemas if present and not already loaded
        if not self._global_schemas_loaded and "schemas" in config:
            self._load_global_schemas(config["schemas"])
            self._global_schemas_loaded = True

        # Load structured output defaults if present
        if "structured_output_defaults" in config:
            self._structured_output_defaults = config["structured_output_defaults"]

        # Extract configuration values from agent_config
        model_config = agent_config.get("model")
        system_prompt = agent_config.get("system_prompt")
        tools_config = agent_config.get("tools", [])
        messages_config = agent_config.get("messages", [])

        # Note: 'prompt' field is handled by AgentAsToolWrapper, not by Agent itself
        # The Agent class doesn't have a prompt parameter - it uses system_prompt
        # The prompt field is used for tool invocation templates

        # Agent metadata
        agent_id = agent_config.get("agent_id")
        name = agent_config.get("name")
        description = agent_config.get("description")

        # Advanced configuration
        callback_handler_config = agent_config.get("callback_handler")
        conversation_manager_config = agent_config.get("conversation_manager")
        record_direct_tool_call = agent_config.get("record_direct_tool_call", True)
        load_tools_from_directory = agent_config.get("load_tools_from_directory", False)
        trace_attributes = agent_config.get("trace_attributes")
        state_config = agent_config.get("state")
        hooks_config = agent_config.get("hooks", [])
        session_manager_config = agent_config.get("session_manager")

        # Load model
        model = self._load_model(model_config)

        # Load tools
        tools = self._load_tools(tools_config)

        # Load messages
        messages = self._load_messages(messages_config)

        # Load callback handler
        callback_handler = self._load_callback_handler(callback_handler_config)

        # Load conversation manager
        conversation_manager = self._load_conversation_manager(conversation_manager_config)

        # Load state
        state = self._load_state(state_config)

        # Load hooks
        hooks = self._load_hooks(hooks_config)

        # Load session manager
        session_manager = self._load_session_manager(session_manager_config)

        # Create agent
        agent = Agent(
            model=model,
            messages=messages,
            tools=tools,
            system_prompt=system_prompt,
            callback_handler=callback_handler,
            conversation_manager=conversation_manager,
            record_direct_tool_call=record_direct_tool_call,
            load_tools_from_directory=load_tools_from_directory,
            trace_attributes=trace_attributes,
            agent_id=agent_id,
            name=name,
            description=description,
            state=state,
            hooks=hooks,
            session_manager=session_manager,
        )

        # Configure structured output if specified
        if "structured_output" in agent_config:
            self._configure_agent_structured_output(agent, agent_config["structured_output"])

        return agent

    def serialize_agent(self, agent: Agent) -> Dict[str, Any]:
        """Serialize an Agent instance to a dictionary configuration.

        Args:
            agent: Agent instance to serialize.

        Returns:
            Dictionary containing the agent's configuration with top-level 'agent' key.

        Note:
            The 'prompt' field is not serialized here as it's specific to AgentAsToolWrapper
            and not part of the core Agent configuration.
        """
        agent_config = {}

        # Basic configuration
        if hasattr(agent.model, "model_id"):
            agent_config["model"] = agent.model.model_id
        elif hasattr(agent.model, "config") and agent.model.config.get("model_id"):
            agent_config["model"] = agent.model.config["model_id"]

        if agent.system_prompt:
            agent_config["system_prompt"] = agent.system_prompt

        # Tools configuration
        if hasattr(agent, "tool_registry") and agent.tool_registry:
            tools_config = []
            for tool_name in agent.tool_names:
                tools_config.append({"name": tool_name})
            if tools_config:
                agent_config["tools"] = tools_config

        # Messages
        if agent.messages:
            agent_config["messages"] = agent.messages

        # Agent metadata
        if agent.agent_id != "default":
            agent_config["agent_id"] = agent.agent_id
        if agent.name != "Strands Agents":
            agent_config["name"] = agent.name
        if agent.description:
            agent_config["description"] = agent.description

        # Advanced configuration
        if agent.record_direct_tool_call is not True:
            agent_config["record_direct_tool_call"] = agent.record_direct_tool_call
        if agent.load_tools_from_directory is not False:
            agent_config["load_tools_from_directory"] = agent.load_tools_from_directory
        if agent.trace_attributes:
            agent_config["trace_attributes"] = agent.trace_attributes

        # State
        if agent.state and agent.state.get():
            agent_config["state"] = agent.state.get()

        return {"agent": agent_config}

    def _load_model(self, model_config: Optional[Union[str, Dict[str, Any]]]) -> Optional[Model]:
        """Load a model from configuration.

        Args:
            model_config: Model configuration (string model ID or dict).

        Returns:
            Model instance or None.
        """
        if model_config is None:
            return None

        if isinstance(model_config, str):
            return BedrockModel(model_id=model_config)

        if isinstance(model_config, dict):
            model_type = model_config.get("type", "bedrock")
            if model_type == "bedrock":
                model_id = model_config.get("model_id")
                if not model_id:
                    raise ValueError("model_id is required for bedrock model")
                return BedrockModel(
                    model_id=model_id,
                    temperature=model_config.get("temperature"),
                    max_tokens=model_config.get("max_tokens"),
                    streaming=model_config.get("streaming", True),
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        raise ValueError(f"Invalid model configuration: {model_config}")

    def _load_tools(self, tools_config: List[Union[str, Dict[str, Any]]]) -> Optional[List[Any]]:
        """Load tools from configuration.

        Args:
            tools_config: List of tool configurations. Each item can be:
                         - String: Tool identifier for lookup
                         - Dict: Either tool lookup config or multi-agent tool config
                           - Tool lookup: {"name": "tool_name", "module": "optional_module"}
                           - Agent-as-tool: {"name": "tool_name", "agent": {...}}
                           - Graph-as-tool: {"name": "tool_name", "graph": {...}}
                           - Swarm-as-tool: {"name": "tool_name", "swarm": {...}}

        Returns:
            List of loaded tools or None.
        """
        if not tools_config:
            return None

        tools = []
        tool_loader = self._get_tool_config_loader()

        for tool_config in tools_config:
            if isinstance(tool_config, str):
                # Simple string identifier - load existing tool
                tool = tool_loader.load_tool(tool_config)
                tools.append(tool)
            elif isinstance(tool_config, dict):
                # Dictionary configuration
                if "agent" in tool_config or "graph" in tool_config or "swarm" in tool_config:
                    # Multi-agent tool configuration (agent-as-tool, graph-as-tool, swarm-as-tool)
                    # Pass entire dict to tool loader for auto-detection and loading
                    tool = tool_loader.load_tool(tool_config)
                    tools.append(tool)
                else:
                    # Traditional tool lookup configuration with name and optional module
                    name = tool_config.get("name")
                    module = tool_config.get("module")
                    if not name:
                        raise ValueError("Tool configuration must include 'name' field")
                    tool = tool_loader.load_tool(name, module)
                    tools.append(tool)
            else:
                raise ValueError(f"Invalid tool configuration: {tool_config}")

        return tools

    def _load_messages(self, messages_config: Optional[List[Dict[str, Any]]]) -> Optional[Messages]:
        """Load messages from configuration.

        Args:
            messages_config: List of message configurations.

        Returns:
            Messages list or None.
        """
        if not messages_config:
            return None

        # For now, return the messages as-is
        # In a full implementation, you might want to validate and transform them
        return messages_config  # type: ignore[return-value]

    def _load_callback_handler(self, callback_config: Optional[Union[str, Dict[str, Any]]]) -> Optional[Any]:
        """Load callback handler from configuration.

        Args:
            callback_config: Callback handler configuration.

        Returns:
            Callback handler instance or None.
        """
        if callback_config is None:
            return None

        if callback_config == "null":
            return null_callback_handler
        elif callback_config == "printing" or callback_config == "default":
            return PrintingCallbackHandler()
        elif isinstance(callback_config, dict):
            handler_type = callback_config.get("type", "printing")
            if handler_type == "printing":
                return PrintingCallbackHandler()
            elif handler_type == "null":
                return null_callback_handler
            else:
                raise ValueError(f"Unsupported callback handler type: {handler_type}")

        raise ValueError(f"Invalid callback handler configuration: {callback_config}")

    def _load_conversation_manager(self, cm_config: Optional[Dict[str, Any]]) -> Optional[ConversationManager]:
        """Load conversation manager from configuration.

        Args:
            cm_config: Conversation manager configuration.

        Returns:
            ConversationManager instance or None.
        """
        if cm_config is None:
            return None

        cm_type = cm_config.get("type", "sliding_window")
        if cm_type == "sliding_window":
            return SlidingWindowConversationManager(
                window_size=cm_config.get("window_size", 40),
                should_truncate_results=cm_config.get("should_truncate_results", True),
            )
        else:
            raise ValueError(f"Unsupported conversation manager type: {cm_type}")

    def _load_state(self, state_config: Optional[Dict[str, Any]]) -> Optional[AgentState]:
        """Load agent state from configuration.

        Args:
            state_config: State configuration dictionary.

        Returns:
            AgentState instance or None.
        """
        if state_config is None:
            return None

        return AgentState(initial_state=state_config)

    def _load_hooks(self, hooks_config: List[Dict[str, Any]]) -> Optional[List[HookProvider]]:
        """Load hooks from configuration.

        Args:
            hooks_config: List of hook configurations.

        Returns:
            List of HookProvider instances or None.
        """
        if not hooks_config:
            return None

        # For now, return None as hook loading would require more complex implementation
        # In a full implementation, you would dynamically load and instantiate hook providers
        logger.warning("Hook loading from configuration is not yet implemented")
        return None

    def _load_session_manager(self, sm_config: Optional[Dict[str, Any]]) -> Optional[SessionManager]:
        """Load session manager from configuration.

        Args:
            sm_config: Session manager configuration.

        Returns:
            SessionManager instance or None.
        """
        if sm_config is None:
            return None

        # For now, return None as session manager loading would require more complex implementation
        # In a full implementation, you would dynamically load and instantiate session managers
        logger.warning("Session manager loading from configuration is not yet implemented")
        return None

    def _load_global_schemas(self, schemas_config: List[Dict[str, Any]]) -> None:
        """Load global schema registry from configuration.

        Args:
            schemas_config: List of schema configurations
        """
        for schema_config in schemas_config:
            try:
                self.schema_registry.register_from_config(schema_config)
                logger.debug("Loaded global schema: %s", schema_config.get("name"))
            except Exception as e:
                logger.error("Failed to load schema %s: %s", schema_config.get("name", "unknown"), e)
                raise

    def _configure_agent_structured_output(self, agent: Agent, structured_config: Union[str, Dict[str, Any]]) -> None:
        """Configure structured output for an agent.

        Args:
            agent: Agent instance to configure
            structured_config: Structured output configuration (string reference or dict)
        """
        try:
            # Case 1: Simple string reference
            if isinstance(structured_config, str):
                schema_class = self.schema_registry.resolve_schema_reference(structured_config)
                self._attach_structured_output_to_agent(agent, schema_class)

            # Case 2: Detailed configuration
            elif isinstance(structured_config, dict):
                schema_ref = structured_config.get("schema")
                if not schema_ref:
                    raise ValueError("Structured output configuration must specify 'schema'")

                schema_class = self.schema_registry.resolve_schema_reference(schema_ref)
                validation_config = structured_config.get("validation", {})
                error_config = structured_config.get("error_handling", {})

                # Merge with defaults
                merged_validation = {**self._structured_output_defaults.get("validation", {}), **validation_config}
                merged_error_handling = {**self._structured_output_defaults.get("error_handling", {}), **error_config}

                self._attach_structured_output_to_agent(agent, schema_class, merged_validation, merged_error_handling)

            else:
                raise ValueError("structured_output must be a string reference or configuration dict")

            logger.debug("Configured structured output for agent %s", agent.name)

        except Exception as e:
            logger.error("Failed to configure structured output for agent %s: %s", agent.name, e)
            raise

    def _attach_structured_output_to_agent(
        self,
        agent: Agent,
        schema_class: type[BaseModel],
        validation_config: Optional[Dict[str, Any]] = None,
        error_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Attach structured output configuration to an agent.

        Args:
            agent: Agent instance
            schema_class: Pydantic model class for structured output
            validation_config: Validation configuration options
            error_config: Error handling configuration options
        """
        # Store the schema class and configuration on the agent
        agent._structured_output_schema = schema_class  # type: ignore[attr-defined]
        agent._structured_output_validation = validation_config or {}  # type: ignore[attr-defined]
        agent._structured_output_error_handling = error_config or {}  # type: ignore[attr-defined]

        # Store original methods for potential future use
        agent._original_structured_output = agent.structured_output  # type: ignore[attr-defined]
        agent._original_structured_output_async = agent.structured_output_async  # type: ignore[attr-defined]

        # Add a new configured structured output method
        def configured_structured_output(prompt: Optional[Union[str, list]] = None) -> Any:
            """Structured output using the configured schema."""
            return agent._original_structured_output(schema_class, prompt)  # type: ignore[attr-defined]

        # Replace the structured_output method to use configured schema by default
        def new_structured_output(output_model_or_prompt: Any = None, prompt: Any = None) -> Any:
            """Enhanced structured output that can use configured schema or explicit model."""
            # If called with two arguments (original API: output_model, prompt)
            if prompt is not None:
                return agent._original_structured_output(output_model_or_prompt, prompt)  # type: ignore[attr-defined]
            # If called with one argument that's a type (original API: output_model only)
            elif hasattr(output_model_or_prompt, "__bases__") and issubclass(output_model_or_prompt, BaseModel):
                return agent._original_structured_output(output_model_or_prompt, None)  # type: ignore[attr-defined]
            # If called with one argument that's a string/list or None (new API: prompt only)
            else:
                return agent._original_structured_output(schema_class, output_model_or_prompt)  # type: ignore[attr-defined]

        # Replace the method
        agent.structured_output = new_structured_output  # type: ignore[assignment]

        # Add convenience method with schema name
        schema_name = schema_class.__name__.lower()
        method_name = f"extract_{schema_name}"
        setattr(agent, method_name, configured_structured_output)

        logger.debug("Attached structured output schema %s to agent", schema_class.__name__)

    def get_schema_registry(self) -> SchemaRegistry:
        """Get the schema registry instance.

        Returns:
            SchemaRegistry instance
        """
        return self.schema_registry

    def list_schemas(self) -> Dict[str, str]:
        """List all registered schemas.

        Returns:
            Dictionary mapping schema names to their types
        """
        return self.schema_registry.list_schemas()
