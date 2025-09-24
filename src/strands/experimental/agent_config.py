# ABOUTME: Experimental agent configuration with toAgent() method for creating Agent instances
# ABOUTME: Extends core AgentConfig with experimental instantiation patterns using ToolPool
"""Experimental agent configuration with enhanced instantiation patterns."""

import json
import importlib

from .tool_pool import ToolPool

# File prefix for configuration file paths
FILE_PREFIX = "file://"

# Minimum viable list of tools to enable agent building
# This list is experimental and will be revisited as tools evolve
DEFAULT_TOOLS = ["file_read", "editor", "http_request", "shell", "use_agent"]


class AgentConfig:
    """Agent configuration with toAgent() method and ToolPool integration."""
    
    def __init__(self, config_source: str | dict[str, any], tool_pool: ToolPool | None = None, raise_exception_on_missing_tool: bool = True):
        """Initialize AgentConfig from file path or dictionary.
        
        Args:
            config_source: Path to JSON config file (must start with 'file://') or config dictionary
            tool_pool: Optional ToolPool to select tools from when 'tools' is specified in config
            raise_exception_on_missing_tool: If False, skip missing tools instead of raising ImportError
        """
        if isinstance(config_source, str):
            # Require file:// prefix for file paths
            if not config_source.startswith(FILE_PREFIX):
                raise ValueError(f"File paths must be prefixed with '{FILE_PREFIX}'")
            
            # Remove file:// prefix and load from file
            file_path = config_source.removeprefix(FILE_PREFIX)
            with open(file_path, 'r') as f:
                config_data = json.load(f)
        else:
            # Use dictionary directly
            config_data = config_source
        
        self.model = config_data.get('model')
        self.system_prompt = config_data.get('prompt')  # Only accept 'prompt' key
        self._raise_exception_on_missing_tool = raise_exception_on_missing_tool
        
        # Process tools configuration if provided
        config_tools = config_data.get('tools')
        if config_tools is not None and tool_pool is None:
            raise ValueError("Tool names specified in config but no ToolPool provided")
        
        # Handle tool selection from ToolPool
        if tool_pool is not None:
            self._tool_pool = tool_pool
        else:
            # Create default ToolPool with strands_tools
            self._tool_pool = self._create_default_tool_pool()
        
        # Track configured tools separately from full tool pool
        self._configured_tools = []
            
        # Apply tool selection if specified
        if config_tools is not None:
            # Validate all tool names exist in the ToolPool
            available_tools = self._tool_pool.list_tool_names()
            for tool_name in config_tools:
                if tool_name not in available_tools:
                    if self._raise_exception_on_missing_tool:
                        raise ValueError(f"Tool '{tool_name}' not found in ToolPool. Available tools: {available_tools}")
                    # Skip missing tools when flag is False
                    continue
            
            # Store selected tools from the ToolPool (only ones that exist)
            all_tools = self._tool_pool.get_tools()
            for tool in all_tools:
                if tool.tool_name in config_tools:
                    self._configured_tools.append(tool)
        # If no tools specified in config, use no tools (empty list)
    
    def _create_default_tool_pool(self) -> ToolPool:
        """Create default ToolPool with strands_tools."""
        pool = ToolPool()
        
        for tool in DEFAULT_TOOLS:
            try:
                module_name = f"strands_tools.{tool}"
                tool_module = importlib.import_module(module_name)
                pool.add_tools_from_module(tool_module)
            except ImportError:
                if self._raise_exception_on_missing_tool:
                    raise ImportError(
                        f"strands_tools is not available and no ToolPool was specified. "
                        f"Either install strands_tools with 'pip install strands-agents-tools' "
                        f"or provide your own ToolPool with your own tools."
                    )
                # Skip missing tools when flag is False
                continue
        
        return pool
    
    @property
    def tool_pool(self) -> ToolPool:
        """Get the full ToolPool (superset of all available tools).
        
        Returns:
            ToolPool instance containing all available tools
        """
        return self._tool_pool
    
    @property
    def configured_tools(self) -> list:
        """Get the configured tools (subset selected for this agent).
        
        Returns:
            List of tools configured for this agent
        """
        return self._configured_tools
    
    def to_agent(self, **kwargs: any):
        """Create an Agent instance from this configuration.
        
        Args:
            **kwargs: Additional parameters to override config values.
                     Supports all Agent constructor parameters.
            
        Returns:
            Configured Agent instance
            
        Example:
            # Using default tools from strands_tools
            config = AgentConfig({
                "model": "anthropic.claude-3-5-sonnet-20241022-v2:0", 
                "prompt": "You are a helpful assistant",
                "tools": ["file_read"]
            })
            agent = config.to_agent()
            response = agent("Read the contents of README.md")
            
            # Using custom ToolPool
            from strands import tool
            
            @tool
            def custom_tool(input: str) -> str:
                return f"Custom: {input}"
                
            custom_pool = ToolPool([custom_tool])
            config = AgentConfig({
                "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
                "prompt": "You are a custom assistant",
                "tools": ["custom_tool"]
            }, tool_pool=custom_pool)
            agent = config.to_agent()
        """
        # Import here to avoid circular imports:
        # experimental/agent_config.py -> agent.agent -> event_loop.event_loop -> 
        # experimental.hooks -> experimental.__init__.py -> AgentConfig
        from ..agent.agent import Agent
        
        # Start with config values
        agent_params = {}
        
        if self.model is not None:
            agent_params['model'] = self.model
        if self.system_prompt is not None:
            agent_params['system_prompt'] = self.system_prompt
            
        # Use configured tools (subset of tool pool)
        agent_params['tools'] = self._configured_tools
        
        # Override with any other provided kwargs
        agent_params.update(kwargs)
        
        return Agent(**agent_params)
