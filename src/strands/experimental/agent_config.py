"""Experimental agent configuration with enhanced instantiation patterns."""

import json
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Import here to avoid circular imports:
    # experimental/agent_config.py -> agent.agent -> event_loop.event_loop -> 
    # experimental.hooks -> experimental.__init__.py -> AgentConfig
    from ..agent.agent import Agent

from .tool_box import ToolBox

# File prefix for configuration file paths
FILE_PREFIX = "file://"

# Minimum viable list of tools to enable agent building
# This list is experimental and will be revisited as tools evolve
DEFAULT_TOOLS = ["file_read", "editor", "http_request", "shell", "use_agent"]


class AgentConfig:
    """Agent configuration with to_agent() method and ToolBox integration.
    
    Example config.json:
    {
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "prompt": "You are a helpful assistant",
        "tools": ["file_read", "editor"]
    }
    """
    
    def __init__(self, config_source: str | dict[str, any], tool_box: ToolBox | None = None, raise_exception_on_missing_tool: bool = True):
        """Initialize AgentConfig from file path or dictionary.
        
        Args:
            config_source: Path to JSON config file (must start with 'file://') or config dictionary
            tool_box: Optional ToolBox to select tools from when 'tools' is specified in config
            raise_exception_on_missing_tool: If False, skip missing tools instead of raising ImportError
            
        Example:
            # Dictionary config
            config = AgentConfig({
                "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
                "prompt": "You are a helpful assistant",
                "tools": ["file_read", "editor"]
            })
            
            # File config
            config = AgentConfig("file://config.json")
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
        
        # Handle tool selection from ToolBox
        if tool_box is not None:
            self._toolbox = tool_box
        else:
            # Create default ToolBox with strands_tools
            self._toolbox = self._create_default_toolbox()
        
        # Process tools configuration if provided
        config_tools = config_data.get('tools')
        
        # Track configured tools separately from full tool pool
        self._configured_tools = []
            
        # Apply tool selection if specified
        if config_tools is not None:
            # Validate all tool names exist in the ToolBox
            available_tools = self._toolbox.list_tool_names()
            
            if any(tool_name not in available_tools for tool_name in config_tools) and self._raise_exception_on_missing_tool:
                missing_tool = next(tool_name for tool_name in config_tools if tool_name not in available_tools)
                raise ValueError(f"Tool '{missing_tool}' not found in ToolBox. Available tools: {available_tools}")
            
            # Store selected tools from the ToolBox (only ones that exist)
            all_tools = self._toolbox.list_tools()
            for tool in all_tools:
                if tool.tool_name in config_tools:
                    self._configured_tools.append(tool)
        # If no tools specified in config, use no tools (empty list)
    
    def _create_default_toolbox(self) -> ToolBox:
        """Create default ToolBox with strands_tools."""
        pool = ToolBox()
        
        for tool in DEFAULT_TOOLS:
            try:
                module_name = f"strands_tools.{tool}"
                tool_module = importlib.import_module(module_name)
                pool.add_tools_from_module(tool_module)
            except ImportError:
                if self._raise_exception_on_missing_tool:
                    raise ImportError(
                        f"strands_tools is not available and no ToolBox was specified. "
                        f"Either install strands_tools with 'pip install strands-agents-tools' "
                        f"or provide your own ToolBox with your own tools."
                    )
                # Skip missing tools when flag is False
                continue
        
        return pool
    
    @property
    def toolbox(self) -> ToolBox:
        """Get the full ToolBox (superset of all available tools).
        
        Returns:
            ToolBox instance containing all available tools
        """
        return self._toolbox
    
    @property
    def configured_tools(self) -> list:
        """Get the configured tools (subset selected for this agent).
        
        Returns:
            List of tools configured for this agent
        """
        return self._configured_tools
    
    def to_agent(self, **kwargs: any) -> "Agent":
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
            
            # Using custom ToolBox
            from strands import tool
            
            @tool
            def custom_tool(input: str) -> str:
                return f"Custom: {input}"
                
            custom_toolbox = ToolBox([custom_tool])
            config = AgentConfig({
                "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
                "prompt": "You are a custom assistant",
                "tools": ["custom_tool"]
            }, tool_box=custom_toolbox)
            agent = config.to_agent()
        """
        # Import at runtime since TYPE_CHECKING import is not available during execution
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
