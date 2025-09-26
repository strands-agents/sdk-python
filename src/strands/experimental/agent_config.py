"""Experimental agent configuration utilities.

This module provides utilities for creating agents from configuration files or dictionaries.
"""

import json
from pathlib import Path

from ..agent import Agent


def config_to_agent(config: str | dict[str, any], **kwargs) -> Agent:
    """Create an Agent from a configuration file or dictionary.
    
    Args:
        config: Either a file path (with optional file:// prefix) or a configuration dictionary
        **kwargs: Additional keyword arguments to pass to the Agent constructor
        
    Returns:
        Agent: A configured Agent instance
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        json.JSONDecodeError: If the configuration file contains invalid JSON
        ValueError: If the configuration is invalid
        
    Examples:
        Create agent from file:
        >>> agent = config_to_agent("/path/to/config.json")
        
        Create agent from file with file:// prefix:
        >>> agent = config_to_agent("file:///path/to/config.json")
        
        Create agent from dictionary:
        >>> config = {"model": "anthropic.claude-3-5-sonnet-20241022-v2:0", "tools": ["calculator"]}
        >>> agent = config_to_agent(config)
    """
    # Parse configuration
    if isinstance(config, str):
        # Handle file path
        file_path = config
        
        # Remove file:// prefix if present
        if file_path.startswith("file://"):
            file_path = file_path[7:]
            
        # Load JSON from file
        config_path = Path(file_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
            
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    elif isinstance(config, dict):
        config_dict = config.copy()
    else:
        raise ValueError("Config must be a file path string or dictionary")
    
    # Prepare Agent constructor arguments
    agent_kwargs = {}
    
    # Map configuration keys to Agent constructor parameters
    config_mapping = {
        "model": "model",
        "prompt": "system_prompt", 
        "tools": "tools",
        "name": "name",
    }
    
    # Only include non-None values from config
    for config_key, agent_param in config_mapping.items():
        if config_key in config_dict and config_dict[config_key] is not None:
            agent_kwargs[agent_param] = config_dict[config_key]
    
    # Override with any additional kwargs provided
    agent_kwargs.update(kwargs)
    
    # Create and return Agent
    return Agent(**agent_kwargs)
