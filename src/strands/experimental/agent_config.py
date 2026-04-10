"""Experimental agent configuration utilities.

This module provides utilities for creating agents from configuration files or dictionaries.

Note: Configuration-based agent setup only works for tools that don't require code-based
instantiation. For tools that need constructor arguments or complex setup, use the
programmatic approach after creating the agent:

    agent = config_to_agent("config.json")
    # Add tools that need code-based instantiation
    agent.tool_registry.process_tools([ToolWithConfigArg(HttpsConnection("localhost"))])

The ``model`` field supports two formats:

**String format (backward compatible — defaults to Bedrock):**
    {"model": "us.anthropic.claude-sonnet-4-20250514-v1:0"}

**Object format (supports all providers):**
    {
        "model": {
            "provider": "anthropic",
            "model_id": "claude-sonnet-4-20250514",
            "max_tokens": 10000,
            "client_args": {"api_key": "$ANTHROPIC_API_KEY"}
        }
    }

Environment variable references (``$VAR`` or ``${VAR}``) in model config values are resolved
automatically before provider instantiation.

Note: The following constructor parameters cannot be specified from JSON because they require
code-based instantiation: ``boto_session`` (Bedrock, SageMaker), ``client`` (OpenAI, Gemini),
``gemini_tools`` (Gemini). Use ``region_name`` / ``client_args`` as JSON-friendly alternatives.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import ValidationError

# JSON Schema for agent configuration
AGENT_CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Agent Configuration",
    "description": "Configuration schema for creating agents",
    "type": "object",
    "properties": {
        "name": {"description": "Name of the agent", "type": ["string", "null"], "default": None},
        "model": {
            "description": (
                "The model to use for this agent. Can be a string (Bedrock model_id) "
                "or an object with a 'provider' field for any supported provider."
            ),
            "oneOf": [
                {"type": "string"},
                {"type": "null"},
                {
                    "type": "object",
                    "properties": {
                        "provider": {
                            "description": "The model provider name",
                            "type": "string",
                        }
                    },
                    "required": ["provider"],
                    "additionalProperties": True,
                },
            ],
            "default": None,
        },
        "prompt": {
            "description": "The system prompt for the agent. Provides high level context to the agent.",
            "type": ["string", "null"],
            "default": None,
        },
        "tools": {
            "description": "List of tools the agent can use. Can be file paths, "
            "Python module names, or @tool annotated functions in files.",
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        },
    },
    "additionalProperties": False,
}

# Pre-compile validator for better performance
_VALIDATOR = jsonschema.Draft7Validator(AGENT_CONFIG_SCHEMA)

# Pattern for matching environment variable references
_ENV_VAR_PATTERN = re.compile(r"^\$\{([^}]+)\}$|^\$([A-Za-z_][A-Za-z0-9_]*)$")

# Provider name to model class name — resolved via strands.models lazy __getattr__
PROVIDER_MAP: dict[str, str] = {
    "bedrock": "BedrockModel",
    "anthropic": "AnthropicModel",
    "openai": "OpenAIModel",
    "gemini": "GeminiModel",
    "ollama": "OllamaModel",
    "litellm": "LiteLLMModel",
    "mistral": "MistralModel",
    "llamaapi": "LlamaAPIModel",
    "llamacpp": "LlamaCppModel",
    "sagemaker": "SageMakerAIModel",
    "writer": "WriterModel",
    "openai_responses": "OpenAIResponsesModel",
}


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve environment variable references in config values.

    String values matching ``$VAR_NAME`` or ``${VAR_NAME}`` are replaced with the
    corresponding environment variable value. Dicts and lists are traversed recursively.

    Args:
        value: The value to resolve. Can be a string, dict, list, or any other type.

    Returns:
        The resolved value with environment variable references replaced.

    Raises:
        ValueError: If a referenced environment variable is not set.
    """
    if isinstance(value, str):
        match = _ENV_VAR_PATTERN.match(value)
        if match:
            var_name = match.group(1) or match.group(2)
            env_value = os.environ.get(var_name)
            if env_value is None:
                raise ValueError(f"Environment variable '{var_name}' is not set")
            return env_value
        return value
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def _create_model_from_dict(model_config: dict[str, Any]) -> Any:
    """Create a Model instance from a provider config dict.

    Routes the config to the appropriate model class based on the ``provider`` field,
    then delegates to the class's ``from_dict`` method. All imports are lazy to avoid
    requiring optional dependencies that are not installed.

    Args:
        model_config: Dict containing at least a ``provider`` key and provider-specific params.

    Returns:
        A configured Model instance for the specified provider.

    Raises:
        ValueError: If the provider name is not recognized.
        ImportError: If the provider's optional dependencies are not installed.
    """
    config = model_config.copy()
    provider = config.pop("provider")

    class_name = PROVIDER_MAP.get(provider)
    if class_name is None:
        supported = ", ".join(sorted(PROVIDER_MAP.keys()))
        raise ValueError(f"Unknown model provider: '{provider}'. Supported providers: {supported}")

    from .. import models

    model_cls = getattr(models, class_name)
    return model_cls.from_dict(config)


def config_to_agent(config: str | dict[str, Any], **kwargs: dict[str, Any]) -> Any:
    """Create an Agent from a configuration file or dictionary.

    This function supports tools that can be loaded declaratively (file paths, module names,
    or @tool annotated functions). For tools requiring code-based instantiation with constructor
    arguments, add them programmatically after creating the agent:

        agent = config_to_agent("config.json")
        agent.process_tools([ToolWithConfigArg(HttpsConnection("localhost"))])

    Args:
        config: Either a file path (with optional file:// prefix) or a configuration dictionary
        **kwargs: Additional keyword arguments to pass to the Agent constructor

    Returns:
        Agent: A configured Agent instance

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        json.JSONDecodeError: If the configuration file contains invalid JSON
        ValueError: If the configuration is invalid or tools cannot be loaded

    Examples:
        Create agent from file:
        >>> agent = config_to_agent("/path/to/config.json")

        Create agent from file with file:// prefix:
        >>> agent = config_to_agent("file:///path/to/config.json")

        Create agent from dictionary:
        >>> config = {"model": "anthropic.claude-3-5-sonnet-20241022-v2:0", "tools": ["calculator"]}
        >>> agent = config_to_agent(config)

        Create agent with object model config:
        >>> config = {
        ...     "model": {"provider": "openai", "model_id": "gpt-4o", "client_args": {"api_key": "$OPENAI_API_KEY"}}
        ... }
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

        with open(config_path) as f:
            config_dict = json.load(f)
    elif isinstance(config, dict):
        config_dict = config.copy()
    else:
        raise ValueError("Config must be a file path string or dictionary")

    # Validate configuration against schema
    try:
        _VALIDATOR.validate(config_dict)
    except ValidationError as e:
        # Provide more detailed error message
        error_path = " -> ".join(str(p) for p in e.absolute_path) if e.absolute_path else "root"
        raise ValueError(f"Configuration validation error at {error_path}: {e.message}") from e

    # Prepare Agent constructor arguments
    agent_kwargs: dict[str, Any] = {}

    # Handle model field — string vs object format
    model_value = config_dict.get("model")
    if isinstance(model_value, dict):
        # Object format: resolve env vars and create Model instance via factory
        resolved_config = _resolve_env_vars(model_value)
        agent_kwargs["model"] = _create_model_from_dict(resolved_config)
    elif model_value is not None:
        # String format (backward compat): pass directly as model_id to Agent
        agent_kwargs["model"] = model_value

    # Map remaining configuration keys to Agent constructor parameters
    config_mapping = {
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

    # Import Agent at runtime to avoid circular imports
    from ..agent import Agent

    # Create and return Agent
    return Agent(**agent_kwargs)
