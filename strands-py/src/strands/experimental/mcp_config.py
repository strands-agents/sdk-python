"""MCP server configuration parsing and MCPClient factory.

This module handles parsing MCP server configurations from dictionaries or JSON files
and creating MCPClient instances with the appropriate transport callables. It accepts the
ecosystem-standard ``mcpServers`` wrapper format used by Claude Desktop, Cursor, and VS Code,
so an existing config can be loaded directly.

Supported transport types:

- stdio: Local subprocess via stdin/stdout (auto-detected when 'command' is present)
- sse: Server-Sent Events over HTTP (auto-detected when 'url' is present without explicit transport)
- streamable-http: Streamable HTTP transport

Two ergonomic features keep secrets out of committed config files and make paths portable:

- ``${env:VAR}`` interpolation in string values is replaced with the value of the ``VAR``
  environment variable, so tokens and other secrets can stay in the environment.
- ``~`` in the ``command`` and ``cwd`` paths is expanded to the user's home directory.

Example::

    {
        "mcpServers": {
            "aws_docs": {
                "command": "~/bin/uvx",
                "args": ["awslabs.aws-documentation-mcp-server@latest"],
                "prefix": "aws"
            },
            "remote": {
                "url": "https://example.com/sse",
                "headers": {"Authorization": "Bearer ${env:MCP_TOKEN}"}
            },
            "legacy": {
                "command": "old-server",
                "disabled": true
            }
        }
    }
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import ValidationError
from mcp import StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from ..tools.mcp.mcp_client import MCPClient, ToolFilters

logger = logging.getLogger(__name__)

# Matches ${env:VAR_NAME} where VAR_NAME is a valid environment variable identifier.
_ENV_VAR_PATTERN = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}")

MCP_SERVER_CONFIG_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "MCP Server Configuration",
    "description": "Configuration for a single MCP server.",
    "type": "object",
    "properties": {
        "transport": {
            "description": "Transport type. Auto-detected from 'command' (stdio) or 'url' (sse) if omitted.",
            "type": "string",
            "enum": ["stdio", "sse", "streamable-http"],
        },
        "command": {"description": "Command to run for stdio transport.", "type": "string"},
        "args": {
            "description": "Arguments for the stdio command.",
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        },
        "env": {
            "description": "Environment variables for the stdio command.",
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "cwd": {"description": "Working directory for the stdio command.", "type": "string"},
        "url": {"description": "URL for sse or streamable-http transport.", "type": "string"},
        "headers": {
            "description": "HTTP headers for sse or streamable-http transport.",
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "prefix": {"description": "Prefix to apply to tool names from this server.", "type": "string"},
        "disabled": {
            "description": "When true, the server is skipped and no client is created for it.",
            "type": "boolean",
            "default": False,
        },
        "startup_timeout": {
            "description": "Timeout in seconds for server initialization. Defaults to 30.",
            "type": "integer",
            "default": 30,
        },
        "tool_filters": {
            "description": "Filters for controlling which tools are loaded.",
            "type": "object",
            "properties": {
                "allowed": {
                    "description": "List of regex patterns for tools to include.",
                    "type": "array",
                    "items": {"type": "string"},
                },
                "rejected": {
                    "description": "List of regex patterns for tools to exclude.",
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

_SERVER_VALIDATOR = jsonschema.Draft7Validator(MCP_SERVER_CONFIG_SCHEMA)


def _interpolate_env_vars(value: Any) -> Any:
    """Recursively replace ``${env:VAR}`` references in string values with environment variables.

    Strings, and the strings nested inside lists and dictionaries, are interpolated. Other
    value types are returned unchanged.

    Args:
        value: A configuration value (string, list, dict, or scalar) to interpolate.

    Returns:
        The value with all ``${env:VAR}`` references replaced by their environment values.

    Raises:
        ValueError: If a referenced environment variable is not set.
    """
    if isinstance(value, str):

        def _replace(match: re.Match[str]) -> str:
            var_name = match.group(1)
            if var_name not in os.environ:
                raise ValueError(f"environment variable '{var_name}' referenced by '${{env:{var_name}}}' is not set")
            return os.environ[var_name]

        return _ENV_VAR_PATTERN.sub(_replace, value)
    if isinstance(value, list):
        return [_interpolate_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: _interpolate_env_vars(item) for key, item in value.items()}
    return value


def _parse_tool_filters(config: dict[str, Any] | None) -> ToolFilters | None:
    """Parse a tool filter configuration into a ToolFilters instance.

    All filter strings are compiled as regex patterns and matched using ``re.match``
    (prefix match from start of string). Use ``"^echo$"`` for exact matching.
    ``"echo"`` will match any tool name starting with "echo" (e.g. "echo_extra").

    Args:
        config: Tool filter configuration dict with 'allowed' and/or 'rejected' lists,
            or None.

    Returns:
        A ToolFilters instance, or None if config is None or empty.

    Raises:
        ValueError: If a filter string is not a valid regex pattern.
    """
    if not config:
        return None

    result: ToolFilters = {}

    if "allowed" in config:
        allowed: list[Any] = []
        for pattern_str in config["allowed"]:
            try:
                allowed.append(re.compile(pattern_str))
            except re.error as e:
                raise ValueError(f"invalid regex pattern in tool_filters.allowed: '{pattern_str}': {e}") from e
        result["allowed"] = allowed

    if "rejected" in config:
        rejected: list[Any] = []
        for pattern_str in config["rejected"]:
            try:
                rejected.append(re.compile(pattern_str))
            except re.error as e:
                raise ValueError(f"invalid regex pattern in tool_filters.rejected: '{pattern_str}': {e}") from e
        result["rejected"] = rejected

    return result if result else None


def _create_mcp_client_from_config(server_name: str, config: dict[str, Any]) -> MCPClient:
    """Create an MCPClient instance from a server configuration dictionary.

    Before validation, ``${env:VAR}`` references in string values are replaced with the
    corresponding environment variables, and ``~`` in the ``command`` and ``cwd`` paths is
    expanded to the user's home directory.

    Transport type is auto-detected based on the presence of 'command' (stdio) or 'url' (sse),
    unless explicitly specified via the 'transport' field.

    Args:
        server_name: Name of the server (used in error messages).
        config: Server configuration dictionary.

    Returns:
        A configured MCPClient instance.

    Raises:
        ValueError: If the configuration is invalid, missing required fields, or references
            an environment variable that is not set.
    """
    # Resolve ${env:VAR} references before validation so secrets can live in the environment.
    config = _interpolate_env_vars(config)

    # Validate against schema
    try:
        _SERVER_VALIDATOR.validate(config)
    except ValidationError as e:
        error_path = " -> ".join(str(p) for p in e.absolute_path) if e.absolute_path else "root"
        raise ValueError(f"server '{server_name}' configuration validation error at {error_path}: {e.message}") from e

    # Determine transport type
    transport = config.get("transport")
    command = config.get("command")
    url = config.get("url")

    if transport is None:
        if command:
            transport = "stdio"
        elif url:
            transport = "sse"
        else:
            raise ValueError(
                f"server '{server_name}' must specify either 'command' (for stdio) or 'url' (for sse/http)"
            )

    # Extract common MCPClient parameters
    prefix = config.get("prefix")
    startup_timeout = config.get("startup_timeout", 30)
    tool_filters = _parse_tool_filters(config.get("tool_filters"))

    # Build transport callable based on type
    if transport == "stdio":
        if not command:
            raise ValueError(f"server '{server_name}': 'command' is required for stdio transport")
        # Expand ~ in filesystem paths so configs are portable across machines.
        command = os.path.expanduser(command)
        cwd = config.get("cwd")
        if cwd is not None:
            cwd = os.path.expanduser(cwd)
        args = config.get("args", [])
        env = config.get("env")

        def _stdio_transport() -> Any:
            params = StdioServerParameters(command=command, args=args, env=env, cwd=cwd)
            return stdio_client(params)

        transport_callable = _stdio_transport
    elif transport == "sse":
        if not url:
            raise ValueError(f"server '{server_name}': 'url' is required for sse transport")
        headers = config.get("headers")

        def _sse_transport() -> Any:
            return sse_client(url=url, headers=headers)

        transport_callable = _sse_transport
    elif transport == "streamable-http":
        if not url:
            raise ValueError(f"server '{server_name}': 'url' is required for streamable-http transport")
        headers = config.get("headers")

        def _streamable_http_transport() -> Any:
            return streamablehttp_client(url=url, headers=headers)

        transport_callable = _streamable_http_transport
    else:
        raise ValueError(f"server '{server_name}': unsupported transport type '{transport}'")

    logger.debug(
        "server_name=<%s>, transport=<%s> | creating MCP client from config",
        server_name,
        transport,
    )

    return MCPClient(
        transport_callable,
        startup_timeout=startup_timeout,
        tool_filters=tool_filters,
        prefix=prefix,
    )


def load_mcp_clients_from_config(config: str | dict[str, Any]) -> list[MCPClient]:
    """Load MCP client instances from a configuration file or dictionary.

    Expects the standard ``mcpServers`` wrapper format used by Claude Desktop, Cursor,
    VS Code, etc::

        {
            "mcpServers": {
                "server_name": { "command": "...", ... }
            }
        }

    String values support ``${env:VAR}`` interpolation (replaced with the value of the ``VAR``
    environment variable), and ``~`` in the ``command`` and ``cwd`` paths is expanded to the
    user's home directory. Servers marked ``"disabled": true`` are skipped.

    Args:
        config: Either a file path (with optional file:// prefix) to a JSON config file,
            or a dictionary with a ``mcpServers`` key mapping server names to configs.

    Returns:
        A list of MCPClient instances, one per enabled server. Pass it directly to an agent
        with ``Agent(tools=clients)``.

    Raises:
        FileNotFoundError: If the config file does not exist.
        json.JSONDecodeError: If the config file contains invalid JSON.
        ValueError: If the config format is invalid, a server config is not a dictionary or is
            invalid, or a referenced environment variable is not set.

    Examples:
        Load from a dictionary:
        >>> clients = load_mcp_clients_from_config({"mcpServers": {"echo": {"command": "echo"}}})

        Load from a file and build an agent:
        >>> clients = load_mcp_clients_from_config("mcp.json")
        >>> agent = Agent(tools=clients)
    """
    if isinstance(config, str):
        file_path = config
        if file_path.startswith("file://"):
            file_path = file_path[7:]

        config_path = Path(file_path)
        if not config_path.exists():
            raise FileNotFoundError(f"MCP configuration file not found: {file_path}")

        with open(config_path) as f:
            config_dict: dict[str, Any] = json.load(f)
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise ValueError("Config must be a file path string or dictionary")

    if "mcpServers" not in config_dict or not isinstance(config_dict["mcpServers"], dict):
        raise ValueError("Config must contain an 'mcpServers' key with a dictionary of server configurations")

    servers = config_dict["mcpServers"]
    clients: list[MCPClient] = []
    for server_name, server_config in servers.items():
        if not isinstance(server_config, dict):
            raise ValueError(f"server '{server_name}' configuration must be a dictionary")
        if server_config.get("disabled", False):
            logger.debug("server_name=<%s> | skipping disabled MCP server", server_name)
            continue
        clients.append(_create_mcp_client_from_config(server_name, server_config))

    logger.debug("loaded_servers=<%d> | MCP clients created from config", len(clients))

    return clients
