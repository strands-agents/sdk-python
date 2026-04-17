"""MCP server configuration loading and MCPClient factory.

This module handles parsing MCP server configurations from dictionaries or JSON files
and creating MCPClient instances with the appropriate transport. The configuration format
is compatible with the Claude Desktop / Cursor / VS Code mcpServers convention.

Supported transport types:
- stdio: Subprocess-based transport (default when ``command`` is present)
- sse: Server-Sent Events transport
- streamable-http: Streamable HTTP transport

Example configuration::

    {
        "aws_docs": {
            "command": "uvx",
            "args": ["awslabs.aws-documentation-mcp-server@latest"],
            "env": {"AWS_PROFILE": "default"},
            "prefix": "aws",
            "startup_timeout": 30,
            "tool_filters": {
                "allowed": ["search_.*"],
                "rejected": ["dangerous_tool"]
            }
        },
        "remote_service": {
            "transport": "sse",
            "url": "http://localhost:8000/sse",
            "headers": {"Authorization": "Bearer token"},
            "prefix": "remote"
        }
    }
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import ValidationError

from .mcp_client import MCPClient, ToolFilters

logger = logging.getLogger(__name__)

MCP_SERVER_CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "MCP Server Configuration",
    "description": "Configuration for one or more MCP servers",
    "type": "object",
    "additionalProperties": {
        "type": "object",
        "properties": {
            "command": {
                "description": "Command to run the MCP server (stdio transport)",
                "type": "string",
            },
            "args": {
                "description": "Arguments for the command",
                "type": "array",
                "items": {"type": "string"},
                "default": [],
            },
            "env": {
                "description": "Environment variables for the subprocess",
                "type": "object",
                "additionalProperties": {"type": "string"},
                "default": {},
            },
            "cwd": {
                "description": "Working directory for stdio subprocess",
                "type": ["string", "null"],
                "default": None,
            },
            "transport": {
                "description": "Transport type: stdio, sse, or streamable-http (auto-detected if omitted)",
                "type": "string",
                "enum": ["stdio", "sse", "streamable-http"],
            },
            "url": {
                "description": "URL of the remote MCP server (sse or streamable-http transport)",
                "type": "string",
            },
            "headers": {
                "description": "HTTP headers for sse/http transports",
                "type": "object",
                "additionalProperties": {"type": "string"},
                "default": {},
            },
            "prefix": {
                "description": "Prefix for tool names to avoid collisions",
                "type": ["string", "null"],
                "default": None,
            },
            "startup_timeout": {
                "description": "Timeout for server initialization in seconds",
                "type": "integer",
                "minimum": 1,
                "default": 30,
            },
            "tool_filters": {
                "description": "Tool filter config with allowed and rejected patterns",
                "type": "object",
                "properties": {
                    "allowed": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "rejected": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    },
}

_VALIDATOR = jsonschema.Draft7Validator(MCP_SERVER_CONFIG_SCHEMA)


def _detect_transport(server_config: dict[str, Any]) -> str:
    """Detect the transport type from server configuration.

    Detection logic:
    - If ``transport`` is explicitly set, use it
    - If ``command`` is present, default to ``stdio``
    - If ``url`` is present, default to ``streamable-http``
    - Otherwise raise ValueError

    Args:
        server_config: Configuration dictionary for a single MCP server.

    Returns:
        The transport type string.

    Raises:
        ValueError: If transport cannot be determined from the configuration.
    """
    if "transport" in server_config:
        return server_config["transport"]

    if "command" in server_config:
        return "stdio"

    if "url" in server_config:
        return "streamable-http"

    raise ValueError(
        "cannot determine transport type: provide 'transport', 'command' (for stdio), or 'url' (for sse/http)"
    )


def _parse_tool_filters(filters_config: dict[str, Any] | None) -> ToolFilters | None:
    """Parse tool filter configuration into ToolFilters.

    String patterns are compiled as regex patterns. Exact-match strings (no regex
    metacharacters) are compiled into anchored patterns for safety.

    Args:
        filters_config: Raw filter configuration with ``allowed`` and/or ``rejected`` lists.

    Returns:
        A ToolFilters dict or None if no filters are configured.
    """
    if not filters_config:
        return None

    tool_filters: ToolFilters = {}

    for key in ("allowed", "rejected"):
        if key in filters_config:
            compiled: list[str | re.Pattern[str]] = []
            for pattern in filters_config[key]:
                try:
                    compiled.append(re.compile(pattern))
                except re.error:
                    # If it fails to compile as regex, use as exact string match
                    logger.warning("pattern=<%s> | invalid regex, using as literal string match", pattern)
                    compiled.append(pattern)
            tool_filters[key] = compiled  # type: ignore[literal-required]

    return tool_filters if tool_filters else None


def _create_stdio_transport(server_config: dict[str, Any]) -> Any:
    """Create a stdio transport callable from configuration.

    Args:
        server_config: Server configuration containing ``command`` and optional ``args``, ``env``, ``cwd``.

    Returns:
        A callable that returns an MCPTransport context manager.

    Raises:
        ValueError: If ``command`` is not specified.
    """
    if "command" not in server_config:
        raise ValueError("'command' is required for stdio transport")

    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=server_config["command"],
        args=server_config.get("args", []),
        env=server_config.get("env") or None,
        cwd=server_config.get("cwd"),
    )

    def transport_callable() -> Any:
        return stdio_client(params)

    return transport_callable


def _create_sse_transport(server_config: dict[str, Any]) -> Any:
    """Create an SSE transport callable from configuration.

    Args:
        server_config: Server configuration containing ``url`` and optional ``headers``.

    Returns:
        A callable that returns an MCPTransport context manager.

    Raises:
        ValueError: If ``url`` is not specified.
    """
    if "url" not in server_config:
        raise ValueError("'url' is required for sse transport")

    from mcp.client.sse import sse_client

    url = server_config["url"]
    headers = server_config.get("headers", {})

    def transport_callable() -> Any:
        return sse_client(url=url, headers=headers)

    return transport_callable


def _create_streamable_http_transport(server_config: dict[str, Any]) -> Any:
    """Create a streamable-http transport callable from configuration.

    Args:
        server_config: Server configuration containing ``url`` and optional ``headers``.

    Returns:
        A callable that returns an MCPTransport context manager.

    Raises:
        ValueError: If ``url`` is not specified.
    """
    if "url" not in server_config:
        raise ValueError("'url' is required for streamable-http transport")

    from mcp.client.streamable_http import streamablehttp_client

    url = server_config["url"]
    headers = server_config.get("headers", {})

    def transport_callable() -> Any:
        return streamablehttp_client(url=url, headers=headers)

    return transport_callable


_TRANSPORT_FACTORIES = {
    "stdio": _create_stdio_transport,
    "sse": _create_sse_transport,
    "streamable-http": _create_streamable_http_transport,
}


def _create_mcp_client(server_name: str, server_config: dict[str, Any]) -> MCPClient:
    """Create an MCPClient from a single server configuration entry.

    Args:
        server_name: Name of the server (used in error messages).
        server_config: Configuration dictionary for the server.

    Returns:
        A configured MCPClient instance (not yet started).

    Raises:
        ValueError: If the configuration is invalid.
    """
    transport_type = _detect_transport(server_config)
    logger.debug("server=<%s>, transport=<%s> | creating MCP client", server_name, transport_type)

    factory = _TRANSPORT_FACTORIES.get(transport_type)
    if factory is None:
        raise ValueError(f"unsupported transport type: {transport_type}")

    transport_callable = factory(server_config)

    # Build MCPClient constructor kwargs
    client_kwargs: dict[str, Any] = {
        "transport_callable": transport_callable,
    }

    if "startup_timeout" in server_config:
        client_kwargs["startup_timeout"] = server_config["startup_timeout"]

    if "prefix" in server_config and server_config["prefix"] is not None:
        client_kwargs["prefix"] = server_config["prefix"]

    tool_filters = _parse_tool_filters(server_config.get("tool_filters"))
    if tool_filters is not None:
        client_kwargs["tool_filters"] = tool_filters

    return MCPClient(**client_kwargs)


def load_mcp_clients_from_config(
    config: str | dict[str, Any],
) -> dict[str, MCPClient]:
    """Load MCP clients from a configuration file or dictionary.

    Creates MCPClient instances for each server defined in the configuration.
    The clients are returned unstarted — they will be started automatically
    when used with an Agent (via the ToolProvider lifecycle) or can be started
    manually with ``client.start()``.

    Args:
        config: Either a file path to a JSON configuration file, or a dictionary
            mapping server names to their configuration. The file can contain
            either a top-level ``mcp_servers`` key or be a flat mapping of
            server names to configs.

    Returns:
        A dictionary mapping server names to MCPClient instances.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        ValueError: If the configuration is invalid.

    Examples:
        Load from file::

            clients = load_mcp_clients_from_config("mcp_config.json")

        Load from dictionary::

            clients = load_mcp_clients_from_config({
                "aws_docs": {
                    "command": "uvx",
                    "args": ["awslabs.aws-documentation-mcp-server@latest"]
                }
            })

        Use with agent::

            from strands import Agent
            clients = load_mcp_clients_from_config("mcp_config.json")
            agent = Agent(tools=list(clients.values()))
    """
    if isinstance(config, str):
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"MCP configuration file not found: {config}")
        with open(config_path) as f:
            config_dict = json.load(f)
    elif isinstance(config, dict):
        config_dict = config.copy()
    else:
        raise ValueError("config must be a file path string or dictionary")

    # Support both top-level mcp_servers key and flat dict
    if "mcp_servers" in config_dict and isinstance(config_dict["mcp_servers"], dict):
        servers_config = config_dict["mcp_servers"]
    else:
        servers_config = config_dict

    # Validate against schema
    try:
        _VALIDATOR.validate(servers_config)
    except ValidationError as e:
        error_path = " -> ".join(str(p) for p in e.absolute_path) if e.absolute_path else "root"
        raise ValueError(f"MCP configuration validation error at {error_path}: {e.message}") from e

    clients: dict[str, MCPClient] = {}
    for server_name, server_config in servers_config.items():
        try:
            clients[server_name] = _create_mcp_client(server_name, server_config)
            logger.debug("server=<%s> | MCP client created successfully", server_name)
        except Exception as e:
            raise ValueError(f"failed to create MCP client for server '{server_name}': {e}") from e

    logger.debug("total_clients=<%d> | all MCP clients created", len(clients))
    return clients
