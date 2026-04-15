"""MCP server configuration parsing and MCPClient factory.

This module handles parsing MCP server configurations from dictionaries or JSON files
and creating MCPClient instances with the appropriate transport callables.

Supported transport types:
- stdio: Local subprocess via stdin/stdout (auto-detected when 'command' is present)
- sse: Server-Sent Events over HTTP (auto-detected when 'url' is present without explicit transport)
- streamable-http: Streamable HTTP transport
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import ValidationError
from mcp import StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from ..tools.mcp.mcp_client import MCPClient, ToolFilters

logger = logging.getLogger(__name__)

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
        allowed: list[re.Pattern[str]] = []
        for pattern_str in config["allowed"]:
            try:
                allowed.append(re.compile(pattern_str))
            except re.error as e:
                raise ValueError(f"invalid regex pattern in tool_filters.allowed: '{pattern_str}': {e}") from e
        result["allowed"] = allowed

    if "rejected" in config:
        rejected: list[re.Pattern[str]] = []
        for pattern_str in config["rejected"]:
            try:
                rejected.append(re.compile(pattern_str))
            except re.error as e:
                raise ValueError(f"invalid regex pattern in tool_filters.rejected: '{pattern_str}': {e}") from e
        result["rejected"] = rejected

    return result if result else None


def _create_mcp_client_from_config(server_name: str, config: dict[str, Any]) -> MCPClient:
    """Create an MCPClient instance from a server configuration dictionary.

    Transport type is auto-detected based on the presence of 'command' (stdio) or 'url' (sse),
    unless explicitly specified via the 'transport' field.

    Args:
        server_name: Name of the server (used in error messages).
        config: Server configuration dictionary.

    Returns:
        A configured MCPClient instance.

    Raises:
        ValueError: If the configuration is invalid or missing required fields.
    """
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

        def _stdio_transport() -> Any:
            params = StdioServerParameters(
                command=config["command"],
                args=config.get("args", []),
                env=config.get("env"),
                cwd=config.get("cwd"),
            )
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
            return streamable_http_client(url=url, headers=headers)

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


def load_mcp_clients_from_config(config: str | dict[str, Any]) -> dict[str, MCPClient]:
    """Load MCP client instances from a configuration file or dictionary.

    Expects the standard ``mcpServers`` wrapper format used by Claude Desktop, VS Code, etc::

        {
            "mcpServers": {
                "server_name": { "command": "...", ... }
            }
        }

    Args:
        config: Either a file path (with optional file:// prefix) to a JSON config file,
            or a dictionary with a ``mcpServers`` key mapping server names to configs.

    Returns:
        A dictionary mapping server names to MCPClient instances.

    Raises:
        FileNotFoundError: If the config file does not exist.
        json.JSONDecodeError: If the config file contains invalid JSON.
        ValueError: If the config format is invalid or a server config is invalid.
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
    clients: dict[str, MCPClient] = {}
    for server_name, server_config in servers.items():
        clients[server_name] = _create_mcp_client_from_config(server_name, server_config)

    logger.debug("loaded_servers=<%d> | MCP clients created from config", len(clients))

    return clients
