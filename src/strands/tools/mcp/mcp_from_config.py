"""MCP server configuration loading utilities."""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

from mcp import StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from .mcp_client import MCPClient
from .mcp_types import MCPTransport


class MCPTransportType(Enum):
    """MCP transport types."""

    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable-http"
    SSE = "sse"


@dataclass
class MCPTransportConfig(ABC):
    """Base configuration for MCP Transports."""

    name: str
    timeout: float = 60000
    transport_type: Optional[MCPTransportType] = None

    @abstractmethod
    def create_transport_callable(self) -> Callable[[], MCPTransport]:
        """Create a transport callable for this configuration."""
        pass


@dataclass
class StdioTransportConfig(MCPTransportConfig):
    """Configuration for STDIO transport (local subprocess)."""

    command: str = ""
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None

    def __post_init__(self) -> None:
        """Set transport type after initialization."""
        self.transport_type = MCPTransportType.STDIO

    def create_transport_callable(self) -> Callable[[], MCPTransport]:
        """Create STDIO transport callable."""

        def transport_callable() -> MCPTransport:
            server_params = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=self.env,
                cwd=self.cwd,
            )
            return stdio_client(server_params)

        return transport_callable


@dataclass
class HTTPTransportConfig(MCPTransportConfig):
    """Configuration for HTTP transport."""

    url: str = ""
    authorization_token: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set transport type after initialization."""
        self.transport_type = MCPTransportType.STREAMABLE_HTTP

    def create_transport_callable(self) -> Callable[[], MCPTransport]:
        """Create STREAMABLE_HTTP transport callable."""

        def transport_callable() -> MCPTransport:
            headers = self.headers.copy()
            if self.authorization_token:
                headers["Authorization"] = f"Bearer {self.authorization_token}"
            return streamablehttp_client(self.url, headers=headers if headers else None)

        return transport_callable


@dataclass
class SSETransportConfig(MCPTransportConfig):
    """Configuration for SSE transport."""

    url: str = ""
    authorization_token: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set transport type after initialization."""
        self.transport_type = MCPTransportType.SSE

    def create_transport_callable(self) -> Callable[[], MCPTransport]:
        """Create SSE transport callable."""

        def transport_callable() -> MCPTransport:
            headers = self.headers.copy() if self.headers else {}
            if self.authorization_token:
                headers["Authorization"] = f"Bearer {self.authorization_token}"
            return sse_client(self.url, headers=headers if headers else None)

        return transport_callable


def _infer_transport_from_url(url: str) -> str:
    """Infer transport type from URL when none specified.

    - If path contains '/sse' (optionally followed by / ? & or end), treat as 'sse'
    - Otherwise default to 'streamable-http'
    """
    try:
        path = (urlparse(url).path or "").lower()
    except Exception:
        return "streamable-http"
    return "sse" if re.search(r"/sse(/|\?|&|$)", path) else "streamable-http"


class MCPServerConfig:
    """Configuration for an MCP server following MCP standards."""

    def __init__(self, transport_config: MCPTransportConfig) -> None:
        """Initialize MCP server configuration."""
        self.transport_config = transport_config
        self.name = transport_config.name
        self.timeout = transport_config.timeout

    def create_client(self) -> MCPClient:
        """Create an MCPClient from this configuration."""
        return MCPClient(self.transport_config.create_transport_callable())

    @classmethod
    def from_config(cls, config_path: str) -> List["MCPServerConfig"]:
        """Load MCP server configurations from a config file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file) as f:
            config_data = json.load(f)

        if "mcpServers" in config_data and config_data["mcpServers"] is not None:
            return cls._parse_mcp_servers_format(config_data["mcpServers"])

        return []

    @classmethod
    def _parse_mcp_servers_format(cls, mcp_servers: Dict[str, Dict]) -> List["MCPServerConfig"]:
        """Parse mcpServers format (Claude Desktop/legacy and enhanced)."""
        servers = []
        server_names = set()

        for name, server_config in mcp_servers.items():
            if len(name) == 0 or len(name) > 250 or name in server_names:
                raise ValueError(f"Invalid server name: {name}")
            server_names.add(name)

            # Accept multiple keys, infer from URL when absent
            raw_transport = (
                server_config.get("transport")
                or server_config.get("transportType")
                or server_config.get("transport_type")
            )
            if raw_transport is None and "url" in server_config:
                raw_transport = _infer_transport_from_url(str(server_config["url"]))
            transport_type = (raw_transport or "stdio").lower()

            timeout = server_config.get("timeout", 60000)

            try:
                if transport_type == "stdio" or "command" in server_config:
                    if "command" not in server_config:
                        raise ValueError(f"STDIO server {name} missing 'command'")

                    transport_config: MCPTransportConfig = StdioTransportConfig(
                        name=name,
                        command=server_config["command"],
                        args=server_config.get("args", []),
                        env=server_config.get("env", {}),
                        cwd=server_config.get("cwd"),
                        timeout=timeout,
                    )

                elif transport_type == "streamable-http":
                    if "url" not in server_config:
                        raise ValueError(f"Steamable-HTTP server {name} missing 'url'")

                    transport_config = HTTPTransportConfig(
                        name=name,
                        url=server_config["url"],
                        authorization_token=server_config.get("authorization_token"),
                        headers=server_config.get("headers", {}),
                        timeout=timeout,
                    )

                elif transport_type == "sse":
                    if "url" not in server_config:
                        raise ValueError(f"SSE server {name} missing 'url'")

                    transport_config = SSETransportConfig(
                        name=name,
                        url=server_config["url"],
                        authorization_token=server_config.get("authorization_token"),
                        headers=server_config.get("headers", {}),
                        timeout=timeout,
                    )
                else:
                    raise ValueError(f"Unsupported transport type: {transport_type}")

                servers.append(cls(transport_config))

            except Exception as e:
                raise ValueError(f"Invalid configuration for server {name}: {e}") from e

        return servers
