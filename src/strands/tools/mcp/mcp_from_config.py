"""MCP server configuration loading utilities."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

from .mcp_client import MCPClient
from .mcp_types import MCPTransport


class MCPServerConfig:
    """Configuration for an MCP server following MCP standards."""

    def __init__(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ):
        """Initialize MCP server configuration.

        Args:
            name: Server name
            command: Command to run the server
            args: Command arguments
            env: Environment variables
            timeout: Timeout in milliseconds
        """
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.timeout = timeout or 60000

    def create_client(self) -> MCPClient:
        """Create an MCPClient from this configuration."""

        def transport_callable() -> MCPTransport:
            server_params = StdioServerParameters(command=self.command, args=self.args, env=self.env)
            return stdio_client(server_params)

        return MCPClient(transport_callable)

    @classmethod
    def from_config(cls, config_path: str) -> List["MCPServerConfig"]:
        """Load MCP server configurations from standard mcp.json format.

        Args:
            config_path: Path to the MCP configuration file

        Returns:
            List of MCPServerConfig instances

        Config file examples:
        Anthropic MCP Server Config Examples: (https://modelcontextprotocol.io/examples)
        AmazonQ MCP Server Config Examples: (https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/command-line-mcp-understanding-config.html)

        Expected format:
        {
          "mcpServers": {
            "server-name": {
              "command": "command-to-run",
              "args": ["arg1", "arg2"],
              "env": {
                "ENV_VAR1": "value1",
                "ENV_VAR2": "value2"
              },
              "timeout": 60000
            }
          }
        }

        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file) as f:
            config_data = json.load(f)

        servers = []
        mcp_server_name = set()
        mcp_servers = config_data.get("mcpServers", {})
        expected_attrs = {"command", "args", "env", "timeout"}

        for name, server_config in mcp_servers.items():
            if len(name) == 0 or len(name) > 250 or server_config.get("command") is None or name in mcp_server_name:
                raise ValueError(f"Invalid server configuration for {name}")
            if set(server_config.keys()) - expected_attrs:
                raise ValueError(f"Invalid server configuration for {name}")

            servers.append(
                cls(
                    name=name,
                    command=server_config["command"],
                    args=server_config.get("args"),
                    env=server_config.get("env"),
                    timeout=server_config.get("timeout"),
                )
            )
            mcp_server_name.add(name)

        return servers
