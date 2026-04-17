"""Integration tests for MCP configuration loading.

These tests verify end-to-end MCP tool usage from configuration,
using the echo_server.py for stdio transport testing.
"""

import json
import sys
import tempfile

import pytest

from strands import Agent
from strands.experimental import config_to_agent
from strands.tools.mcp import MCPClient, load_mcp_clients_from_config

ECHO_SERVER_PATH = "tests_integ/mcp/echo_server.py"


@pytest.fixture
def echo_server_config():
    """Configuration for the echo server via stdio transport."""
    return {
        "echo": {
            "command": sys.executable,
            "args": [ECHO_SERVER_PATH],
            "startup_timeout": 30,
        }
    }


@pytest.fixture
def echo_server_config_with_prefix():
    """Configuration for the echo server with a tool name prefix."""
    return {
        "echo": {
            "command": sys.executable,
            "args": [ECHO_SERVER_PATH],
            "prefix": "test",
            "startup_timeout": 30,
        }
    }


@pytest.fixture
def echo_server_config_with_filters():
    """Configuration for the echo server with tool filters."""
    return {
        "echo": {
            "command": sys.executable,
            "args": [ECHO_SERVER_PATH],
            "tool_filters": {
                "allowed": ["echo"],
                "rejected": ["get_weather"],
            },
            "startup_timeout": 30,
        }
    }


class TestLoadMcpClientsFromConfig:
    """Tests for loading MCP clients from configuration dictionaries."""

    def test_load_stdio_client(self, echo_server_config):
        """Verify a stdio MCP client can be loaded from config."""
        clients = load_mcp_clients_from_config(echo_server_config)
        assert "echo" in clients
        assert isinstance(clients["echo"], MCPClient)

    def test_load_from_file(self, echo_server_config):
        """Verify MCP clients can be loaded from a JSON config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(echo_server_config, f)
            f.flush()
            temp_path = f.name

        clients = load_mcp_clients_from_config(temp_path)
        assert "echo" in clients
        assert isinstance(clients["echo"], MCPClient)

    def test_load_from_file_with_mcp_servers_key(self, echo_server_config):
        """Verify loading from file where config is nested under mcp_servers key."""
        wrapped_config = {"mcp_servers": echo_server_config}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(wrapped_config, f)
            f.flush()
            temp_path = f.name

        clients = load_mcp_clients_from_config(temp_path)
        assert "echo" in clients

    def test_client_can_list_tools(self, echo_server_config):
        """Verify loaded client can connect and list tools."""
        clients = load_mcp_clients_from_config(echo_server_config)
        client = clients["echo"]

        with client:
            tools = client.list_tools_sync()
            tool_names = [t.tool_name for t in tools]
            assert "echo" in tool_names

    def test_client_with_prefix(self, echo_server_config_with_prefix):
        """Verify tool prefix is applied from config."""
        clients = load_mcp_clients_from_config(echo_server_config_with_prefix)
        client = clients["echo"]

        with client:
            tools = client.list_tools_sync()
            tool_names = [t.tool_name for t in tools]
            # All tool names should be prefixed
            assert all(name.startswith("test_") for name in tool_names)

    def test_client_with_tool_filters(self, echo_server_config_with_filters):
        """Verify tool filters are applied from config."""
        clients = load_mcp_clients_from_config(echo_server_config_with_filters)
        client = clients["echo"]

        with client:
            tools = client.list_tools_sync()
            tool_names = [t.tool_name for t in tools]
            assert "echo" in tool_names
            assert "get_weather" not in tool_names


class TestConfigToAgentWithMcpServers:
    """Tests for config_to_agent with mcp_servers field."""

    def test_agent_with_mcp_servers(self, echo_server_config):
        """Verify an agent can be created with MCP tools from config."""
        config = {
            "mcp_servers": echo_server_config,
        }
        agent = config_to_agent(config)
        assert "echo" in agent.tool_names

    def test_agent_with_mcp_servers_from_file(self, echo_server_config):
        """Verify an agent can be created from a config file with mcp_servers."""
        config = {
            "mcp_servers": echo_server_config,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()
            temp_path = f.name

        agent = config_to_agent(temp_path)
        assert "echo" in agent.tool_names

    def test_agent_can_call_mcp_tool(self, echo_server_config):
        """Verify an agent with MCP config tools can invoke them."""
        config = {
            "mcp_servers": echo_server_config,
        }
        agent = config_to_agent(config)
        result = agent.tool.echo(to_echo="hello from config")
        assert "hello from config" in str(result)

    def test_agent_standalone_usage(self, echo_server_config):
        """Verify standalone usage: load clients, then pass to Agent."""
        clients = load_mcp_clients_from_config(echo_server_config)
        agent = Agent(tools=list(clients.values()))
        assert "echo" in agent.tool_names

    def test_multiple_servers_config(self):
        """Verify multiple stdio servers can be configured."""
        config = {
            "echo_a": {
                "command": sys.executable,
                "args": [ECHO_SERVER_PATH],
                "prefix": "a",
                "startup_timeout": 30,
            },
            "echo_b": {
                "command": sys.executable,
                "args": [ECHO_SERVER_PATH],
                "prefix": "b",
                "startup_timeout": 30,
            },
        }
        clients = load_mcp_clients_from_config(config)
        assert len(clients) == 2

        agent = Agent(tools=list(clients.values()))
        tool_names = agent.tool_names
        assert any(name.startswith("a_") for name in tool_names)
        assert any(name.startswith("b_") for name in tool_names)
