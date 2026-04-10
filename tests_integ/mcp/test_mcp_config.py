"""Integration tests for loading MCP servers from config."""

import json
import os
import tempfile

import pytest

from strands import Agent
from strands.experimental.mcp_config import load_mcp_clients_from_config


def test_load_stdio_server_from_config():
    """Test loading a stdio MCP server from config dict and using it with an agent."""
    config = {
        "mcpServers": {
            "echo": {
                "command": "python",
                "args": ["tests_integ/mcp/echo_server.py"],
                "prefix": "cfg",
                "tool_filters": {"allowed": ["cfg_echo"]},
            }
        }
    }

    clients = load_mcp_clients_from_config(config)
    assert "echo" in clients

    agent = Agent(tools=list(clients.values()))
    assert "cfg_echo" in agent.tool_names

    result = agent.tool.cfg_echo(to_echo="Config Test")
    assert "Config Test" in str(result)

    agent.cleanup()


def test_load_stdio_server_from_json_file():
    """Test loading a stdio MCP server from a JSON config file."""
    config_data = {
        "mcpServers": {
            "echo": {
                "command": "python",
                "args": ["tests_integ/mcp/echo_server.py"],
                "prefix": "file",
                "tool_filters": {"allowed": ["file_echo"]},
            }
        }
    }
    temp_path = ""

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        clients = load_mcp_clients_from_config(temp_path)
        assert "echo" in clients

        agent = Agent(tools=list(clients.values()))
        assert "file_echo" in agent.tool_names

        result = agent.tool.file_echo(to_echo="File Config Test")
        assert "File Config Test" in str(result)

        agent.cleanup()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_load_multiple_servers_from_config():
    """Test loading multiple MCP servers from a single config."""
    config = {
        "mcpServers": {
            "server1": {
                "command": "python",
                "args": ["tests_integ/mcp/echo_server.py"],
                "prefix": "s1",
                "tool_filters": {"allowed": ["s1_echo"]},
            },
            "server2": {
                "command": "python",
                "args": ["tests_integ/mcp/echo_server.py"],
                "prefix": "s2",
                "tool_filters": {"allowed": ["s2_echo"]},
            },
        }
    }

    clients = load_mcp_clients_from_config(config)
    assert len(clients) == 2

    agent = Agent(tools=list(clients.values()))
    assert "s1_echo" in agent.tool_names
    assert "s2_echo" in agent.tool_names

    result1 = agent.tool.s1_echo(to_echo="From Server 1")
    assert "From Server 1" in str(result1)

    result2 = agent.tool.s2_echo(to_echo="From Server 2")
    assert "From Server 2" in str(result2)

    agent.cleanup()
