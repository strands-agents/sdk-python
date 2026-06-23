"""Integration tests for loading MCP servers from config.

These tests use the local stdio ``echo_server.py`` and are network-free.
"""

import json
import os
import tempfile

from strands import Agent
from strands.experimental.mcp_config import load_mcp_clients_from_config

_ECHO_SERVER_ARGS = ["tests_integ/mcp/echo_server.py"]


def test_load_stdio_server_from_config():
    """Load a stdio MCP server from a config dict and use it with an agent."""
    config = {
        "mcpServers": {
            "echo": {
                "command": "python",
                "args": _ECHO_SERVER_ARGS,
                "prefix": "cfg",
                "tool_filters": {"allowed": ["cfg_echo$"]},
            }
        }
    }

    clients = load_mcp_clients_from_config(config)
    assert len(clients) == 1

    agent = Agent(tools=clients)
    assert "cfg_echo" in agent.tool_names

    result = agent.tool.cfg_echo(to_echo="Config Test")
    assert "Config Test" in str(result)

    agent.cleanup()


def test_load_stdio_server_from_json_file():
    """Load a stdio MCP server from a JSON config file."""
    config_data = {
        "mcpServers": {
            "echo": {
                "command": "python",
                "args": _ECHO_SERVER_ARGS,
                "prefix": "file",
                "tool_filters": {"allowed": ["file_echo$"]},
            }
        }
    }
    temp_path = ""

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        clients = load_mcp_clients_from_config(temp_path)
        assert len(clients) == 1

        agent = Agent(tools=clients)
        assert "file_echo" in agent.tool_names

        result = agent.tool.file_echo(to_echo="File Config Test")
        assert "File Config Test" in str(result)

        agent.cleanup()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_env_interpolation_and_disabled_skip(monkeypatch):
    """Resolve ``${env:VAR}`` in the prefix and skip a disabled server."""
    monkeypatch.setenv("ECHO_PREFIX", "dyn")
    config = {
        "mcpServers": {
            "echo": {
                "command": "python",
                "args": _ECHO_SERVER_ARGS,
                "prefix": "${env:ECHO_PREFIX}",
                "tool_filters": {"allowed": ["dyn_echo$"]},
            },
            "disabled_echo": {
                "command": "python",
                "args": _ECHO_SERVER_ARGS,
                "disabled": True,
            },
        }
    }

    clients = load_mcp_clients_from_config(config)
    assert len(clients) == 1

    agent = Agent(tools=clients)
    assert "dyn_echo" in agent.tool_names

    result = agent.tool.dyn_echo(to_echo="Dynamic Prefix")
    assert "Dynamic Prefix" in str(result)

    agent.cleanup()
