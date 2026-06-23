"""Tests for experimental config_to_agent function."""

import json
import os
import tempfile

import pytest

from strands.experimental import config_to_agent


def test_config_to_agent_with_dict():
    """Test config_to_agent can be created with dict config."""
    config = {"model": "test-model"}
    agent = config_to_agent(config)
    assert agent.model.config["model_id"] == "test-model"


def test_config_to_agent_with_system_prompt():
    """Test config_to_agent handles system prompt correctly."""
    config = {"model": "test-model", "prompt": "Test prompt"}
    agent = config_to_agent(config)
    assert agent.system_prompt == "Test prompt"


def test_config_to_agent_with_tools_list():
    """Test config_to_agent handles tools list without failing."""
    # Use a simple test that doesn't require actual tool loading
    config = {"model": "test-model", "tools": []}
    agent = config_to_agent(config)
    assert agent.model.config["model_id"] == "test-model"


def test_config_to_agent_with_kwargs_override():
    """Test that kwargs can override config values."""
    config = {"model": "test-model", "prompt": "Config prompt"}
    agent = config_to_agent(config, system_prompt="Override prompt")
    assert agent.system_prompt == "Override prompt"


def test_config_to_agent_file_prefix_required():
    """Test that file paths without file:// prefix work."""
    import json
    import tempfile

    config_data = {"model": "test-model"}
    temp_path = ""

    # We need to create files like this for windows compatibility
    try:
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            f.flush()
            temp_path = f.name

        agent = config_to_agent(temp_path)
        assert agent.model.config["model_id"] == "test-model"
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_config_to_agent_file_prefix_valid():
    """Test that file:// prefix is properly handled."""
    config_data = {"model": "test-model", "prompt": "Test prompt"}
    temp_path = ""

    try:
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            f.flush()
            temp_path = f.name

        agent = config_to_agent(f"file://{temp_path}")
        assert agent.model.config["model_id"] == "test-model"
        assert agent.system_prompt == "Test prompt"
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_config_to_agent_file_not_found():
    """Test that FileNotFoundError is raised for missing files."""
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        config_to_agent("/nonexistent/path/config.json")


def test_config_to_agent_invalid_json():
    """Test that JSONDecodeError is raised for invalid JSON."""
    try:
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name

        with pytest.raises(json.JSONDecodeError):
            config_to_agent(temp_path)
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_config_to_agent_invalid_config_type():
    """Test that ValueError is raised for invalid config types."""
    with pytest.raises(ValueError, match="Config must be a file path string or dictionary"):
        config_to_agent(123)


def test_config_to_agent_with_name():
    """Test config_to_agent handles agent name."""
    config = {"model": "test-model", "name": "TestAgent"}
    agent = config_to_agent(config)
    assert agent.name == "TestAgent"


def test_config_to_agent_ignores_none_values():
    """Test that None values in config are ignored."""
    config = {"model": "test-model", "prompt": None, "name": None}
    agent = config_to_agent(config)
    assert agent.model.config["model_id"] == "test-model"
    # Agent should use its defaults for None values


def test_config_to_agent_validation_error_invalid_field():
    """Test that invalid fields raise validation errors."""
    config = {"model": "test-model", "invalid_field": "value"}
    with pytest.raises(ValueError, match="Configuration validation error"):
        config_to_agent(config)


def test_config_to_agent_validation_error_wrong_type():
    """Test that wrong field types raise validation errors."""
    config = {"model": "test-model", "tools": "not-a-list"}
    with pytest.raises(ValueError, match="Configuration validation error"):
        config_to_agent(config)


def test_config_to_agent_validation_error_invalid_tool_item():
    """Test that invalid tool items raise validation errors."""
    config = {"model": "test-model", "tools": ["valid-tool", 123]}
    with pytest.raises(ValueError, match="Configuration validation error"):
        config_to_agent(config)


def test_config_to_agent_validation_error_invalid_tool():
    """Test that invalid tools raise helpful error messages."""
    config = {"model": "test-model", "tools": ["nonexistent_tool"]}
    with pytest.raises(ValueError, match="Failed to load tool nonexistent_tool"):
        config_to_agent(config)


def test_config_to_agent_validation_error_missing_module():
    """Test that missing modules raise helpful error messages."""
    config = {"model": "test-model", "tools": ["nonexistent.module.tool"]}
    with pytest.raises(ValueError, match="Failed to load tool nonexistent.module.tool"):
        config_to_agent(config)


def test_config_to_agent_validation_error_missing_function():
    """Test that missing functions in existing modules raise helpful error messages."""
    config = {"model": "test-model", "tools": ["json.nonexistent_function"]}
    with pytest.raises(ValueError, match="Failed to load tool json.nonexistent_function"):
        config_to_agent(config)


def test_config_to_agent_with_tool():
    """Test that missing functions in existing modules raise helpful error messages."""
    config = {"model": "test-model", "tools": ["tests.fixtures.say_tool:say"]}
    agent = config_to_agent(config)
    assert "say" in agent.tool_names


def test_config_to_agent_with_mcp_servers_adds_clients(monkeypatch):
    """mcp_servers entries are loaded as MCPClient instances and added to the agent's tools."""
    captured = {}

    class _FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    # Patch the Agent imported lazily inside config_to_agent.
    monkeypatch.setattr("strands.agent.Agent", _FakeAgent)

    config = {
        "model": "test-model",
        "mcp_servers": {
            "echo": {"command": "python", "args": ["echo_server.py"], "prefix": "echo"},
            "remote": {"url": "https://example.com/sse"},
        },
    }

    config_to_agent(config)

    from strands.tools.mcp import MCPClient

    tools = captured.get("tools", [])
    assert len(tools) == 2
    assert all(isinstance(t, MCPClient) for t in tools)


def test_config_to_agent_mcp_servers_appended_after_regular_tools(monkeypatch):
    """MCP clients are appended after declaratively loaded tools, preserving order."""
    captured = {}

    class _FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("strands.agent.Agent", _FakeAgent)

    config = {
        "model": "test-model",
        "tools": ["tests.fixtures.say_tool:say"],
        "mcp_servers": {"echo": {"command": "python", "args": ["echo_server.py"]}},
    }

    config_to_agent(config)

    from strands.tools.mcp import MCPClient

    tools = captured.get("tools", [])
    assert len(tools) == 2
    # Regular tool first, MCP client appended last.
    assert isinstance(tools[-1], MCPClient)


def test_config_to_agent_mcp_servers_skips_disabled(monkeypatch):
    """Servers marked disabled produce no client."""
    captured = {}

    class _FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("strands.agent.Agent", _FakeAgent)

    config = {
        "model": "test-model",
        "mcp_servers": {
            "echo": {"command": "python", "args": ["echo_server.py"]},
            "legacy": {"command": "old-server", "disabled": True},
        },
    }

    config_to_agent(config)

    from strands.tools.mcp import MCPClient

    tools = captured.get("tools", [])
    assert len(tools) == 1
    assert isinstance(tools[0], MCPClient)


def test_config_to_agent_mcp_servers_env_interpolation(monkeypatch):
    """${env:VAR} references in mcp_servers config are resolved via the shared loader."""
    captured = {}

    class _FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("strands.agent.Agent", _FakeAgent)
    monkeypatch.setenv("MCP_TEST_TOKEN", "secret-value")

    config = {
        "model": "test-model",
        "mcp_servers": {
            "remote": {
                "url": "https://example.com/sse",
                "headers": {"Authorization": "Bearer ${env:MCP_TEST_TOKEN}"},
            }
        },
    }

    # Should not raise; missing env var would raise ValueError.
    config_to_agent(config)
    from strands.tools.mcp import MCPClient

    assert len(captured.get("tools", [])) == 1
    assert isinstance(captured["tools"][0], MCPClient)


def test_config_to_agent_mcp_servers_invalid_server_config(monkeypatch):
    """An invalid per-server config surfaces a clear error with the server name."""
    monkeypatch.setattr("strands.agent.Agent", lambda **kwargs: None)

    config = {
        "model": "test-model",
        "mcp_servers": {"broken": {"unknown_field": "x"}},
    }

    with pytest.raises(ValueError, match="broken"):
        config_to_agent(config)


def test_config_to_agent_mcp_servers_wrong_type():
    """mcp_servers must be an object/dict, not a list."""
    config = {"model": "test-model", "mcp_servers": ["not", "a", "dict"]}
    with pytest.raises(ValueError, match="Configuration validation error"):
        config_to_agent(config)
