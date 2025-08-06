import json
from unittest.mock import patch

import pytest

from strands.tools.mcp.mcp_from_config import MCPServerConfig


@pytest.fixture
def mcp_config_data():
    """Valid MCP configuration data."""
    return {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                "env": {"NODE_ENV": "production"},
                "timeout": 60000,
            },
            "calculator": {"command": "python", "args": ["calculator_server.py"]},
        }
    }


@pytest.fixture
def mcp_config_file(tmp_path, mcp_config_data):
    """Create a temporary MCP config file."""
    config_path = tmp_path / "mcp_config.json"
    with open(config_path, "w") as f:
        json.dump(mcp_config_data, f)
    return str(config_path)


@pytest.fixture
def invalid_mcp_config_data():
    """Invalid MCP configuration data for testing validation."""
    return {
        "mcpServers": {
            "": {  # Empty name
                "command": "echo"
            },
            "no_command": {  # Missing command
                "args": ["test"]
            },
        }
    }


class TestMCPServerConfig:
    """Test MCPServerConfig core functionality."""

    def test_init_defaults(self):
        """Test MCPServerConfig initialization with defaults."""
        config = MCPServerConfig(name="test", command="echo")

        assert config.name == "test"
        assert config.command == "echo"
        assert config.args == []
        assert config.env == {}
        assert config.timeout == 60000

    @patch("strands.tools.mcp.mcp_from_config.MCPClient")
    def test_create_client(self, mock_mcp_client):
        """Test create_client method creates MCPClient."""
        config = MCPServerConfig(name="test", command="echo")

        result = config.create_client()

        mock_mcp_client.assert_called_once()
        assert result == mock_mcp_client.return_value

    def test_from_config_valid_file(self, mcp_config_file):
        """Test from_config with valid config file."""
        servers = MCPServerConfig.from_config(mcp_config_file)

        assert len(servers) == 2
        server_names = {s.name for s in servers}
        assert server_names == {"filesystem", "calculator"}

    def test_from_config_file_not_found(self):
        """Test from_config with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            MCPServerConfig.from_config("/nonexistent/path.json")

    def test_from_config_validation_error(self, tmp_path, invalid_mcp_config_data):
        """Test from_config with validation errors."""
        config_path = tmp_path / "invalid.json"
        with open(config_path, "w") as f:
            json.dump(invalid_mcp_config_data, f)

        with pytest.raises(ValueError):
            MCPServerConfig.from_config(str(config_path))
