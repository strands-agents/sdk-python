import json
from unittest.mock import patch

import pytest

from strands.tools.mcp.mcp_from_config import (
    HTTPTransportConfig,
    MCPServerConfig,
    MCPTransportType,
    SSETransportConfig,
    StdioTransportConfig,
    _infer_transport_from_url,
)


def create_config_file(tmp_path, config_data):
    """Helper to create config file."""
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f)
    return str(config_path)


class TestMCPTransportType:
    """Test MCPTransportType enum."""

    def test_enum_values(self):
        """Test enum has correct values."""
        assert MCPTransportType.STDIO.value == "stdio"
        assert MCPTransportType.STREAMABLE_HTTP.value == "streamable-http"
        assert MCPTransportType.SSE.value == "sse"


class TestStdioTransportConfig:
    """Test StdioTransportConfig."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        config = StdioTransportConfig(name="test")

        assert config.name == "test"
        assert config.command == ""
        assert config.args == []
        assert config.env == {}
        assert config.cwd is None
        assert config.timeout == 60000
        assert config.transport_type == MCPTransportType.STDIO

    @patch("strands.tools.mcp.mcp_from_config.stdio_client")
    def test_create_transport_callable(self, mock_stdio_client):
        """Test transport callable creation."""
        config = StdioTransportConfig(
            name="test", command="python", args=["script.py"], env={"VAR": "value"}, cwd="/tmp"
        )

        transport_callable = config.create_transport_callable()
        transport_callable()

        mock_stdio_client.assert_called_once()


class TestHTTPTransportConfig:
    """Test HTTPTransportConfig."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        config = HTTPTransportConfig(name="test")

        assert config.name == "test"
        assert config.url == ""
        assert config.authorization_token is None
        assert config.headers == {}
        assert config.timeout == 60000
        assert config.transport_type == MCPTransportType.STREAMABLE_HTTP

    @patch("strands.tools.mcp.mcp_from_config.streamablehttp_client")
    def test_create_transport_callable_with_auth(self, mock_client):
        """Test transport callable with authorization."""
        config = HTTPTransportConfig(
            name="test", url="http://example.com/mcp", authorization_token="token123", headers={"Custom": "header"}
        )

        transport_callable = config.create_transport_callable()
        transport_callable()

        mock_client.assert_called_once()


class TestSSETransportConfig:
    """Test SSETransportConfig."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        config = SSETransportConfig(name="test")

        assert config.name == "test"
        assert config.url == ""
        assert config.authorization_token is None
        assert config.headers == {}
        assert config.timeout == 60000
        assert config.transport_type == MCPTransportType.SSE

    @patch("strands.tools.mcp.mcp_from_config.sse_client")
    def test_create_transport_callable_with_auth(self, mock_client):
        """Test transport callable with authorization."""
        config = SSETransportConfig(
            name="test",
            url="http://example.com/sse",
            authorization_token="token123",
            headers={"Accept": "text/event-stream"},
        )

        transport_callable = config.create_transport_callable()
        transport_callable()

        mock_client.assert_called_once()


class TestInferTransportFromUrl:
    """Test _infer_transport_from_url function."""

    def test_infer_sse_from_path(self):
        """Test SSE inference from URL path."""
        assert _infer_transport_from_url("http://example.com/sse") == "sse"
        assert _infer_transport_from_url("http://example.com/api/sse") == "sse"

    def test_infer_http_from_path(self):
        """Test HTTP inference from URL path."""
        assert _infer_transport_from_url("http://example.com/mcp") == "streamable-http"
        assert _infer_transport_from_url("http://example.com/api") == "streamable-http"


class TestMCPServerConfig:
    """Test MCPServerConfig."""

    @patch("strands.tools.mcp.mcp_from_config.MCPClient")
    def test_create_client(self, mock_mcp_client):
        """Test client creation."""
        transport_config = StdioTransportConfig(name="test", command="echo")
        config = MCPServerConfig(transport_config)

        result = config.create_client()

        mock_mcp_client.assert_called_once()
        assert result == mock_mcp_client.return_value

    def test_from_config_stdio(self, tmp_path):
        """Test parsing STDIO config."""
        config_data = {
            "mcpServers": {"test": {"transport": "stdio", "command": "echo", "args": ["hello"], "timeout": 30000}}
        }
        config_path = create_config_file(tmp_path, config_data)
        servers = MCPServerConfig.from_config(config_path)

        assert len(servers) == 1
        assert isinstance(servers[0].transport_config, StdioTransportConfig)
        assert servers[0].transport_config.command == "echo"

    def test_from_config_http(self, tmp_path):
        """Test parsing HTTP config."""
        config_data = {
            "mcpServers": {"test": {"transport": "streamable-http", "url": "http://example.com/mcp", "timeout": 30000}}
        }
        config_path = create_config_file(tmp_path, config_data)
        servers = MCPServerConfig.from_config(config_path)

        assert len(servers) == 1
        assert isinstance(servers[0].transport_config, HTTPTransportConfig)
        assert servers[0].transport_config.url == "http://example.com/mcp"

    def test_from_config_sse(self, tmp_path):
        """Test parsing SSE config."""
        config_data = {"mcpServers": {"test": {"transport": "sse", "url": "http://example.com/sse", "timeout": 30000}}}
        config_path = create_config_file(tmp_path, config_data)
        servers = MCPServerConfig.from_config(config_path)

        assert len(servers) == 1
        assert isinstance(servers[0].transport_config, SSETransportConfig)
        assert servers[0].transport_config.url == "http://example.com/sse"

    def test_from_config_file_not_found(self):
        """Test with nonexistent file."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            MCPServerConfig.from_config("/nonexistent/path.json")

    def test_validation_missing_command(self, tmp_path):
        """Test validation with missing command for STDIO."""
        config_data = {"mcpServers": {"test": {"transport": "stdio"}}}
        config_path = create_config_file(tmp_path, config_data)

        with pytest.raises(ValueError, match="STDIO server test missing 'command'"):
            MCPServerConfig.from_config(config_path)

    def test_validation_missing_url(self, tmp_path):
        """Test validation with missing URL for HTTP."""
        config_data = {"mcpServers": {"test": {"transport": "streamable-http"}}}
        config_path = create_config_file(tmp_path, config_data)

        with pytest.raises(ValueError, match="Steamable-HTTP server test missing 'url'"):
            MCPServerConfig.from_config(config_path)
