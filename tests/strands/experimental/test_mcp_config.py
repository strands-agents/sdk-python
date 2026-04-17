"""Tests for MCP config parsing and MCPClient factory."""

import json
import os
import re
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from strands.experimental.mcp_config import (
    _create_mcp_client_from_config,
    _parse_tool_filters,
    load_mcp_clients_from_config,
)


class TestParseToolFilters:
    """Tests for _parse_tool_filters function."""

    def test_returns_none_for_none_input(self):
        assert _parse_tool_filters(None) is None

    def test_returns_none_for_empty_dict(self):
        assert _parse_tool_filters({}) is None

    def test_compiles_allowed_patterns(self):
        config = {"allowed": ["echo", "search"]}
        result = _parse_tool_filters(config)
        assert result is not None
        assert len(result["allowed"]) == 2
        assert isinstance(result["allowed"][0], re.Pattern)
        assert result["allowed"][0].match("echo")

    def test_compiles_rejected_patterns(self):
        config = {"rejected": ["dangerous_tool"]}
        result = _parse_tool_filters(config)
        assert result is not None
        assert isinstance(result["rejected"][0], re.Pattern)
        assert result["rejected"][0].match("dangerous_tool")

    def test_compiles_regex_patterns(self):
        config = {"allowed": ["search_.*", "get_\\w+"]}
        result = _parse_tool_filters(config)
        assert result is not None
        assert result["allowed"][0].match("search_docs")
        assert result["allowed"][1].match("get_data")

    def test_mixed_allowed_and_rejected(self):
        config = {"allowed": ["search_.*"], "rejected": ["dangerous_tool"]}
        result = _parse_tool_filters(config)
        assert result is not None
        assert "allowed" in result
        assert "rejected" in result

    def test_invalid_regex_raises_error(self):
        config = {"allowed": ["[invalid"]}
        with pytest.raises(ValueError, match="invalid regex pattern"):
            _parse_tool_filters(config)


class TestCreateMcpClientFromConfig:
    """Tests for _create_mcp_client_from_config function."""

    @patch("strands.experimental.mcp_config.stdio_client")
    @patch("strands.experimental.mcp_config.StdioServerParameters")
    def test_stdio_transport_from_command(self, mock_params_cls, mock_stdio_client):
        config = {"command": "uvx", "args": ["some-server@latest"]}
        mock_params_cls.return_value = MagicMock()
        client = _create_mcp_client_from_config("test_server", config)
        assert client is not None

    @patch("strands.experimental.mcp_config.sse_client")
    def test_sse_transport_explicit(self, mock_sse_client):
        config = {"transport": "sse", "url": "http://localhost:8000/sse"}
        client = _create_mcp_client_from_config("test_server", config)
        assert client is not None

    @patch("strands.experimental.mcp_config.streamablehttp_client")
    def test_streamable_http_transport_explicit(self, mock_http_client):
        config = {"transport": "streamable-http", "url": "http://localhost:8000/mcp"}
        client = _create_mcp_client_from_config("test_server", config)
        assert client is not None

    @patch("strands.experimental.mcp_config.sse_client")
    def test_url_without_transport_defaults_to_sse(self, mock_sse_client):
        config = {"url": "http://localhost:8000/sse"}
        client = _create_mcp_client_from_config("test_server", config)
        assert client is not None

    def test_prefix_passed_to_mcp_client(self):
        config = {"command": "echo", "prefix": "myprefix"}
        with (
            patch("strands.experimental.mcp_config.stdio_client"),
            patch("strands.experimental.mcp_config.StdioServerParameters"),
            patch("strands.experimental.mcp_config.MCPClient") as mock_mcp_client,
        ):
            _create_mcp_client_from_config("test_server", config)
            assert mock_mcp_client.call_args[1]["prefix"] == "myprefix"

    def test_startup_timeout_passed_to_mcp_client(self):
        config = {"command": "echo", "startup_timeout": 60}
        with (
            patch("strands.experimental.mcp_config.stdio_client"),
            patch("strands.experimental.mcp_config.StdioServerParameters"),
            patch("strands.experimental.mcp_config.MCPClient") as mock_mcp_client,
        ):
            _create_mcp_client_from_config("test_server", config)
            assert mock_mcp_client.call_args[1]["startup_timeout"] == 60

    def test_tool_filters_passed_to_mcp_client(self):
        config = {"command": "echo", "tool_filters": {"allowed": ["echo"], "rejected": ["dangerous_.*"]}}
        with (
            patch("strands.experimental.mcp_config.stdio_client"),
            patch("strands.experimental.mcp_config.StdioServerParameters"),
            patch("strands.experimental.mcp_config.MCPClient") as mock_mcp_client,
        ):
            _create_mcp_client_from_config("test_server", config)
            tool_filters = mock_mcp_client.call_args[1]["tool_filters"]
            assert "allowed" in tool_filters
            assert "rejected" in tool_filters

    def test_missing_command_and_url_raises_error(self):
        config = {"prefix": "test"}
        with pytest.raises(ValueError, match="must specify either 'command'.*or 'url'"):
            _create_mcp_client_from_config("test_server", config)

    def test_missing_url_for_sse_raises_error(self):
        config = {"transport": "sse"}
        with pytest.raises(ValueError, match="'url' is required"):
            _create_mcp_client_from_config("test_server", config)

    def test_invalid_transport_raises_error(self):
        config = {"transport": "websocket", "url": "ws://localhost:8000"}
        with pytest.raises(ValueError, match="configuration validation error"):
            _create_mcp_client_from_config("test_server", config)

    def test_invalid_startup_timeout_type_raises_error(self):
        config = {"command": "echo", "startup_timeout": "30"}
        with pytest.raises(ValueError, match="configuration validation error"):
            _create_mcp_client_from_config("test_server", config)


class TestLoadMcpClientsFromConfig:
    """Tests for load_mcp_clients_from_config function."""

    def test_load_from_dict(self):
        config = {
            "mcpServers": {
                "server1": {"command": "echo", "args": []},
                "server2": {"command": "cat", "args": []},
            }
        }
        with (
            patch("strands.experimental.mcp_config.stdio_client"),
            patch("strands.experimental.mcp_config.StdioServerParameters"),
            patch("strands.experimental.mcp_config.MCPClient") as mock_mcp_client,
        ):
            mock_mcp_client.return_value = MagicMock()
            clients = load_mcp_clients_from_config(config)
            assert len(clients) == 2

    def test_load_from_json_file(self):
        config_data = {"mcpServers": {"my_server": {"command": "echo", "args": ["hello"]}}}
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(config_data, f)
                temp_path = f.name
            with (
                patch("strands.experimental.mcp_config.stdio_client"),
                patch("strands.experimental.mcp_config.StdioServerParameters"),
                patch("strands.experimental.mcp_config.MCPClient") as mock_mcp_client,
            ):
                mock_mcp_client.return_value = MagicMock()
                clients = load_mcp_clients_from_config(temp_path)
                assert len(clients) == 1
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_file_not_found_raises_error(self):
        with pytest.raises(FileNotFoundError):
            load_mcp_clients_from_config("/nonexistent/path/config.json")

    def test_invalid_json_raises_error(self):
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                f.write("not json")
                temp_path = f.name
            with pytest.raises(json.JSONDecodeError):
                load_mcp_clients_from_config(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_invalid_config_type_raises_error(self):
        with pytest.raises(ValueError, match="must be a file path string or dictionary"):
            load_mcp_clients_from_config(123)  # type: ignore[arg-type]

    def test_empty_mcp_servers_returns_empty(self):
        clients = load_mcp_clients_from_config({"mcpServers": {}})
        assert clients == []

    def test_missing_mcp_servers_key_raises_error(self):
        with pytest.raises(ValueError, match="mcpServers"):
            load_mcp_clients_from_config({"server1": {"command": "echo"}})

    def test_server_creation_error_includes_server_name(self):
        config = {"mcpServers": {"bad_server": {"transport": "invalid"}}}
        with pytest.raises(ValueError, match="bad_server"):
            load_mcp_clients_from_config(config)
