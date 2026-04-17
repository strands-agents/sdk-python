"""Tests for MCP configuration loading and MCPClient factory."""

import json
import os
import re
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from strands.tools.mcp.mcp_config import (
    _create_mcp_client,
    _detect_transport,
    _parse_tool_filters,
    load_mcp_clients_from_config,
)


class TestDetectTransport:
    """Tests for transport type auto-detection."""

    def test_explicit_stdio(self):
        assert _detect_transport({"transport": "stdio", "command": "echo"}) == "stdio"

    def test_explicit_sse(self):
        assert _detect_transport({"transport": "sse", "url": "http://localhost:8000/sse"}) == "sse"

    def test_explicit_streamable_http(self):
        assert _detect_transport({"transport": "streamable-http", "url": "http://localhost:8000/mcp"}) == "streamable-http"

    def test_infer_stdio_from_command(self):
        assert _detect_transport({"command": "uvx"}) == "stdio"

    def test_infer_streamable_http_from_url(self):
        assert _detect_transport({"url": "http://localhost:8000/mcp"}) == "streamable-http"

    def test_explicit_transport_overrides_command(self):
        """Explicit transport takes priority over command presence."""
        assert _detect_transport({"transport": "sse", "command": "echo", "url": "http://x"}) == "sse"

    def test_command_takes_priority_over_url(self):
        """When both command and url are present without explicit transport, stdio wins."""
        assert _detect_transport({"command": "echo", "url": "http://x"}) == "stdio"

    def test_no_transport_info_raises(self):
        with pytest.raises(ValueError, match="cannot determine transport type"):
            _detect_transport({})

    def test_no_transport_info_with_irrelevant_keys_raises(self):
        with pytest.raises(ValueError, match="cannot determine transport type"):
            _detect_transport({"prefix": "test"})


class TestParseToolFilters:
    """Tests for tool filter parsing."""

    def test_none_input(self):
        assert _parse_tool_filters(None) is None

    def test_empty_dict(self):
        assert _parse_tool_filters({}) is None

    def test_allowed_patterns(self):
        filters = _parse_tool_filters({"allowed": ["search_.*", "list_.*"]})
        assert filters is not None
        assert "allowed" in filters
        assert len(filters["allowed"]) == 2
        # Should be compiled regex patterns
        assert isinstance(filters["allowed"][0], re.Pattern)
        assert isinstance(filters["allowed"][1], re.Pattern)

    def test_rejected_patterns(self):
        filters = _parse_tool_filters({"rejected": ["dangerous_tool"]})
        assert filters is not None
        assert "rejected" in filters
        assert len(filters["rejected"]) == 1

    def test_both_allowed_and_rejected(self):
        filters = _parse_tool_filters({
            "allowed": ["search_.*"],
            "rejected": ["search_dangerous"],
        })
        assert filters is not None
        assert "allowed" in filters
        assert "rejected" in filters

    def test_invalid_regex_falls_back_to_string(self):
        """Invalid regex patterns should be kept as literal strings with a warning."""
        filters = _parse_tool_filters({"allowed": ["[invalid"]})
        assert filters is not None
        assert len(filters["allowed"]) == 1
        # Should be a string, not a compiled pattern
        assert isinstance(filters["allowed"][0], str)
        assert filters["allowed"][0] == "[invalid"

    def test_regex_pattern_compilation(self):
        filters = _parse_tool_filters({"allowed": ["^search_.*$"]})
        assert filters is not None
        pattern = filters["allowed"][0]
        assert isinstance(pattern, re.Pattern)
        assert pattern.match("search_docs")
        assert not pattern.match("list_docs")


class TestCreateMcpClient:
    """Tests for MCPClient creation from config."""

    @patch("mcp.client.stdio.stdio_client")
    def test_stdio_client(self, mock_stdio):
        mock_stdio.return_value = MagicMock()
        config = {"command": "echo", "args": ["hello"]}
        client = _create_mcp_client("test_server", config)
        assert client is not None

    @patch("mcp.client.sse.sse_client")
    def test_sse_client(self, mock_sse):
        mock_sse.return_value = MagicMock()
        config = {"transport": "sse", "url": "http://localhost:8000/sse"}
        client = _create_mcp_client("test_server", config)
        assert client is not None

    @patch("mcp.client.streamable_http.streamablehttp_client")
    def test_streamable_http_client(self, mock_http):
        mock_http.return_value = MagicMock()
        config = {"transport": "streamable-http", "url": "http://localhost:8000/mcp"}
        client = _create_mcp_client("test_server", config)
        assert client is not None

    @patch("mcp.client.stdio.stdio_client")
    def test_client_with_prefix(self, mock_stdio):
        mock_stdio.return_value = MagicMock()
        config = {"command": "echo", "prefix": "test"}
        client = _create_mcp_client("test_server", config)
        assert client._prefix == "test"

    @patch("mcp.client.stdio.stdio_client")
    def test_client_with_startup_timeout(self, mock_stdio):
        mock_stdio.return_value = MagicMock()
        config = {"command": "echo", "startup_timeout": 60}
        client = _create_mcp_client("test_server", config)
        assert client._startup_timeout == 60

    @patch("mcp.client.stdio.stdio_client")
    def test_client_with_tool_filters(self, mock_stdio):
        mock_stdio.return_value = MagicMock()
        config = {
            "command": "echo",
            "tool_filters": {"allowed": ["search_.*"], "rejected": ["dangerous"]},
        }
        client = _create_mcp_client("test_server", config)
        assert client._tool_filters is not None

    def test_unsupported_transport_raises(self):
        with pytest.raises(ValueError, match="cannot determine transport type"):
            _create_mcp_client("test_server", {})


class TestLoadMcpClientsFromConfig:
    """Tests for loading MCPClient instances from configuration."""

    @patch("strands.tools.mcp.mcp_config._create_mcp_client")
    def test_load_from_dict(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        config = {
            "server_a": {"command": "echo", "args": ["hello"]},
            "server_b": {"transport": "sse", "url": "http://localhost:8000/sse"},
        }
        clients = load_mcp_clients_from_config(config)

        assert len(clients) == 2
        assert "server_a" in clients
        assert "server_b" in clients
        assert mock_create.call_count == 2

    @patch("strands.tools.mcp.mcp_config._create_mcp_client")
    def test_load_from_file(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        config_data = {
            "test_server": {"command": "echo"},
        }
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
                json.dump(config_data, f)
                f.flush()
                temp_path = f.name

            clients = load_mcp_clients_from_config(temp_path)
            assert len(clients) == 1
            assert "test_server" in clients
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @patch("strands.tools.mcp.mcp_config._create_mcp_client")
    def test_load_from_file_with_mcp_servers_key(self, mock_create):
        """Support files where servers are nested under mcp_servers key."""
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        config_data = {
            "mcp_servers": {
                "test_server": {"command": "echo"},
            }
        }
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
                json.dump(config_data, f)
                f.flush()
                temp_path = f.name

            clients = load_mcp_clients_from_config(temp_path)
            assert len(clients) == 1
            assert "test_server" in clients
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="MCP configuration file not found"):
            load_mcp_clients_from_config("/nonexistent/mcp_config.json")

    def test_load_invalid_json_file(self):
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
                f.write("not valid json")
                temp_path = f.name

            with pytest.raises(json.JSONDecodeError):
                load_mcp_clients_from_config(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_invalid_config_type(self):
        with pytest.raises(ValueError, match="config must be a file path string or dictionary"):
            load_mcp_clients_from_config(123)

    def test_load_empty_config(self):
        clients = load_mcp_clients_from_config({})
        assert len(clients) == 0

    def test_validation_error_invalid_server_config(self):
        config = {
            "bad_server": {"invalid_field": "value"},
        }
        with pytest.raises(ValueError, match="MCP configuration validation error"):
            load_mcp_clients_from_config(config)

    def test_validation_error_wrong_type(self):
        config = {
            "bad_server": {"command": 123},
        }
        with pytest.raises(ValueError, match="MCP configuration validation error"):
            load_mcp_clients_from_config(config)

    @patch("strands.tools.mcp.mcp_config._create_mcp_client")
    def test_create_failure_wraps_error(self, mock_create):
        mock_create.side_effect = RuntimeError("transport init failed")

        config = {"test_server": {"command": "echo"}}
        with pytest.raises(ValueError, match="failed to create MCP client for server 'test_server'"):
            load_mcp_clients_from_config(config)


class TestStdioTransport:
    """Tests for stdio transport creation."""

    @patch("mcp.client.stdio.stdio_client")
    def test_create_stdio_with_all_options(self, mock_stdio):
        from strands.tools.mcp.mcp_config import _create_stdio_transport

        config = {
            "command": "uvx",
            "args": ["my-server"],
            "env": {"KEY": "value"},
            "cwd": "/tmp",
        }
        transport_fn = _create_stdio_transport(config)
        assert callable(transport_fn)

    def test_create_stdio_missing_command(self):
        from strands.tools.mcp.mcp_config import _create_stdio_transport

        with pytest.raises(ValueError, match="'command' is required"):
            _create_stdio_transport({})


class TestSseTransport:
    """Tests for SSE transport creation."""

    @patch("mcp.client.sse.sse_client")
    def test_create_sse_with_headers(self, mock_sse):
        from strands.tools.mcp.mcp_config import _create_sse_transport

        config = {
            "url": "http://localhost:8000/sse",
            "headers": {"Authorization": "[REDACTED_TOKEN]"},
        }
        transport_fn = _create_sse_transport(config)
        assert callable(transport_fn)

    def test_create_sse_missing_url(self):
        from strands.tools.mcp.mcp_config import _create_sse_transport

        with pytest.raises(ValueError, match="'url' is required"):
            _create_sse_transport({})


class TestStreamableHttpTransport:
    """Tests for streamable-http transport creation."""

    @patch("mcp.client.streamable_http.streamablehttp_client")
    def test_create_http_with_headers(self, mock_http):
        from strands.tools.mcp.mcp_config import _create_streamable_http_transport

        config = {
            "url": "http://localhost:8000/mcp",
            "headers": {"Authorization": "[REDACTED_TOKEN]"},
        }
        transport_fn = _create_streamable_http_transport(config)
        assert callable(transport_fn)

    def test_create_http_missing_url(self):
        from strands.tools.mcp.mcp_config import _create_streamable_http_transport

        with pytest.raises(ValueError, match="'url' is required"):
            _create_streamable_http_transport({})
