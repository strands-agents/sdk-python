"""Tests for MCP config parsing and MCPClient factory."""

import json
import os
import re
import tempfile
from unittest.mock import MagicMock, patch

import jsonschema
import pytest

from strands.experimental.mcp_config import (
    MCP_SERVER_CONFIG_SCHEMA,
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
        assert isinstance(result["allowed"][1], re.Pattern)
        assert result["allowed"][0].match("echo")
        assert result["allowed"][1].match("search")

    def test_compiles_rejected_patterns(self):
        config = {"rejected": ["dangerous_tool"]}
        result = _parse_tool_filters(config)
        assert result is not None
        assert len(result["rejected"]) == 1
        assert isinstance(result["rejected"][0], re.Pattern)
        assert result["rejected"][0].match("dangerous_tool")

    def test_compiles_regex_patterns(self):
        config = {"allowed": ["search_.*", "get_\\w+"]}
        result = _parse_tool_filters(config)
        assert result is not None
        assert len(result["allowed"]) == 2
        assert result["allowed"][0].match("search_docs")
        assert result["allowed"][1].match("get_data")

    def test_exact_strings_work_as_regex(self):
        """Exact strings like 'echo' match themselves when compiled as regex."""
        config = {"allowed": ["echo"]}
        result = _parse_tool_filters(config)
        assert result is not None
        assert result["allowed"][0].match("echo")
        assert result["allowed"][0].match("echo_extra")

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

    def test_invalid_regex_in_rejected_raises_error(self):
        config = {"rejected": ["(unclosed"]}
        with pytest.raises(ValueError, match="invalid regex pattern"):
            _parse_tool_filters(config)


class TestCreateMcpClientFromConfig:
    """Tests for _create_mcp_client_from_config function."""

    @patch("strands.experimental.mcp_config.stdio_client")
    @patch("strands.experimental.mcp_config.StdioServerParameters")
    def test_stdio_transport_from_command(self, mock_params_cls, mock_stdio_client):
        """Config with 'command' should create a stdio transport."""
        config = {"command": "uvx", "args": ["some-server@latest"]}
        mock_params_cls.return_value = MagicMock()

        client = _create_mcp_client_from_config("test_server", config)
        assert client is not None

    @patch("strands.experimental.mcp_config.stdio_client")
    @patch("strands.experimental.mcp_config.StdioServerParameters")
    def test_stdio_with_env_and_cwd(self, mock_params_cls, mock_stdio_client):
        """Stdio config should pass env and cwd to StdioServerParameters."""
        config = {
            "command": "node",
            "args": ["server.js"],
            "env": {"NODE_ENV": "production"},
            "cwd": "/opt/server",
        }
        mock_params_cls.return_value = MagicMock()

        client = _create_mcp_client_from_config("test_server", config)

        client._transport_callable()
        mock_params_cls.assert_called_once_with(
            command="node",
            args=["server.js"],
            env={"NODE_ENV": "production"},
            cwd="/opt/server",
        )

    @patch("strands.experimental.mcp_config.sse_client")
    def test_sse_transport_explicit(self, mock_sse_client):
        """Config with transport='sse' should create an SSE transport."""
        config = {"transport": "sse", "url": "http://localhost:8000/sse"}

        client = _create_mcp_client_from_config("test_server", config)
        assert client is not None

    @patch("strands.experimental.mcp_config.streamable_http_client")
    def test_streamable_http_transport_explicit(self, mock_http_client):
        """Config with transport='streamable-http' should create a streamable-http transport."""
        config = {"transport": "streamable-http", "url": "http://localhost:8000/mcp"}

        client = _create_mcp_client_from_config("test_server", config)
        assert client is not None

    @patch("strands.experimental.mcp_config.sse_client")
    def test_url_without_transport_defaults_to_sse(self, mock_sse_client):
        """Config with url but no transport should default to sse."""
        config = {"url": "http://localhost:8000/sse"}

        client = _create_mcp_client_from_config("test_server", config)
        assert client is not None

    @patch("strands.experimental.mcp_config.sse_client")
    def test_sse_with_headers(self, mock_sse_client):
        """SSE config should pass headers."""
        config = {
            "transport": "sse",
            "url": "http://localhost:8000/sse",
            "headers": {"Authorization": "Bearer token123"},
        }

        client = _create_mcp_client_from_config("test_server", config)
        assert client is not None

    @patch("strands.experimental.mcp_config.streamable_http_client")
    def test_streamable_http_with_headers(self, mock_http_client):
        """Streamable HTTP config should pass headers."""
        config = {
            "transport": "streamable-http",
            "url": "http://localhost:8000/mcp",
            "headers": {"Authorization": "Bearer token123"},
        }

        client = _create_mcp_client_from_config("test_server", config)
        assert client is not None

    def test_prefix_passed_to_mcp_client(self):
        """Prefix should be passed to MCPClient constructor."""
        config = {"command": "echo", "prefix": "myprefix"}

        with (
            patch("strands.experimental.mcp_config.stdio_client"),
            patch("strands.experimental.mcp_config.StdioServerParameters"),
            patch("strands.experimental.mcp_config.MCPClient") as mock_mcp_client,
        ):
            _create_mcp_client_from_config("test_server", config)
            call_kwargs = mock_mcp_client.call_args
            assert call_kwargs[1]["prefix"] == "myprefix"

    def test_startup_timeout_passed_to_mcp_client(self):
        """startup_timeout should be passed to MCPClient constructor."""
        config = {"command": "echo", "startup_timeout": 60}

        with (
            patch("strands.experimental.mcp_config.stdio_client"),
            patch("strands.experimental.mcp_config.StdioServerParameters"),
            patch("strands.experimental.mcp_config.MCPClient") as mock_mcp_client,
        ):
            _create_mcp_client_from_config("test_server", config)
            call_kwargs = mock_mcp_client.call_args
            assert call_kwargs[1]["startup_timeout"] == 60

    def test_tool_filters_passed_to_mcp_client(self):
        """tool_filters config should be parsed and passed to MCPClient."""
        config = {
            "command": "echo",
            "tool_filters": {"allowed": ["echo"], "rejected": ["dangerous_.*"]},
        }

        with (
            patch("strands.experimental.mcp_config.stdio_client"),
            patch("strands.experimental.mcp_config.StdioServerParameters"),
            patch("strands.experimental.mcp_config.MCPClient") as mock_mcp_client,
        ):
            _create_mcp_client_from_config("test_server", config)
            call_kwargs = mock_mcp_client.call_args
            tool_filters = call_kwargs[1]["tool_filters"]
            assert "allowed" in tool_filters
            assert "rejected" in tool_filters

    def test_default_startup_timeout(self):
        """Default startup_timeout should be 30 if not specified."""
        config = {"command": "echo"}

        with (
            patch("strands.experimental.mcp_config.stdio_client"),
            patch("strands.experimental.mcp_config.StdioServerParameters"),
            patch("strands.experimental.mcp_config.MCPClient") as mock_mcp_client,
        ):
            _create_mcp_client_from_config("test_server", config)
            call_kwargs = mock_mcp_client.call_args
            assert call_kwargs[1]["startup_timeout"] == 30

    def test_invalid_transport_raises_error(self):
        """Unknown transport type should raise ValueError."""
        config = {"transport": "websocket", "url": "ws://localhost:8000"}

        with pytest.raises(ValueError, match="configuration validation error"):
            _create_mcp_client_from_config("test_server", config)

    def test_missing_command_and_url_raises_error(self):
        """Config without command or url should raise ValueError."""
        config = {"prefix": "test"}

        with pytest.raises(ValueError, match="must specify either 'command'.*or 'url'"):
            _create_mcp_client_from_config("test_server", config)

    def test_missing_url_for_sse_raises_error(self):
        """SSE transport without url should raise ValueError."""
        config = {"transport": "sse"}

        with pytest.raises(ValueError, match="'url' is required"):
            _create_mcp_client_from_config("test_server", config)

    def test_missing_url_for_streamable_http_raises_error(self):
        """Streamable HTTP transport without url should raise ValueError."""
        config = {"transport": "streamable-http"}

        with pytest.raises(ValueError, match="'url' is required"):
            _create_mcp_client_from_config("test_server", config)

    def test_invalid_startup_timeout_type_raises_error(self):
        """Non-integer startup_timeout should raise ValueError."""
        config = {"command": "echo", "startup_timeout": "30"}

        with pytest.raises(ValueError, match="configuration validation error"):
            _create_mcp_client_from_config("test_server", config)


class TestLoadMcpClientsFromConfig:
    """Tests for load_mcp_clients_from_config function."""

    def test_load_from_dict(self):
        """Loading from a dict with mcpServers wrapper."""
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
            assert "server1" in clients
            assert "server2" in clients

    def test_load_from_json_file(self):
        """Loading from a JSON file path."""
        config_data = {
            "mcpServers": {
                "my_server": {"command": "echo", "args": ["hello"]},
            }
        }
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
                assert "my_server" in clients
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_from_file_with_prefix(self):
        """Loading from a file:// prefixed path."""
        config_data = {"mcpServers": {"server1": {"command": "echo"}}}
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
                clients = load_mcp_clients_from_config(f"file://{temp_path}")

                assert len(clients) == 1
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_file_not_found_raises_error(self):
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_mcp_clients_from_config("/nonexistent/path/config.json")

    def test_invalid_json_raises_error(self):
        """Invalid JSON content should raise JSONDecodeError."""
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
        """Non-string, non-dict input should raise ValueError."""
        with pytest.raises(ValueError, match="must be a file path string or dictionary"):
            load_mcp_clients_from_config(123)  # type: ignore[arg-type]

    def test_empty_mcp_servers_returns_empty(self):
        """Empty mcpServers dict should return empty dict."""
        clients = load_mcp_clients_from_config({"mcpServers": {}})
        assert clients == {}

    def test_missing_mcp_servers_key_raises_error(self):
        """Dict without mcpServers key should raise ValueError."""
        with pytest.raises(ValueError, match="mcpServers"):
            load_mcp_clients_from_config({"server1": {"command": "echo"}})

    def test_server_creation_error_includes_server_name(self):
        """Error creating a server should include the server name in the message."""
        config = {"mcpServers": {"bad_server": {"transport": "invalid"}}}

        with pytest.raises(ValueError, match="bad_server"):
            load_mcp_clients_from_config(config)


class TestMcpServerConfigSchema:
    """Tests for MCP_SERVER_CONFIG_SCHEMA validation."""

    def _validate(self, config):
        jsonschema.Draft7Validator(MCP_SERVER_CONFIG_SCHEMA).validate(config)

    def test_valid_stdio_config(self):
        self._validate({"command": "uvx", "args": ["server@latest"], "env": {"KEY": "val"}, "cwd": "/tmp"})

    def test_valid_sse_config(self):
        self._validate({"transport": "sse", "url": "http://localhost:8000/sse", "headers": {"Auth": "Bearer tok"}})

    def test_valid_streamable_http_config(self):
        self._validate({"transport": "streamable-http", "url": "http://localhost:8000/mcp"})

    def test_valid_config_with_all_common_fields(self):
        self._validate({
            "command": "echo",
            "prefix": "myprefix",
            "startup_timeout": 60,
            "tool_filters": {"allowed": ["echo"], "rejected": ["dangerous_.*"]},
        })

    def test_rejects_unknown_property(self):
        with pytest.raises(jsonschema.ValidationError):
            self._validate({"command": "echo", "unknown_field": "value"})

    def test_rejects_invalid_transport_enum(self):
        with pytest.raises(jsonschema.ValidationError):
            self._validate({"transport": "websocket", "url": "ws://localhost"})

    def test_rejects_non_string_command(self):
        with pytest.raises(jsonschema.ValidationError):
            self._validate({"command": 123})

    def test_rejects_non_array_args(self):
        with pytest.raises(jsonschema.ValidationError):
            self._validate({"command": "echo", "args": "not-an-array"})

    def test_rejects_non_string_args_items(self):
        with pytest.raises(jsonschema.ValidationError):
            self._validate({"command": "echo", "args": [123]})

    def test_rejects_non_object_env(self):
        with pytest.raises(jsonschema.ValidationError):
            self._validate({"command": "echo", "env": "not-an-object"})

    def test_rejects_non_string_env_values(self):
        with pytest.raises(jsonschema.ValidationError):
            self._validate({"command": "echo", "env": {"KEY": 123}})

    def test_rejects_non_integer_startup_timeout(self):
        with pytest.raises(jsonschema.ValidationError):
            self._validate({"command": "echo", "startup_timeout": "30"})

    def test_rejects_unknown_tool_filter_property(self):
        with pytest.raises(jsonschema.ValidationError):
            self._validate({"command": "echo", "tool_filters": {"allowed": ["echo"], "extra": True}})

    def test_rejects_non_string_header_values(self):
        with pytest.raises(jsonschema.ValidationError):
            self._validate({"transport": "sse", "url": "http://localhost", "headers": {"Key": 123}})

    def test_schema_is_valid_json_schema(self):
        """The schema itself should be a valid JSON Schema draft-07."""
        jsonschema.Draft7Validator.check_schema(MCP_SERVER_CONFIG_SCHEMA)
