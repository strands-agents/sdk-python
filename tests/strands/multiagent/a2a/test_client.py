"""Tests for the A2A client module."""

from unittest.mock import MagicMock, patch

import httpx
import pytest
from strands.multiagent.a2a.client import (
    A2AClient,
    A2AError,
    build_agentcore_url,
    extract_region_from_arn,
)


class TestBuildAgentcoreUrl:
    """Tests for build_agentcore_url function."""

    def test_valid_arn(self):
        """Test URL building with a valid ARN."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/my-agent"
        url = build_agentcore_url(arn)

        assert url.startswith("https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/")
        assert url.endswith("/invocations")
        assert "arn%3Aaws%3Abedrock-agentcore" in url

    def test_invalid_arn_prefix(self):
        """Test that invalid ARN prefix raises ValueError."""
        with pytest.raises(ValueError, match="Invalid AgentCore ARN format"):
            build_agentcore_url("arn:aws:lambda:us-east-1:123456789012:function/my-function")

    def test_invalid_arn_format(self):
        """Test that malformed ARN raises ValueError."""
        with pytest.raises(ValueError, match="Invalid"):
            build_agentcore_url("arn:aws:bedrock-agentcore")


class TestExtractRegionFromArn:
    """Tests for extract_region_from_arn function."""

    def test_valid_arn(self):
        """Test region extraction from a valid ARN."""
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/my-agent"
        assert extract_region_from_arn(arn) == "us-east-1"

    def test_invalid_arn_format(self):
        """Test that malformed ARN raises ValueError."""
        with pytest.raises(ValueError, match="Invalid ARN format"):
            extract_region_from_arn("invalid-arn")


class TestA2AClientInitialization:
    """Tests for A2AClient initialization."""

    def test_init_with_url(self):
        """Test basic initialization with URL."""
        client = A2AClient(url="http://localhost:9000")

        assert client.url == "http://localhost:9000"
        assert client._auth is None
        assert client._timeout == 300.0

    def test_init_strips_trailing_slash(self):
        """Test that trailing slashes are stripped from URL."""
        client = A2AClient(url="http://localhost:9000/")
        assert client.url == "http://localhost:9000"

    def test_init_with_auth_and_headers(self):
        """Test initialization with authentication and headers."""
        mock_auth = MagicMock(spec=httpx.Auth)
        headers = {"X-Custom-Header": "value"}
        client = A2AClient(url="http://localhost:9000", auth=mock_auth, timeout=600.0, headers=headers)

        assert client._auth == mock_auth
        assert client._timeout == 600.0
        assert client._headers == headers


class TestA2AClientFromUrl:
    """Tests for A2AClient.from_url factory method."""

    def test_from_url_basic(self):
        """Test creating client from URL."""
        client = A2AClient.from_url("http://localhost:9000")

        assert client.url == "http://localhost:9000"
        assert client._auth is None


class TestA2AClientRequestBuilding:
    """Tests for A2A request building methods."""

    def test_build_task_request(self):
        """Test building a task/send request."""
        client = A2AClient(url="http://localhost:9000")
        request = client._build_task_request("task-123", "session-456", "Hello")

        assert request["jsonrpc"] == "2.0"
        assert request["method"] == "tasks/send"
        assert request["id"] == "task-123"
        assert request["params"]["id"] == "task-123"
        assert request["params"]["sessionId"] == "session-456"
        assert request["params"]["message"]["role"] == "user"
        assert request["params"]["message"]["parts"][0]["text"] == "Hello"


class TestA2AClientResponseParsing:
    """Tests for A2A response parsing methods."""

    def test_extract_text_from_response_success(self):
        """Test extracting text from a successful response."""
        client = A2AClient(url="http://localhost:9000")
        response = {
            "jsonrpc": "2.0",
            "id": "task-123",
            "result": {"artifacts": [{"parts": [{"kind": "text", "text": "Hello World"}]}]},
        }
        assert client._extract_text_from_response(response) == "Hello World"

    def test_extract_text_from_response_error(self):
        """Test that error response raises A2AError."""
        client = A2AClient(url="http://localhost:9000")
        response = {
            "jsonrpc": "2.0",
            "id": "task-123",
            "error": {"code": -32600, "message": "Invalid Request"},
        }

        with pytest.raises(A2AError) as exc_info:
            client._extract_text_from_response(response)

        assert exc_info.value.code == -32600

    def test_extract_text_from_response_empty(self):
        """Test extracting text from response with no artifacts."""
        client = A2AClient(url="http://localhost:9000")
        response = {"jsonrpc": "2.0", "id": "task-123", "result": {}}
        assert client._extract_text_from_response(response) == ""


class TestA2AClientMethods:
    """Tests for A2AClient methods."""

    def test_get_agent_card(self):
        """Test getting agent card."""
        client = A2AClient(url="http://localhost:9000")
        expected_card = {"name": "Test Agent"}

        mock_response = MagicMock()
        mock_response.json.return_value = expected_card
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_class:
            mock_http_client = MagicMock()
            mock_http_client.get.return_value = mock_response
            mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
            mock_http_client.__exit__ = MagicMock(return_value=None)
            mock_client_class.return_value = mock_http_client

            assert client.get_agent_card() == expected_card

    def test_get_agent_card_cached(self):
        """Test that agent card is cached."""
        client = A2AClient(url="http://localhost:9000")
        client._agent_card = {"name": "Cached Agent"}
        assert client.get_agent_card() == {"name": "Cached Agent"}

    def test_send_task(self):
        """Test sending a task."""
        client = A2AClient(url="http://localhost:9000")
        response_data = {
            "jsonrpc": "2.0",
            "id": "task-123",
            "result": {"artifacts": [{"parts": [{"kind": "text", "text": "8"}]}]},
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_class:
            mock_http_client = MagicMock()
            mock_http_client.post.return_value = mock_response
            mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
            mock_http_client.__exit__ = MagicMock(return_value=None)
            mock_client_class.return_value = mock_http_client

            assert client.send_task("Calculate 3 + 5") == "8"


class TestA2AError:
    """Tests for A2AError exception."""

    def test_a2a_error(self):
        """Test A2AError has correct attributes and string representation."""
        error = A2AError(code=-32600, message="Invalid Request")

        assert error.code == -32600
        assert error.message == "Invalid Request"
        assert str(error) == "A2A Error -32600: Invalid Request"
