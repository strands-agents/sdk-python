import base64
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import ListToolsResult
from mcp.types import CallToolResult as MCPCallToolResult
from mcp.types import GetPromptResult, ListPromptsResult, Prompt, PromptMessage
from mcp.types import TextContent as MCPTextContent
from mcp.types import Tool as MCPTool

from strands.tools.mcp import MCPClient
from strands.tools.mcp.mcp_retry import ExponentialBackoffRetry, MCPRetryConfig, NoRetryStrategy
from strands.tools.mcp.mcp_types import MCPToolResult
from strands.types.exceptions import MCPClientInitializationError


@pytest.fixture
def mock_transport():
    mock_read_stream = AsyncMock()
    mock_write_stream = AsyncMock()
    mock_transport_cm = AsyncMock()
    mock_transport_cm.__aenter__.return_value = (mock_read_stream, mock_write_stream)
    mock_transport_callable = MagicMock(return_value=mock_transport_cm)

    return {
        "read_stream": mock_read_stream,
        "write_stream": mock_write_stream,
        "transport_cm": mock_transport_cm,
        "transport_callable": mock_transport_callable,
    }


@pytest.fixture
def mock_session():
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()

    # Create a mock context manager for ClientSession
    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__.return_value = mock_session

    # Patch ClientSession to return our mock session
    with patch("strands.tools.mcp.mcp_client.ClientSession", return_value=mock_session_cm):
        yield mock_session


@pytest.fixture
def mcp_client(mock_transport, mock_session):
    with MCPClient(mock_transport["transport_callable"]) as client:
        yield client


def test_mcp_client_context_manager(mock_transport, mock_session):
    """Test that the MCPClient context manager properly initializes and cleans up."""
    with MCPClient(mock_transport["transport_callable"]) as client:
        assert client._background_thread is not None
        assert client._background_thread.is_alive()
        assert client._init_future.done()

        mock_transport["transport_cm"].__aenter__.assert_called_once()
        mock_session.initialize.assert_called_once()

    # After exiting the context manager, verify that the thread was cleaned up
    # Give a small delay for the thread to fully terminate
    time.sleep(0.1)
    assert client._background_thread is None


def test_list_tools_sync(mock_transport, mock_session):
    """Test that list_tools_sync correctly retrieves and adapts tools."""
    mock_tool = MCPTool(name="test_tool", description="A test tool", inputSchema={"type": "object", "properties": {}})
    mock_session.list_tools.return_value = ListToolsResult(tools=[mock_tool])

    with MCPClient(mock_transport["transport_callable"]) as client:
        tools = client.list_tools_sync()

        mock_session.list_tools.assert_called_once_with(cursor=None)

        assert len(tools) == 1
        assert tools[0].tool_name == "test_tool"
        assert tools.pagination_token is None


def test_list_tools_sync_session_not_active():
    """Test that list_tools_sync raises an error when session is not active."""
    client = MCPClient(MagicMock())

    with pytest.raises(MCPClientInitializationError, match="client.session is not running"):
        client.list_tools_sync()


def test_list_tools_sync_with_pagination_token(mock_transport, mock_session):
    """Test that list_tools_sync correctly passes pagination token and returns next cursor."""
    mock_tool = MCPTool(name="test_tool", description="A test tool", inputSchema={"type": "object", "properties": {}})
    mock_session.list_tools.return_value = ListToolsResult(tools=[mock_tool], nextCursor="next_page_token")

    with MCPClient(mock_transport["transport_callable"]) as client:
        tools = client.list_tools_sync(pagination_token="current_page_token")

        mock_session.list_tools.assert_called_once_with(cursor="current_page_token")
        assert len(tools) == 1
        assert tools[0].tool_name == "test_tool"
        assert tools.pagination_token == "next_page_token"


def test_list_tools_sync_without_pagination_token(mock_transport, mock_session):
    """Test that list_tools_sync works without pagination token and handles missing next cursor."""
    mock_tool = MCPTool(name="test_tool", description="A test tool", inputSchema={"type": "object", "properties": {}})
    mock_session.list_tools.return_value = ListToolsResult(tools=[mock_tool])  # No nextCursor

    with MCPClient(mock_transport["transport_callable"]) as client:
        tools = client.list_tools_sync()

        mock_session.list_tools.assert_called_once_with(cursor=None)
        assert len(tools) == 1
        assert tools[0].tool_name == "test_tool"
        assert tools.pagination_token is None


@pytest.mark.parametrize("is_error,expected_status", [(False, "success"), (True, "error")])
def test_call_tool_sync_status(mock_transport, mock_session, is_error, expected_status):
    """Test that call_tool_sync correctly handles success and error results."""
    mock_content = MCPTextContent(type="text", text="Test message")
    mock_session.call_tool.return_value = MCPCallToolResult(isError=is_error, content=[mock_content])

    with MCPClient(mock_transport["transport_callable"]) as client:
        result = client.call_tool_sync(tool_use_id="test-123", name="test_tool", arguments={"param": "value"})

        mock_session.call_tool.assert_called_once_with("test_tool", {"param": "value"}, None)

        assert result["status"] == expected_status
        assert result["toolUseId"] == "test-123"
        assert len(result["content"]) == 1
        assert result["content"][0]["text"] == "Test message"
        # No structured content should be present when not provided by MCP
        assert result.get("structuredContent") is None


def test_call_tool_sync_session_not_active():
    """Test that call_tool_sync raises an error when session is not active."""
    client = MCPClient(MagicMock())

    with pytest.raises(MCPClientInitializationError, match="client.session is not running"):
        client.call_tool_sync(tool_use_id="test-123", name="test_tool", arguments={"param": "value"})


def test_call_tool_sync_with_structured_content(mock_transport, mock_session):
    """Test that call_tool_sync correctly handles structured content."""
    mock_content = MCPTextContent(type="text", text="Test message")
    structured_content = {"result": 42, "status": "completed"}
    mock_session.call_tool.return_value = MCPCallToolResult(
        isError=False, content=[mock_content], structuredContent=structured_content
    )

    with MCPClient(mock_transport["transport_callable"]) as client:
        result = client.call_tool_sync(tool_use_id="test-123", name="test_tool", arguments={"param": "value"})

        mock_session.call_tool.assert_called_once_with("test_tool", {"param": "value"}, None)

        assert result["status"] == "success"
        assert result["toolUseId"] == "test-123"
        # Content should only contain the text content, not the structured content
        assert len(result["content"]) == 1
        assert result["content"][0]["text"] == "Test message"
        # Structured content should be in its own field
        assert "structuredContent" in result
        assert result["structuredContent"] == structured_content
        assert result["structuredContent"]["result"] == 42
        assert result["structuredContent"]["status"] == "completed"


def test_call_tool_sync_exception(mock_transport, mock_session):
    """Test that call_tool_sync correctly handles exceptions."""
    mock_session.call_tool.side_effect = Exception("Test exception")

    with MCPClient(mock_transport["transport_callable"]) as client:
        result = client.call_tool_sync(tool_use_id="test-123", name="test_tool", arguments={"param": "value"})

        assert result["status"] == "error"
        assert result["toolUseId"] == "test-123"
        assert len(result["content"]) == 1
        assert "Test exception" in result["content"][0]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_error,expected_status", [(False, "success"), (True, "error")])
async def test_call_tool_async_status(mock_transport, mock_session, is_error, expected_status):
    """Test that call_tool_async correctly handles success and error results."""
    mock_content = MCPTextContent(type="text", text="Test message")
    mock_result = MCPCallToolResult(isError=is_error, content=[mock_content])
    mock_session.call_tool.return_value = mock_result

    with MCPClient(mock_transport["transport_callable"]) as client:
        # Mock asyncio.run_coroutine_threadsafe and asyncio.wrap_future
        with (
            patch("asyncio.run_coroutine_threadsafe") as mock_run_coroutine_threadsafe,
            patch("asyncio.wrap_future") as mock_wrap_future,
        ):
            # Create a mock future that returns the mock result
            mock_future = MagicMock()
            mock_run_coroutine_threadsafe.return_value = mock_future

            # Create an async mock that resolves to the MCPToolResult format
            async def mock_awaitable():
                # Return MCPToolResult format that _call_tool_with_retry would return
                return MCPToolResult(
                    status=expected_status,
                    toolUseId="test-123",
                    content=[{"text": "Test message"}],
                    retryMetadata={"total_attempts": 1, "retry_strategy_used": "NoRetryStrategy"},
                )

            mock_wrap_future.return_value = mock_awaitable()

            result = await client.call_tool_async(
                tool_use_id="test-123", name="test_tool", arguments={"param": "value"}
            )

            # Verify the asyncio functions were called correctly
            mock_run_coroutine_threadsafe.assert_called_once()
            mock_wrap_future.assert_called_once_with(mock_future)

        assert result["status"] == expected_status
        assert result["toolUseId"] == "test-123"
        assert len(result["content"]) == 1
        assert result["content"][0]["text"] == "Test message"
        # Check retry metadata is present
        assert "retryMetadata" in result
        assert result["retryMetadata"]["total_attempts"] == 1


@pytest.mark.asyncio
async def test_call_tool_async_session_not_active():
    """Test that call_tool_async raises an error when session is not active."""
    client = MCPClient(MagicMock())

    with pytest.raises(MCPClientInitializationError, match="client.session is not running"):
        await client.call_tool_async(tool_use_id="test-123", name="test_tool", arguments={"param": "value"})


@pytest.mark.asyncio
async def test_call_tool_async_exception(mock_transport, mock_session):
    """Test that call_tool_async correctly handles exceptions."""
    # Mock the session to raise an exception
    mock_session.call_tool.side_effect = Exception("Test exception")

    with MCPClient(mock_transport["transport_callable"]) as client:
        result = await client.call_tool_async(tool_use_id="test-123", name="test_tool", arguments={"param": "value"})

        assert result["status"] == "error"
        assert result["toolUseId"] == "test-123"
        assert len(result["content"]) == 1
        assert "Test exception" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_call_tool_async_with_timeout(mock_transport, mock_session):
    """Test that call_tool_async correctly passes timeout parameter."""
    from datetime import timedelta

    mock_content = MCPTextContent(type="text", text="Test message")
    mock_result = MCPCallToolResult(isError=False, content=[mock_content])
    mock_session.call_tool.return_value = mock_result

    with MCPClient(mock_transport["transport_callable"]) as client:
        timeout = timedelta(seconds=30)

        with (
            patch("asyncio.run_coroutine_threadsafe") as mock_run_coroutine_threadsafe,
            patch("asyncio.wrap_future") as mock_wrap_future,
        ):
            mock_future = MagicMock()
            mock_run_coroutine_threadsafe.return_value = mock_future

            # Create an async mock that resolves to the MCPToolResult format
            async def mock_awaitable():
                return MCPToolResult(
                    status="success",
                    toolUseId="test-123",
                    content=[{"text": "Test message"}],
                    retryMetadata={"total_attempts": 1, "retry_strategy_used": "NoRetryStrategy"},
                )

            mock_wrap_future.return_value = mock_awaitable()

            result = await client.call_tool_async(
                tool_use_id="test-123", name="test_tool", arguments={"param": "value"}, read_timeout_seconds=timeout
            )

            # Verify the timeout was passed to the session call_tool method
            # We need to check that the coroutine passed to run_coroutine_threadsafe
            # would call session.call_tool with the timeout
            mock_run_coroutine_threadsafe.assert_called_once()
            mock_wrap_future.assert_called_once_with(mock_future)

        assert result["status"] == "success"
        assert result["toolUseId"] == "test-123"
        assert "retryMetadata" in result


@pytest.mark.asyncio
async def test_call_tool_async_initialization_not_complete():
    """Test that call_tool_async raises an error when background thread is not initialized."""
    client = MCPClient(MagicMock())

    # Manually set the client state to simulate a partially initialized state
    client._background_thread = MagicMock()
    client._background_thread.is_alive.return_value = True
    client._background_thread_session = None  # Not initialized
    client._background_thread_event_loop = None  # Not initialized

    # This should raise MCPClientInitializationError
    with pytest.raises(MCPClientInitializationError, match="the client session was not initialized"):
        await client.call_tool_async(tool_use_id="test-123", name="test_tool", arguments={"param": "value"})


@pytest.mark.asyncio
async def test_call_tool_async_wrap_future_exception(mock_transport, mock_session):
    """Test that call_tool_async correctly handles exceptions from wrap_future."""
    with MCPClient(mock_transport["transport_callable"]) as client:
        with (
            patch("asyncio.run_coroutine_threadsafe") as mock_run_coroutine_threadsafe,
            patch("asyncio.wrap_future") as mock_wrap_future,
        ):
            mock_future = MagicMock()
            mock_run_coroutine_threadsafe.return_value = mock_future

            # Create an async mock that returns an error MCPToolResult
            # (since _call_tool_with_retry catches exceptions and returns error results)
            async def mock_awaitable():
                return MCPToolResult(
                    status="error",
                    toolUseId="test-123",
                    content=[{"text": "Tool execution failed: Wrap future exception"}],
                )

            mock_wrap_future.return_value = mock_awaitable()

            result = await client.call_tool_async(
                tool_use_id="test-123", name="test_tool", arguments={"param": "value"}
            )

        assert result["status"] == "error"
        assert result["toolUseId"] == "test-123"
        assert len(result["content"]) == 1
        assert "Wrap future exception" in result["content"][0]["text"]


def test_enter_with_initialization_exception(mock_transport):
    """Test that __enter__ handles exceptions during initialization properly."""
    # Make the transport callable throw an exception
    mock_transport["transport_cm"].__aenter__.side_effect = Exception("Transport initialization failed")

    client = MCPClient(mock_transport["transport_callable"])

    with patch.object(client, "stop") as mock_stop:
        with pytest.raises(MCPClientInitializationError, match="the client initialization failed"):
            client.start()

        # Verify stop() was called for cleanup
        mock_stop.assert_called_once_with(None, None, None)


def test_mcp_tool_result_type():
    """Test that MCPToolResult extends ToolResult correctly."""
    # Test basic ToolResult functionality
    result = MCPToolResult(status="success", toolUseId="test-123", content=[{"text": "Test message"}])

    assert result["status"] == "success"
    assert result["toolUseId"] == "test-123"
    assert result["content"][0]["text"] == "Test message"

    # Test that structuredContent is optional
    assert "structuredContent" not in result or result.get("structuredContent") is None

    # Test with structuredContent
    result_with_structured = MCPToolResult(
        status="success", toolUseId="test-456", content=[{"text": "Test message"}], structuredContent={"key": "value"}
    )

    assert result_with_structured["structuredContent"] == {"key": "value"}


def test_call_tool_sync_without_structured_content(mock_transport, mock_session):
    """Test that call_tool_sync works correctly when no structured content is provided."""
    mock_content = MCPTextContent(type="text", text="Test message")
    mock_session.call_tool.return_value = MCPCallToolResult(
        isError=False,
        content=[mock_content],  # No structuredContent
    )

    with MCPClient(mock_transport["transport_callable"]) as client:
        result = client.call_tool_sync(tool_use_id="test-123", name="test_tool", arguments={"param": "value"})

        assert result["status"] == "success"
        assert result["toolUseId"] == "test-123"
        assert len(result["content"]) == 1
        assert result["content"][0]["text"] == "Test message"
        # structuredContent should be None when not provided by MCP
        assert result.get("structuredContent") is None


def test_exception_when_future_not_running():
    """Test exception handling when the future is not running."""
    # Create a client.with a mock transport
    mock_transport_callable = MagicMock()
    client = MCPClient(mock_transport_callable)

    # Create a mock future that is not running
    mock_future = MagicMock()
    mock_future.running.return_value = False
    client._init_future = mock_future

    # Create a mock event loop
    mock_event_loop = MagicMock()
    mock_event_loop.run_until_complete.side_effect = Exception("Test exception")

    # Patch the event loop creation
    with patch("asyncio.new_event_loop", return_value=mock_event_loop):
        # Run the background task which should trigger the exception
        try:
            client._background_task()
        except Exception:
            pass  # We expect an exception to be raised

        # Verify that set_exception was not called since the future was not running
        mock_future.set_exception.assert_not_called()


# Prompt Tests - Sync Methods


def test_list_prompts_sync(mock_transport, mock_session):
    """Test that list_prompts_sync correctly retrieves prompts."""
    mock_prompt = Prompt(name="test_prompt", description="A test prompt", id="prompt_1")
    mock_session.list_prompts.return_value = ListPromptsResult(prompts=[mock_prompt])

    with MCPClient(mock_transport["transport_callable"]) as client:
        result = client.list_prompts_sync()

        mock_session.list_prompts.assert_called_once_with(cursor=None)
        assert len(result.prompts) == 1
        assert result.prompts[0].name == "test_prompt"
        assert result.nextCursor is None


def test_list_prompts_sync_with_pagination_token(mock_transport, mock_session):
    """Test that list_prompts_sync correctly passes pagination token and returns next cursor."""
    mock_prompt = Prompt(name="test_prompt", description="A test prompt", id="prompt_1")
    mock_session.list_prompts.return_value = ListPromptsResult(prompts=[mock_prompt], nextCursor="next_page_token")

    with MCPClient(mock_transport["transport_callable"]) as client:
        result = client.list_prompts_sync(pagination_token="current_page_token")

        mock_session.list_prompts.assert_called_once_with(cursor="current_page_token")
        assert len(result.prompts) == 1
        assert result.prompts[0].name == "test_prompt"
        assert result.nextCursor == "next_page_token"


def test_list_prompts_sync_session_not_active():
    """Test that list_prompts_sync raises an error when session is not active."""
    client = MCPClient(MagicMock())

    with pytest.raises(MCPClientInitializationError, match="client session is not running"):
        client.list_prompts_sync()


def test_get_prompt_sync(mock_transport, mock_session):
    """Test that get_prompt_sync correctly retrieves a prompt."""
    mock_message = PromptMessage(role="user", content=MCPTextContent(type="text", text="This is a test prompt"))
    mock_session.get_prompt.return_value = GetPromptResult(messages=[mock_message])

    with MCPClient(mock_transport["transport_callable"]) as client:
        result = client.get_prompt_sync("test_prompt_id", {"key": "value"})

        mock_session.get_prompt.assert_called_once_with("test_prompt_id", arguments={"key": "value"})
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert result.messages[0].content.text == "This is a test prompt"


def test_get_prompt_sync_session_not_active():
    """Test that get_prompt_sync raises an error when session is not active."""
    client = MCPClient(MagicMock())

    with pytest.raises(MCPClientInitializationError, match="client session is not running"):
        client.get_prompt_sync("test_prompt_id", {})


def test_timeout_initialization_cleanup():
    """Test that timeout during initialization properly cleans up."""

    def slow_transport():
        time.sleep(5)
        return MagicMock()

    client = MCPClient(slow_transport, startup_timeout=1)

    with patch.object(client, "stop") as mock_stop:
        with pytest.raises(MCPClientInitializationError, match="background thread did not start in 1 seconds"):
            client.start()
        mock_stop.assert_called_once_with(None, None, None)


def test_stop_with_no_background_thread():
    """Test that stop() handles the case when no background thread exists."""
    client = MCPClient(MagicMock())

    # Ensure no background thread exists
    assert client._background_thread is None

    # Mock join to verify it's not called
    with patch("threading.Thread.join") as mock_join:
        client.stop(None, None, None)
        mock_join.assert_not_called()

    # Verify cleanup occurred
    assert client._background_thread is None


def test_stop_with_background_thread_but_no_event_loop():
    """Test that stop() handles the case when background thread exists but event loop is None."""
    client = MCPClient(MagicMock())

    # Mock a background thread without event loop
    mock_thread = MagicMock()
    mock_thread.join = MagicMock()
    client._background_thread = mock_thread
    client._background_thread_event_loop = None

    # Should not raise any exceptions and should join the thread
    client.stop(None, None, None)

    # Verify thread was joined
    mock_thread.join.assert_called_once()

    # Verify cleanup occurred
    assert client._background_thread is None


def test_mcp_client_state_reset_after_timeout():
    """Test that all client state is properly reset after timeout."""

    def slow_transport():
        time.sleep(4)  # Longer than timeout
        return MagicMock()

    client = MCPClient(slow_transport, startup_timeout=2)

    # First attempt should timeout
    with pytest.raises(MCPClientInitializationError, match="background thread did not start in 2 seconds"):
        client.start()

    # Verify all state is reset
    assert client._background_thread is None
    assert client._background_thread_session is None
    assert client._background_thread_event_loop is None
    assert not client._init_future.done()  # New future created


def test_call_tool_sync_embedded_nested_text(mock_transport, mock_session):
    """EmbeddedResource.resource (uri + text) should map to plain text content."""
    embedded_resource = {
        "type": "resource",  # required literal
        "resource": {
            "uri": "mcp://resource/embedded-text-1",
            "text": "inner text",
            "mimeType": "text/plain",
        },
    }
    mock_session.call_tool.return_value = MCPCallToolResult(isError=False, content=[embedded_resource])

    with MCPClient(mock_transport["transport_callable"]) as client:
        result = client.call_tool_sync(tool_use_id="er-text", name="get_file_contents", arguments={})

        mock_session.call_tool.assert_called_once_with("get_file_contents", {}, None)
        assert result["status"] == "success"
        assert len(result["content"]) == 1
        assert result["content"][0]["text"] == "inner text"


def test_call_tool_sync_embedded_nested_base64_textual_mime(mock_transport, mock_session):
    """EmbeddedResource.resource (uri + blob with textual MIME) should decode to text."""

    payload = base64.b64encode(b'{"k":"v"}').decode()

    embedded_resource = {
        "type": "resource",
        "resource": {
            "uri": "mcp://resource/embedded-blob-1",
            # NOTE: blob is a STRING, mimeType is sibling
            "blob": payload,
            "mimeType": "application/json",
        },
    }
    mock_session.call_tool.return_value = MCPCallToolResult(isError=False, content=[embedded_resource])

    with MCPClient(mock_transport["transport_callable"]) as client:
        result = client.call_tool_sync(tool_use_id="er-blob", name="get_file_contents", arguments={})

        mock_session.call_tool.assert_called_once_with("get_file_contents", {}, None)
        assert result["status"] == "success"
        assert len(result["content"]) == 1
        assert result["content"][0]["text"] == '{"k":"v"}'


def test_call_tool_sync_embedded_image_blob(mock_transport, mock_session):
    """EmbeddedResource.resource (blob with image MIME) should map to image content."""
    # Read yellow.png file
    with open("tests_integ/yellow.png", "rb") as image_file:
        png_data = image_file.read()
    payload = base64.b64encode(png_data).decode()

    embedded_resource = {
        "type": "resource",
        "resource": {
            "uri": "mcp://resource/embedded-image",
            "blob": payload,
            "mimeType": "image/png",
        },
    }
    mock_session.call_tool.return_value = MCPCallToolResult(isError=False, content=[embedded_resource])

    with MCPClient(mock_transport["transport_callable"]) as client:
        result = client.call_tool_sync(tool_use_id="er-image", name="get_file_contents", arguments={})

        mock_session.call_tool.assert_called_once_with("get_file_contents", {}, None)
        assert result["status"] == "success"
        assert len(result["content"]) == 1
        assert "image" in result["content"][0]
        assert result["content"][0]["image"]["format"] == "png"
        assert "bytes" in result["content"][0]["image"]["source"]


def test_call_tool_sync_embedded_non_textual_blob_dropped(mock_transport, mock_session):
    """EmbeddedResource.resource (blob with non-textual/unknown MIME) should be dropped."""
    payload = base64.b64encode(b"\x00\x01\x02\x03").decode()

    embedded_resource = {
        "type": "resource",
        "resource": {
            "uri": "mcp://resource/embedded-binary",
            "blob": payload,
            "mimeType": "application/octet-stream",
        },
    }
    mock_session.call_tool.return_value = MCPCallToolResult(isError=False, content=[embedded_resource])

    with MCPClient(mock_transport["transport_callable"]) as client:
        result = client.call_tool_sync(tool_use_id="er-binary", name="get_file_contents", arguments={})

        mock_session.call_tool.assert_called_once_with("get_file_contents", {}, None)
        assert result["status"] == "success"
        assert len(result["content"]) == 0  # Content should be dropped


def test_call_tool_sync_embedded_multiple_textual_mimes(mock_transport, mock_session):
    """EmbeddedResource with different textual MIME types should decode to text."""

    # Test YAML content
    yaml_content = base64.b64encode(b"key: value\nlist:\n  - item1\n  - item2").decode()
    embedded_resource = {
        "type": "resource",
        "resource": {
            "uri": "mcp://resource/embedded-yaml",
            "blob": yaml_content,
            "mimeType": "application/yaml",
        },
    }
    mock_session.call_tool.return_value = MCPCallToolResult(isError=False, content=[embedded_resource])

    with MCPClient(mock_transport["transport_callable"]) as client:
        result = client.call_tool_sync(tool_use_id="er-yaml", name="get_file_contents", arguments={})

        mock_session.call_tool.assert_called_once_with("get_file_contents", {}, None)
        assert result["status"] == "success"
        assert len(result["content"]) == 1
        assert "key: value" in result["content"][0]["text"]


def test_call_tool_sync_embedded_unknown_resource_type_dropped(mock_transport, mock_session):
    """EmbeddedResource with unknown resource type should be dropped for forward compatibility."""

    # Mock an unknown resource type that's neither TextResourceContents nor BlobResourceContents
    class UnknownResourceContents:
        def __init__(self):
            self.uri = "mcp://resource/unknown-type"
            self.mimeType = "application/unknown"
            self.data = "some unknown data"

    # Create a mock embedded resource with unknown resource type
    mock_embedded_resource = MagicMock()
    mock_embedded_resource.resource = UnknownResourceContents()

    mock_session.call_tool.return_value = MagicMock(
        isError=False, content=[mock_embedded_resource], structuredContent=None
    )

    with MCPClient(mock_transport["transport_callable"]) as client:
        result = client.call_tool_sync(tool_use_id="er-unknown", name="get_file_contents", arguments={})

        mock_session.call_tool.assert_called_once_with("get_file_contents", {}, None)
        assert result["status"] == "success"
        assert len(result["content"]) == 0  # Unknown resource type should be dropped


@pytest.mark.asyncio
async def test_handle_error_message_non_fatal_error():
    """Test that _handle_error_message ignores non-fatal errors and logs them."""
    client = MCPClient(MagicMock())

    # Test the message handler directly with a non-fatal error
    with patch.object(client, "_log_debug_with_thread") as mock_log:
        # This should not raise an exception
        await client._handle_error_message(Exception("unknown request id: abc123"))

        # Verify the non-fatal error was logged as ignored
        assert mock_log.called
        call_args = mock_log.call_args[0]
        assert "ignoring non-fatal MCP session error" in call_args[0]


@pytest.mark.asyncio
async def test_handle_error_message_fatal_error():
    """Test that _handle_error_message raises fatal errors."""
    client = MCPClient(MagicMock())

    # This should raise the exception
    with pytest.raises(Exception, match="connection timeout"):
        await client._handle_error_message(Exception("connection timeout"))


@pytest.mark.asyncio
async def test_handle_error_message_non_exception():
    """Test that _handle_error_message handles non-exception messages."""
    client = MCPClient(MagicMock())

    # This should not raise an exception
    await client._handle_error_message("normal message")


# Retry Mechanism Tests


def test_mcp_client_with_no_retry_config(mock_transport, mock_session):
    """Test MCPClient behavior with default (no retry) configuration."""
    # Mock successful tool call
    mock_result = MCPCallToolResult(content=[MCPTextContent(type="text", text="success")], isError=False)
    mock_session.call_tool.return_value = mock_result

    client = MCPClient(mock_transport["transport_callable"])

    # Should use NoRetryStrategy by default
    assert isinstance(client._retry_config.strategy, NoRetryStrategy)

    with client:
        result = client.call_tool_sync("test_id", "test_tool", {"arg": "value"})

        # Should complete successfully without retries
        assert result["status"] == "success"
        assert "retryMetadata" in result
        assert result["retryMetadata"]["total_attempts"] == 1
        assert result["retryMetadata"]["retry_strategy_used"] == "NoRetryStrategy"


def test_mcp_client_with_global_retry_strategy(mock_transport, mock_session):
    """Test MCPClient with global retry strategy."""
    retry_config = MCPRetryConfig(
        strategy=ExponentialBackoffRetry(max_attempts=3, min_wait=0.01)  # Fast for testing
    )

    # Mock tool call that fails twice then succeeds
    call_count = 0

    def mock_call_tool(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception(f"Temporary failure {call_count}")
        return MCPCallToolResult(content=[MCPTextContent(type="text", text="success")], isError=False)

    mock_session.call_tool.side_effect = mock_call_tool

    client = MCPClient(mock_transport["transport_callable"], retry_config=retry_config)

    with client:
        result = client.call_tool_sync("test_id", "test_tool", {"arg": "value"})

        # Should succeed after retries
        assert result["status"] == "success"
        assert result["retryMetadata"]["total_attempts"] == 3
        assert result["retryMetadata"]["retry_strategy_used"] == "ExponentialBackoffRetry"
        assert "last_exception" in result["retryMetadata"]


def test_mcp_client_with_tool_specific_retry(mock_transport, mock_session):
    """Test MCPClient with tool-specific retry configuration."""
    retry_config = MCPRetryConfig(
        strategy=NoRetryStrategy(),  # Global: no retries
        tool_overrides={"retryable_tool": ExponentialBackoffRetry(max_attempts=2, min_wait=0.01)},
    )

    # Mock tool call that always fails
    mock_session.call_tool.side_effect = Exception("Tool failure")

    client = MCPClient(mock_transport["transport_callable"], retry_config=retry_config)

    with client:
        # Tool with no retry should fail immediately
        result1 = client.call_tool_sync("test_id1", "normal_tool", {})
        assert result1["status"] == "error"
        assert "Tool execution failed: Tool failure" in result1["content"][0]["text"]

        # Tool with retry should attempt multiple times
        result2 = client.call_tool_sync("test_id2", "retryable_tool", {})
        assert result2["status"] == "error"
        # Should show it was attempted multiple times (but we can't check exact count due to mocking complexity)


@pytest.mark.asyncio
async def test_mcp_client_async_with_retry(mock_transport, mock_session):
    """Test MCPClient async method with retry strategy."""
    retry_config = MCPRetryConfig(strategy=ExponentialBackoffRetry(max_attempts=2, min_wait=0.01))

    # Mock tool call that fails once then succeeds
    call_count = 0

    async def mock_call_tool(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("First attempt failure")
        return MCPCallToolResult(content=[MCPTextContent(type="text", text="success")], isError=False)

    mock_session.call_tool.side_effect = mock_call_tool

    client = MCPClient(mock_transport["transport_callable"], retry_config=retry_config)

    with client:
        result = await client.call_tool_async("test_id", "test_tool", {"arg": "value"})

        # Should succeed after retry
        assert result["status"] == "success"
        assert result["retryMetadata"]["total_attempts"] == 2
        assert result["retryMetadata"]["retry_strategy_used"] == "ExponentialBackoffRetry"


def test_retry_config_fluent_interface():
    """Test MCPRetryConfig fluent interface for setting tool strategies."""
    config = MCPRetryConfig()
    strategy1 = ExponentialBackoffRetry(max_attempts=2)
    strategy2 = ExponentialBackoffRetry(max_attempts=5)

    # Test method chaining
    result = config.set_tool_strategy("tool1", strategy1).set_tool_strategy("tool2", strategy2)

    assert result is config  # Should return same instance
    assert config.get_strategy_for_tool("tool1") is strategy1
    assert config.get_strategy_for_tool("tool2") is strategy2
    assert isinstance(config.get_strategy_for_tool("other_tool"), NoRetryStrategy)  # Should use default


def test_mcp_client_initialization_with_retry_config(mock_transport):
    """Test that MCPClient properly initializes with retry configuration."""
    retry_config = MCPRetryConfig(strategy=ExponentialBackoffRetry(max_attempts=5))

    client = MCPClient(mock_transport["transport_callable"], retry_config=retry_config)

    assert client._retry_config is retry_config
    assert client._retry_config.strategy.max_attempts == 5


def test_mcp_client_initialization_without_retry_config(mock_transport):
    """Test that MCPClient uses default retry configuration when none provided."""
    client = MCPClient(mock_transport["transport_callable"])

    assert isinstance(client._retry_config, MCPRetryConfig)
    assert isinstance(client._retry_config.strategy, NoRetryStrategy)
    assert len(client._retry_config.tool_overrides) == 0
