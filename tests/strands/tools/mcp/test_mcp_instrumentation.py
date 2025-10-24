from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage, JSONRPCRequest
from opentelemetry import context, propagate

from strands.tools.mcp.mcp_client import MCPClient
from strands.tools.mcp.mcp_instrumentation import (
    ItemWithContext,
    SessionContextAttachingReader,
    SessionContextSavingWriter,
    TransportContextExtractingReader,
    mcp_instrumentation,
)


@pytest.fixture(autouse=True)
def reset_mcp_instrumentation():
    """Reset MCP instrumentation state before each test."""
    import strands.tools.mcp.mcp_instrumentation as mcp_inst

    mcp_inst._instrumentation_applied = False
    yield
    # Reset after test too
    mcp_inst._instrumentation_applied = False


class TestItemWithContext:
    def test_item_with_context_creation(self):
        """Test that ItemWithContext correctly stores item and context."""
        test_item = {"test": "data"}
        test_context = context.get_current()

        wrapped = ItemWithContext(test_item, test_context)

        assert wrapped.item == test_item
        assert wrapped.ctx == test_context


class TestTransportContextExtractingReader:
    @pytest.fixture
    def mock_wrapped_reader(self):
        """Create a mock wrapped reader."""
        mock_reader = AsyncMock()
        mock_reader.__aenter__ = AsyncMock(return_value=mock_reader)
        mock_reader.__aexit__ = AsyncMock()
        return mock_reader

    def test_init(self, mock_wrapped_reader):
        """Test reader initialization."""
        reader = TransportContextExtractingReader(mock_wrapped_reader)
        assert reader.__wrapped__ == mock_wrapped_reader

    @pytest.mark.asyncio
    async def test_context_manager_methods(self, mock_wrapped_reader):
        """Test async context manager methods delegate correctly."""
        reader = TransportContextExtractingReader(mock_wrapped_reader)

        await reader.__aenter__()
        mock_wrapped_reader.__aenter__.assert_called_once()

        await reader.__aexit__(None, None, None)
        mock_wrapped_reader.__aexit__.assert_called_once_with(None, None, None)

    @pytest.mark.asyncio
    async def test_aiter_with_session_message_and_dict_meta(self, mock_wrapped_reader):
        """Test context extraction from SessionMessage with dict params containing _meta."""
        # Create mock message with dict params containing _meta
        mock_request = MagicMock(spec=JSONRPCRequest)
        mock_request.params = {"_meta": {"traceparent": "test-trace-id"}, "other": "data"}

        mock_message = MagicMock()
        mock_message.root = mock_request

        mock_session_message = MagicMock(spec=SessionMessage)
        mock_session_message.message = mock_message

        async def async_iter():
            for item in [mock_session_message]:
                yield item

        mock_wrapped_reader.__aiter__ = lambda self: async_iter()

        reader = TransportContextExtractingReader(mock_wrapped_reader)

        with (
            patch.object(propagate, "extract") as mock_extract,
            patch.object(context, "attach") as mock_attach,
            patch.object(context, "detach") as mock_detach,
        ):
            mock_context = MagicMock()
            mock_extract.return_value = mock_context
            mock_token = MagicMock()
            mock_attach.return_value = mock_token

            items = []
            async for item in reader:
                items.append(item)

            assert len(items) == 1
            assert items[0] == mock_session_message

            mock_extract.assert_called_once_with({"traceparent": "test-trace-id"})
            mock_attach.assert_called_once_with(mock_context)
            mock_detach.assert_called_once_with(mock_token)

    @pytest.mark.asyncio
    async def test_aiter_with_session_message_and_pydantic_meta(self, mock_wrapped_reader):
        """Test context extraction from SessionMessage with Pydantic params having _meta attribute."""
        # Create mock message with Pydantic-style params
        mock_request = MagicMock(spec=JSONRPCRequest)

        # Create a mock params object that doesn't have 'get' method but has '_meta' attribute
        mock_params = MagicMock()
        # Remove the get method to simulate Pydantic model behavior
        del mock_params.get
        mock_params._meta = {"traceparent": "test-trace-id"}
        mock_request.params = mock_params

        mock_message = MagicMock()
        mock_message.root = mock_request

        mock_session_message = MagicMock(spec=SessionMessage)
        mock_session_message.message = mock_message

        async def async_iter():
            for item in [mock_session_message]:
                yield item

        mock_wrapped_reader.__aiter__ = lambda self: async_iter()

        reader = TransportContextExtractingReader(mock_wrapped_reader)

        with (
            patch.object(propagate, "extract") as mock_extract,
            patch.object(context, "attach") as mock_attach,
            patch.object(context, "detach") as mock_detach,
        ):
            mock_context = MagicMock()
            mock_extract.return_value = mock_context
            mock_token = MagicMock()
            mock_attach.return_value = mock_token

            items = []
            async for item in reader:
                items.append(item)

            assert len(items) == 1
            assert items[0] == mock_session_message

            mock_extract.assert_called_once_with({"traceparent": "test-trace-id"})
            mock_attach.assert_called_once_with(mock_context)
            mock_detach.assert_called_once_with(mock_token)

    @pytest.mark.asyncio
    async def test_aiter_with_jsonrpc_message_no_meta(self, mock_wrapped_reader):
        """Test handling JSONRPCMessage without _meta."""
        mock_request = MagicMock(spec=JSONRPCRequest)
        mock_request.params = {"other": "data"}

        mock_message = MagicMock(spec=JSONRPCMessage)
        mock_message.root = mock_request

        async def async_iter():
            for item in [mock_message]:
                yield item

        mock_wrapped_reader.__aiter__ = lambda self: async_iter()

        reader = TransportContextExtractingReader(mock_wrapped_reader)

        items = []
        async for item in reader:
            items.append(item)

        assert len(items) == 1
        assert items[0] == mock_message

    @pytest.mark.asyncio
    async def test_aiter_with_non_message_item(self, mock_wrapped_reader):
        """Test handling non-message items."""
        other_item = {"not": "a message"}

        async def async_iter():
            for item in [other_item]:
                yield item

        mock_wrapped_reader.__aiter__ = lambda self: async_iter()

        reader = TransportContextExtractingReader(mock_wrapped_reader)

        items = []
        async for item in reader:
            items.append(item)

        assert len(items) == 1
        assert items[0] == other_item


class TestSessionContextSavingWriter:
    @pytest.fixture
    def mock_wrapped_writer(self):
        """Create a mock wrapped writer."""
        mock_writer = AsyncMock()
        mock_writer.__aenter__ = AsyncMock(return_value=mock_writer)
        mock_writer.__aexit__ = AsyncMock()
        mock_writer.send = AsyncMock()
        return mock_writer

    def test_init(self, mock_wrapped_writer):
        """Test writer initialization."""
        writer = SessionContextSavingWriter(mock_wrapped_writer)
        assert writer.__wrapped__ == mock_wrapped_writer

    @pytest.mark.asyncio
    async def test_context_manager_methods(self, mock_wrapped_writer):
        """Test async context manager methods delegate correctly."""
        writer = SessionContextSavingWriter(mock_wrapped_writer)

        await writer.__aenter__()
        mock_wrapped_writer.__aenter__.assert_called_once()

        await writer.__aexit__(None, None, None)
        mock_wrapped_writer.__aexit__.assert_called_once_with(None, None, None)

    @pytest.mark.asyncio
    async def test_send_wraps_item_with_context(self, mock_wrapped_writer):
        """Test that send wraps items with current context."""
        writer = SessionContextSavingWriter(mock_wrapped_writer)
        test_item = {"test": "data"}

        with patch.object(context, "get_current") as mock_get_current:
            mock_context = MagicMock()
            mock_get_current.return_value = mock_context

            await writer.send(test_item)

            mock_get_current.assert_called_once()
            mock_wrapped_writer.send.assert_called_once()

            # Verify the item was wrapped with context
            sent_item = mock_wrapped_writer.send.call_args[0][0]
            assert isinstance(sent_item, ItemWithContext)
            assert sent_item.item == test_item
            assert sent_item.ctx == mock_context


class TestSessionContextAttachingReader:
    @pytest.fixture
    def mock_wrapped_reader(self):
        """Create a mock wrapped reader."""
        mock_reader = AsyncMock()
        mock_reader.__aenter__ = AsyncMock(return_value=mock_reader)
        mock_reader.__aexit__ = AsyncMock()
        return mock_reader

    def test_init(self, mock_wrapped_reader):
        """Test reader initialization."""
        reader = SessionContextAttachingReader(mock_wrapped_reader)
        assert reader.__wrapped__ == mock_wrapped_reader

    @pytest.mark.asyncio
    async def test_context_manager_methods(self, mock_wrapped_reader):
        """Test async context manager methods delegate correctly."""
        reader = SessionContextAttachingReader(mock_wrapped_reader)

        await reader.__aenter__()
        mock_wrapped_reader.__aenter__.assert_called_once()

        await reader.__aexit__(None, None, None)
        mock_wrapped_reader.__aexit__.assert_called_once_with(None, None, None)

    @pytest.mark.asyncio
    async def test_aiter_with_item_with_context(self, mock_wrapped_reader):
        """Test context restoration from ItemWithContext."""
        test_item = {"test": "data"}
        test_context = MagicMock()
        wrapped_item = ItemWithContext(test_item, test_context)

        async def async_iter():
            for item in [wrapped_item]:
                yield item

        mock_wrapped_reader.__aiter__ = lambda self: async_iter()

        reader = SessionContextAttachingReader(mock_wrapped_reader)

        with patch.object(context, "attach") as mock_attach, patch.object(context, "detach") as mock_detach:
            mock_token = MagicMock()
            mock_attach.return_value = mock_token

            items = []
            async for item in reader:
                items.append(item)

            assert len(items) == 1
            assert items[0] == test_item

            mock_attach.assert_called_once_with(test_context)
            mock_detach.assert_called_once_with(mock_token)

    @pytest.mark.asyncio
    async def test_aiter_with_regular_item(self, mock_wrapped_reader):
        """Test handling regular items without context."""
        regular_item = {"regular": "item"}

        async def async_iter():
            for item in [regular_item]:
                yield item

        mock_wrapped_reader.__aiter__ = lambda self: async_iter()

        reader = SessionContextAttachingReader(mock_wrapped_reader)

        items = []
        async for item in reader:
            items.append(item)

        assert len(items) == 1
        assert items[0] == regular_item


# Mock Pydantic-like class for testing
class MockPydanticParams:
    """Mock class that behaves like a Pydantic model."""

    def __init__(self, **data):
        self._data = data

    def model_dump(self):
        return self._data.copy()

    @classmethod
    def model_validate(cls, data):
        return cls(**data)

    def __getattr__(self, name):
        return self._data.get(name)


class TestMCPInstrumentation:
    def test_mcp_instrumentation_called_on_client_init(self):
        """Test that mcp_instrumentation is called when MCPClient is initialized."""
        with patch("strands.tools.mcp.mcp_client.mcp_instrumentation") as mock_instrumentation:
            # Mock transport
            def mock_transport():
                read_stream = AsyncMock()
                write_stream = AsyncMock()
                return read_stream, write_stream

            # Create MCPClient instance - should call mcp_instrumentation
            MCPClient(mock_transport)

            # Verify mcp_instrumentation was called
            mock_instrumentation.assert_called_once()

    def test_mcp_instrumentation_idempotent_with_multiple_clients(self):
        """Test that mcp_instrumentation is only called once even with multiple MCPClient instances."""

        # Mock the wrap_function_wrapper to count calls
        with patch("strands.tools.mcp.mcp_instrumentation.wrap_function_wrapper") as mock_wrap:
            # Mock transport
            def mock_transport():
                read_stream = AsyncMock()
                write_stream = AsyncMock()
                return read_stream, write_stream

            # Create first MCPClient instance - should apply instrumentation
            MCPClient(mock_transport)
            first_call_count = mock_wrap.call_count

            # Create second MCPClient instance - should NOT apply instrumentation again
            MCPClient(mock_transport)

            # wrap_function_wrapper should not be called again for the second client
            assert mock_wrap.call_count == first_call_count

    def test_mcp_instrumentation_calls_wrap_function_wrapper(self):
        """Test that mcp_instrumentation calls the expected wrapper functions."""
        with (
            patch("strands.tools.mcp.mcp_instrumentation.wrap_function_wrapper") as mock_wrap,
            patch("strands.tools.mcp.mcp_instrumentation.register_post_import_hook") as mock_register,
        ):
            mcp_instrumentation()

            # Verify wrap_function_wrapper was called for client patching
            mock_wrap.assert_called_once_with(
                "mcp.shared.session",
                "BaseSession.send_request",
                mock_wrap.call_args_list[0][0][2],  # The patch function
            )

            # Verify register_post_import_hook was called for transport and session wrappers
            assert mock_register.call_count == 2

            # Check that the registered hooks are for the expected modules
            registered_modules = [call[0][1] for call in mock_register.call_args_list]
            assert "mcp.server.streamable_http" in registered_modules
            assert "mcp.server.session" in registered_modules

    def test_patch_mcp_client_injects_context_pydantic_model(self):
        """Test that the client patch injects OpenTelemetry context into Pydantic models."""
        # Create a mock request with tools/call method and Pydantic params
        mock_request = MagicMock()
        mock_request.root.method = "tools/call"

        # Use our mock Pydantic-like class
        mock_params = MockPydanticParams(existing="param")
        mock_request.root.params = mock_params

        # Create the patch function
        with patch("strands.tools.mcp.mcp_instrumentation.wrap_function_wrapper") as mock_wrap:
            mcp_instrumentation()
            patch_function = mock_wrap.call_args_list[0][0][2]

        # Mock the wrapped function
        mock_wrapped = MagicMock()

        with patch.object(propagate, "get_global_textmap") as mock_textmap:
            mock_textmap_instance = MagicMock()
            mock_textmap.return_value = mock_textmap_instance

            # Call the patch function
            patch_function(mock_wrapped, None, [mock_request], {})

            # Verify context was injected
            mock_textmap_instance.inject.assert_called_once()
            mock_wrapped.assert_called_once_with(mock_request)

            # Verify the params object is still a MockPydanticParams (or dict if fallback occurred)
            assert hasattr(mock_request.root.params, "model_dump") or isinstance(mock_request.root.params, dict)

    def test_patch_mcp_client_injects_context_dict_params(self):
        """Test that the client patch injects OpenTelemetry context into dict params."""
        # Create a mock request with tools/call method and dict params
        mock_request = MagicMock()
        mock_request.root.method = "tools/call"
        mock_request.root.params = {"existing": "param"}

        # Create the patch function
        with patch("strands.tools.mcp.mcp_instrumentation.wrap_function_wrapper") as mock_wrap:
            mcp_instrumentation()
            patch_function = mock_wrap.call_args_list[0][0][2]

        # Mock the wrapped function
        mock_wrapped = MagicMock()

        with patch.object(propagate, "get_global_textmap") as mock_textmap:
            mock_textmap_instance = MagicMock()
            mock_textmap.return_value = mock_textmap_instance

            # Call the patch function
            patch_function(mock_wrapped, None, [mock_request], {})

            # Verify context was injected
            mock_textmap_instance.inject.assert_called_once()
            mock_wrapped.assert_called_once_with(mock_request)

            # Verify _meta was added to the params dict
            assert "_meta" in mock_request.root.params

    def test_patch_mcp_client_skips_non_tools_call(self):
        """Test that the client patch skips non-tools/call methods."""
        mock_request = MagicMock()
        mock_request.root.method = "other/method"

        with patch("strands.tools.mcp.mcp_instrumentation.wrap_function_wrapper") as mock_wrap:
            mcp_instrumentation()
            patch_function = mock_wrap.call_args_list[0][0][2]

        mock_wrapped = MagicMock()

        with patch.object(propagate, "get_global_textmap") as mock_textmap:
            mock_textmap_instance = MagicMock()
            mock_textmap.return_value = mock_textmap_instance

            patch_function(mock_wrapped, None, [mock_request], {})

            # Verify context injection was skipped
            mock_textmap_instance.inject.assert_not_called()
            mock_wrapped.assert_called_once_with(mock_request)

    def test_patch_mcp_client_handles_exception_gracefully(self):
        """Test that the client patch handles exceptions gracefully."""
        # Create a mock request that will cause an exception
        mock_request = MagicMock()
        mock_request.root.method = "tools/call"
        mock_request.root.params = MagicMock()
        mock_request.root.params.model_dump.side_effect = Exception("Test exception")

        with patch("strands.tools.mcp.mcp_instrumentation.wrap_function_wrapper") as mock_wrap:
            mcp_instrumentation()
            patch_function = mock_wrap.call_args_list[0][0][2]

        mock_wrapped = MagicMock()

        # Should not raise an exception, should call wrapped function normally
        patch_function(mock_wrapped, None, [mock_request], {})
        mock_wrapped.assert_called_once_with(mock_request)

    def test_patch_mcp_client_pydantic_fallback_to_dict(self):
        """Test that Pydantic model recreation falls back to dict on failure."""

        # Create a Pydantic-like class that fails on model_validate
        class FailingMockPydanticParams:
            def __init__(self, **data):
                self._data = data

            def model_dump(self):
                return self._data.copy()

            def model_validate(self, data):
                raise Exception("Reconstruction failed")

        # Create a mock request with failing Pydantic params
        mock_request = MagicMock()
        mock_request.root.method = "tools/call"

        failing_params = FailingMockPydanticParams(existing="param")
        mock_request.root.params = failing_params

        with patch("strands.tools.mcp.mcp_instrumentation.wrap_function_wrapper") as mock_wrap:
            mcp_instrumentation()
            patch_function = mock_wrap.call_args_list[0][0][2]

        mock_wrapped = MagicMock()

        with patch.object(propagate, "get_global_textmap") as mock_textmap:
            mock_textmap_instance = MagicMock()
            mock_textmap.return_value = mock_textmap_instance

            # Call the patch function
            patch_function(mock_wrapped, None, [mock_request], {})

            # Verify it fell back to dict
            assert isinstance(mock_request.root.params, dict)
            assert "_meta" in mock_request.root.params
            mock_wrapped.assert_called_once_with(mock_request)
