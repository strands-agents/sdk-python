"""Tests for the StrandsA2AExecutor class."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.types import UnsupportedOperationError
from a2a.utils.errors import ServerError

from strands.agent.agent_result import AgentResult as SAAgentResult
from strands.multiagent.a2a.executor import StrandsA2AExecutor
from strands.types.content import ContentBlock


def test_executor_initialization(mock_strands_agent):
    """Test that StrandsA2AExecutor initializes correctly."""
    executor = StrandsA2AExecutor(mock_strands_agent)

    assert executor.agent == mock_strands_agent


def test_classify_file_type():
    """Test file type classification based on MIME type."""
    executor = StrandsA2AExecutor(MagicMock())

    # Test image types
    assert executor._get_file_type_from_mime_type("image/jpeg") == "image"
    assert executor._get_file_type_from_mime_type("image/png") == "image"

    # Test video types
    assert executor._get_file_type_from_mime_type("video/mp4") == "video"
    assert executor._get_file_type_from_mime_type("video/mpeg") == "video"

    # Test document types
    assert executor._get_file_type_from_mime_type("text/plain") == "document"
    assert executor._get_file_type_from_mime_type("application/pdf") == "document"
    assert executor._get_file_type_from_mime_type("application/json") == "document"

    # Test unknown/edge cases
    assert executor._get_file_type_from_mime_type("audio/mp3") == "unknown"
    assert executor._get_file_type_from_mime_type(None) == "unknown"
    assert executor._get_file_type_from_mime_type("") == "unknown"


def test_get_file_format_from_mime_type():
    """Test file format extraction from MIME type using mimetypes library."""
    executor = StrandsA2AExecutor(MagicMock())
    assert executor._get_file_format_from_mime_type("image/jpeg", "image") == "jpeg"
    assert executor._get_file_format_from_mime_type("image/png", "image") == "png"
    assert executor._get_file_format_from_mime_type("image/unknown", "image") == "png"

    # Test video formats
    assert executor._get_file_format_from_mime_type("video/mp4", "video") == "mp4"
    assert executor._get_file_format_from_mime_type("video/3gpp", "video") == "three_gp"
    assert executor._get_file_format_from_mime_type("video/unknown", "video") == "mp4"

    # Test document formats
    assert executor._get_file_format_from_mime_type("application/pdf", "document") == "pdf"
    assert executor._get_file_format_from_mime_type("text/plain", "document") == "txt"
    markdown_result = executor._get_file_format_from_mime_type("text/markdown", "document")
    assert markdown_result in ["markdown", "md"]
    assert executor._get_file_format_from_mime_type("application/unknown", "document") == "txt"

    # Test None/empty cases
    assert executor._get_file_format_from_mime_type(None, "image") == "png"
    assert executor._get_file_format_from_mime_type("", "video") == "mp4"


def test_strip_file_extension():
    """Test file extension stripping."""
    executor = StrandsA2AExecutor(MagicMock())

    assert executor._strip_file_extension("test.txt") == "test"
    assert executor._strip_file_extension("document.pdf") == "document"
    assert executor._strip_file_extension("image.jpeg") == "image"
    assert executor._strip_file_extension("no_extension") == "no_extension"
    assert executor._strip_file_extension("multiple.dots.file.ext") == "multiple.dots.file"


def test_convert_a2a_parts_to_content_blocks_text_part():
    """Test conversion of TextPart to ContentBlock."""
    from a2a.types import TextPart

    executor = StrandsA2AExecutor(MagicMock())

    # Mock TextPart with proper spec
    text_part = MagicMock(spec=TextPart)
    text_part.text = "Hello, world!"

    # Mock Part with TextPart root
    part = MagicMock()
    part.root = text_part

    result = executor._convert_a2a_parts_to_content_blocks([part])

    assert len(result) == 1
    assert result[0] == ContentBlock(text="Hello, world!")


def test_convert_a2a_parts_to_content_blocks_file_part_image_bytes():
    """Test conversion of FilePart with image bytes to ContentBlock."""
    from a2a.types import FilePart

    executor = StrandsA2AExecutor(MagicMock())

    # Create test image bytes (no base64 encoding needed)
    test_bytes = b"fake_image_data"

    # Mock file object
    file_obj = MagicMock()
    file_obj.name = "test_image.jpeg"
    file_obj.mime_type = "image/jpeg"
    file_obj.bytes = test_bytes
    file_obj.uri = None

    # Mock FilePart with proper spec
    file_part = MagicMock(spec=FilePart)
    file_part.file = file_obj

    # Mock Part with FilePart root
    part = MagicMock()
    part.root = file_part

    result = executor._convert_a2a_parts_to_content_blocks([part])

    assert len(result) == 1
    content_block = result[0]
    assert "image" in content_block
    assert content_block["image"]["format"] == "jpeg"
    assert content_block["image"]["source"]["bytes"] == test_bytes


def test_convert_a2a_parts_to_content_blocks_file_part_video_bytes():
    """Test conversion of FilePart with video bytes to ContentBlock."""
    from a2a.types import FilePart

    executor = StrandsA2AExecutor(MagicMock())

    # Create test video bytes (no base64 encoding needed)
    test_bytes = b"fake_video_data"

    # Mock file object
    file_obj = MagicMock()
    file_obj.name = "test_video.mp4"
    file_obj.mime_type = "video/mp4"
    file_obj.bytes = test_bytes
    file_obj.uri = None

    # Mock FilePart with proper spec
    file_part = MagicMock(spec=FilePart)
    file_part.file = file_obj

    # Mock Part with FilePart root
    part = MagicMock()
    part.root = file_part

    result = executor._convert_a2a_parts_to_content_blocks([part])

    assert len(result) == 1
    content_block = result[0]
    assert "video" in content_block
    assert content_block["video"]["format"] == "mp4"
    assert content_block["video"]["source"]["bytes"] == test_bytes


def test_convert_a2a_parts_to_content_blocks_file_part_document_bytes():
    """Test conversion of FilePart with document bytes to ContentBlock."""
    from a2a.types import FilePart

    executor = StrandsA2AExecutor(MagicMock())

    # Create test document bytes (no base64 encoding needed)
    test_bytes = b"fake_document_data"

    # Mock file object
    file_obj = MagicMock()
    file_obj.name = "test_document.pdf"
    file_obj.mime_type = "application/pdf"
    file_obj.bytes = test_bytes
    file_obj.uri = None

    # Mock FilePart with proper spec
    file_part = MagicMock(spec=FilePart)
    file_part.file = file_obj

    # Mock Part with FilePart root
    part = MagicMock()
    part.root = file_part

    result = executor._convert_a2a_parts_to_content_blocks([part])

    assert len(result) == 1
    content_block = result[0]
    assert "document" in content_block
    assert content_block["document"]["format"] == "pdf"
    assert content_block["document"]["name"] == "test_document"
    assert content_block["document"]["source"]["bytes"] == test_bytes


def test_convert_a2a_parts_to_content_blocks_file_part_uri():
    """Test conversion of FilePart with URI to ContentBlock."""
    from a2a.types import FilePart

    executor = StrandsA2AExecutor(MagicMock())

    # Mock file object with URI
    file_obj = MagicMock()
    file_obj.name = "test_image.png"
    file_obj.mime_type = "image/png"
    file_obj.bytes = None
    file_obj.uri = "https://example.com/image.png"

    # Mock FilePart with proper spec
    file_part = MagicMock(spec=FilePart)
    file_part.file = file_obj

    # Mock Part with FilePart root
    part = MagicMock()
    part.root = file_part

    result = executor._convert_a2a_parts_to_content_blocks([part])

    assert len(result) == 1
    content_block = result[0]
    assert "text" in content_block
    assert "test_image" in content_block["text"]
    assert "https://example.com/image.png" in content_block["text"]


def test_convert_a2a_parts_to_content_blocks_file_part_with_bytes():
    """Test conversion of FilePart with bytes data."""
    from a2a.types import FilePart

    executor = StrandsA2AExecutor(MagicMock())

    # Mock file object with bytes (no validation needed since no decoding)
    file_obj = MagicMock()
    file_obj.name = "test_image.png"
    file_obj.mime_type = "image/png"
    file_obj.bytes = b"some_binary_data"
    file_obj.uri = None

    # Mock FilePart with proper spec
    file_part = MagicMock(spec=FilePart)
    file_part.file = file_obj

    # Mock Part with FilePart root
    part = MagicMock()
    part.root = file_part

    result = executor._convert_a2a_parts_to_content_blocks([part])

    assert len(result) == 1
    content_block = result[0]
    assert "image" in content_block
    assert content_block["image"]["source"]["bytes"] == b"some_binary_data"


def test_convert_a2a_parts_to_content_blocks_data_part():
    """Test conversion of DataPart to ContentBlock."""
    from a2a.types import DataPart

    executor = StrandsA2AExecutor(MagicMock())

    # Mock DataPart with proper spec
    test_data = {"key": "value", "number": 42}
    data_part = MagicMock(spec=DataPart)
    data_part.data = test_data

    # Mock Part with DataPart root
    part = MagicMock()
    part.root = data_part

    result = executor._convert_a2a_parts_to_content_blocks([part])

    assert len(result) == 1
    content_block = result[0]
    assert "text" in content_block
    assert "[Structured Data]" in content_block["text"]
    assert "key" in content_block["text"]
    assert "value" in content_block["text"]


def test_convert_a2a_parts_to_content_blocks_mixed_parts():
    """Test conversion of mixed A2A parts to ContentBlocks."""
    from a2a.types import DataPart, TextPart

    executor = StrandsA2AExecutor(MagicMock())

    # Mock TextPart with proper spec
    text_part = MagicMock(spec=TextPart)
    text_part.text = "Text content"
    text_part_mock = MagicMock()
    text_part_mock.root = text_part

    # Mock DataPart with proper spec
    data_part = MagicMock(spec=DataPart)
    data_part.data = {"test": "data"}
    data_part_mock = MagicMock()
    data_part_mock.root = data_part

    parts = [text_part_mock, data_part_mock]
    result = executor._convert_a2a_parts_to_content_blocks(parts)

    assert len(result) == 2
    assert result[0]["text"] == "Text content"
    assert "[Structured Data]" in result[1]["text"]


@pytest.mark.asyncio
async def test_execute_streaming_mode_with_data_events(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that execute processes data events correctly in streaming mode."""

    async def mock_stream(content_blocks):
        """Mock streaming function that yields data events."""
        yield {"data": "First chunk"}
        yield {"data": "Second chunk"}
        yield {"result": MagicMock(spec=SAAgentResult)}

    # Setup mock agent streaming
    mock_strands_agent.stream_async = MagicMock(return_value=mock_stream([]))

    # Create executor
    executor = StrandsA2AExecutor(mock_strands_agent)

    # Mock the task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.context_id = "test-context-id"
    mock_request_context.current_task = mock_task

    # Mock message with parts
    from a2a.types import TextPart

    mock_message = MagicMock()
    text_part = MagicMock(spec=TextPart)
    text_part.text = "Test input"
    part = MagicMock()
    part.root = text_part
    mock_message.parts = [part]
    mock_request_context.message = mock_message

    await executor.execute(mock_request_context, mock_event_queue)

    # Verify agent was called with ContentBlock list
    mock_strands_agent.stream_async.assert_called_once()
    call_args = mock_strands_agent.stream_async.call_args[0][0]
    assert isinstance(call_args, list)
    assert len(call_args) == 1
    assert call_args[0]["text"] == "Test input"

    # Verify events were enqueued
    mock_event_queue.enqueue_event.assert_called()


@pytest.mark.asyncio
async def test_execute_streaming_mode_with_result_event(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that execute processes result events correctly in streaming mode."""

    async def mock_stream(content_blocks):
        """Mock streaming function that yields only result event."""
        yield {"result": MagicMock(spec=SAAgentResult)}

    # Setup mock agent streaming
    mock_strands_agent.stream_async = MagicMock(return_value=mock_stream([]))

    # Create executor
    executor = StrandsA2AExecutor(mock_strands_agent)

    # Mock the task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.context_id = "test-context-id"
    mock_request_context.current_task = mock_task

    # Mock message with parts
    from a2a.types import TextPart

    mock_message = MagicMock()
    text_part = MagicMock(spec=TextPart)
    text_part.text = "Test input"
    part = MagicMock()
    part.root = text_part
    mock_message.parts = [part]
    mock_request_context.message = mock_message

    await executor.execute(mock_request_context, mock_event_queue)

    # Verify agent was called with ContentBlock list
    mock_strands_agent.stream_async.assert_called_once()
    call_args = mock_strands_agent.stream_async.call_args[0][0]
    assert isinstance(call_args, list)
    assert len(call_args) == 1
    assert call_args[0]["text"] == "Test input"

    # Verify events were enqueued
    mock_event_queue.enqueue_event.assert_called()


@pytest.mark.asyncio
async def test_execute_streaming_mode_with_empty_data(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that execute handles empty data events correctly in streaming mode."""

    async def mock_stream(content_blocks):
        """Mock streaming function that yields empty data."""
        yield {"data": ""}
        yield {"result": MagicMock(spec=SAAgentResult)}

    # Setup mock agent streaming
    mock_strands_agent.stream_async = MagicMock(return_value=mock_stream([]))

    # Create executor
    executor = StrandsA2AExecutor(mock_strands_agent)

    # Mock the task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.context_id = "test-context-id"
    mock_request_context.current_task = mock_task

    # Mock message with parts
    from a2a.types import TextPart

    mock_message = MagicMock()
    text_part = MagicMock(spec=TextPart)
    text_part.text = "Test input"
    part = MagicMock()
    part.root = text_part
    mock_message.parts = [part]
    mock_request_context.message = mock_message

    await executor.execute(mock_request_context, mock_event_queue)

    # Verify agent was called with ContentBlock list
    mock_strands_agent.stream_async.assert_called_once()
    call_args = mock_strands_agent.stream_async.call_args[0][0]
    assert isinstance(call_args, list)
    assert len(call_args) == 1
    assert call_args[0]["text"] == "Test input"

    # Verify events were enqueued
    mock_event_queue.enqueue_event.assert_called()


@pytest.mark.asyncio
async def test_execute_streaming_mode_with_unexpected_event(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that execute handles unexpected events correctly in streaming mode."""

    async def mock_stream(content_blocks):
        """Mock streaming function that yields unexpected event."""
        yield {"unexpected": "event"}
        yield {"result": MagicMock(spec=SAAgentResult)}

    # Setup mock agent streaming
    mock_strands_agent.stream_async = MagicMock(return_value=mock_stream([]))

    # Create executor
    executor = StrandsA2AExecutor(mock_strands_agent)

    # Mock the task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.context_id = "test-context-id"
    mock_request_context.current_task = mock_task

    # Mock message with parts
    from a2a.types import TextPart

    mock_message = MagicMock()
    text_part = MagicMock(spec=TextPart)
    text_part.text = "Test input"
    part = MagicMock()
    part.root = text_part
    mock_message.parts = [part]
    mock_request_context.message = mock_message

    await executor.execute(mock_request_context, mock_event_queue)

    # Verify agent was called with ContentBlock list
    mock_strands_agent.stream_async.assert_called_once()
    call_args = mock_strands_agent.stream_async.call_args[0][0]
    assert isinstance(call_args, list)
    assert len(call_args) == 1
    assert call_args[0]["text"] == "Test input"

    # Verify events were enqueued
    mock_event_queue.enqueue_event.assert_called()


@pytest.mark.asyncio
async def test_execute_streaming_mode_fallback_to_text_extraction(
    mock_strands_agent, mock_request_context, mock_event_queue
):
    """Test that execute falls back to text extraction when no A2A parts are available."""

    async def mock_stream(content_blocks):
        """Mock streaming function that yields data events."""
        yield {"data": "Test chunk"}
        yield {"result": MagicMock(spec=SAAgentResult)}

    # Setup mock agent streaming
    mock_strands_agent.stream_async = MagicMock(return_value=mock_stream([]))

    # Create executor
    executor = StrandsA2AExecutor(mock_strands_agent)

    # Mock the task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.context_id = "test-context-id"
    mock_request_context.current_task = mock_task

    # Mock message without parts attribute
    mock_message = MagicMock()
    delattr(mock_message, "parts")  # Remove parts attribute
    mock_request_context.message = mock_message
    mock_request_context.get_user_input.return_value = "Fallback input"

    await executor.execute(mock_request_context, mock_event_queue)

    # Verify agent was called with fallback ContentBlock
    mock_strands_agent.stream_async.assert_called_once()
    call_args = mock_strands_agent.stream_async.call_args[0][0]
    assert isinstance(call_args, list)
    assert len(call_args) == 1
    assert call_args[0]["text"] == "Fallback input"


@pytest.mark.asyncio
async def test_execute_creates_task_when_none_exists(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that execute creates a new task when none exists."""

    async def mock_stream(content_blocks):
        """Mock streaming function that yields data events."""
        yield {"data": "Test chunk"}
        yield {"result": MagicMock(spec=SAAgentResult)}

    # Setup mock agent streaming
    mock_strands_agent.stream_async = MagicMock(return_value=mock_stream([]))

    # Create executor
    executor = StrandsA2AExecutor(mock_strands_agent)

    # Mock no existing task
    mock_request_context.current_task = None

    # Mock message with parts
    from a2a.types import TextPart

    mock_message = MagicMock()
    text_part = MagicMock(spec=TextPart)
    text_part.text = "Test input"
    part = MagicMock()
    part.root = text_part
    mock_message.parts = [part]
    mock_request_context.message = mock_message

    with patch("strands.multiagent.a2a.executor.new_task") as mock_new_task:
        mock_new_task.return_value = MagicMock(id="new-task-id", context_id="new-context-id")

        await executor.execute(mock_request_context, mock_event_queue)

    # Verify task creation and completion events were enqueued
    assert mock_event_queue.enqueue_event.call_count >= 1
    mock_new_task.assert_called_once()


@pytest.mark.asyncio
async def test_execute_streaming_mode_handles_agent_exception(
    mock_strands_agent, mock_request_context, mock_event_queue
):
    """Test that execute handles agent exceptions correctly in streaming mode."""

    # Setup mock agent to raise exception when stream_async is called
    mock_strands_agent.stream_async = MagicMock(side_effect=Exception("Agent error"))

    # Create executor
    executor = StrandsA2AExecutor(mock_strands_agent)

    # Mock the task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.context_id = "test-context-id"
    mock_request_context.current_task = mock_task

    # Mock message with parts
    from a2a.types import TextPart

    mock_message = MagicMock()
    text_part = MagicMock(spec=TextPart)
    text_part.text = "Test input"
    part = MagicMock()
    part.root = text_part
    mock_message.parts = [part]
    mock_request_context.message = mock_message

    with pytest.raises(ServerError):
        await executor.execute(mock_request_context, mock_event_queue)

    # Verify agent was called
    mock_strands_agent.stream_async.assert_called_once()


@pytest.mark.asyncio
async def test_cancel_raises_unsupported_operation_error(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that cancel raises UnsupportedOperationError."""
    executor = StrandsA2AExecutor(mock_strands_agent)

    with pytest.raises(ServerError) as excinfo:
        await executor.cancel(mock_request_context, mock_event_queue)

    # Verify the error is a ServerError containing an UnsupportedOperationError
    assert isinstance(excinfo.value.error, UnsupportedOperationError)


@pytest.mark.asyncio
async def test_handle_agent_result_with_none_result(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that _handle_agent_result handles None result correctly."""
    executor = StrandsA2AExecutor(mock_strands_agent)

    # Mock the task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.context_id = "test-context-id"
    mock_request_context.current_task = mock_task

    # Mock TaskUpdater
    mock_updater = MagicMock()
    mock_updater.complete = AsyncMock()
    mock_updater.add_artifact = AsyncMock()

    # Call _handle_agent_result with None
    await executor._handle_agent_result(None, mock_updater)

    # Verify completion was called
    mock_updater.complete.assert_called_once()


@pytest.mark.asyncio
async def test_handle_agent_result_with_result_but_no_message(
    mock_strands_agent, mock_request_context, mock_event_queue
):
    """Test that _handle_agent_result handles result with no message correctly."""
    executor = StrandsA2AExecutor(mock_strands_agent)

    # Mock the task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.context_id = "test-context-id"
    mock_request_context.current_task = mock_task

    # Mock TaskUpdater
    mock_updater = MagicMock()
    mock_updater.complete = AsyncMock()
    mock_updater.add_artifact = AsyncMock()

    # Create result with no message
    mock_result = MagicMock(spec=SAAgentResult)
    mock_result.message = None

    # Call _handle_agent_result
    await executor._handle_agent_result(mock_result, mock_updater)

    # Verify completion was called
    mock_updater.complete.assert_called_once()


@pytest.mark.asyncio
async def test_handle_agent_result_with_content(mock_strands_agent):
    """Test that _handle_agent_result handles result with content correctly."""
    executor = StrandsA2AExecutor(mock_strands_agent)

    # Mock TaskUpdater
    mock_updater = MagicMock()
    mock_updater.complete = AsyncMock()
    mock_updater.add_artifact = AsyncMock()

    # Create result with content
    mock_result = MagicMock(spec=SAAgentResult)
    mock_result.__str__ = MagicMock(return_value="Test response content")

    # Call _handle_agent_result
    await executor._handle_agent_result(mock_result, mock_updater)

    # Verify artifact was added and task completed
    mock_updater.add_artifact.assert_called_once()
    mock_updater.complete.assert_called_once()

    # Check that the artifact contains the expected content
    call_args = mock_updater.add_artifact.call_args[0][0]
    assert len(call_args) == 1
    assert call_args[0].root.text == "Test response content"


def test_handle_conversion_error():
    """Test that conversion handles errors gracefully."""
    executor = StrandsA2AExecutor(MagicMock())

    # Mock Part that will raise an exception during processing
    problematic_part = MagicMock()
    problematic_part.root = None  # This should cause an AttributeError

    # Should not raise an exception, but return empty list or handle gracefully
    result = executor._convert_a2a_parts_to_content_blocks([problematic_part])

    # The method should handle the error and continue
    assert isinstance(result, list)


def test_convert_a2a_parts_to_content_blocks_empty_list():
    """Test conversion with empty parts list."""
    executor = StrandsA2AExecutor(MagicMock())

    result = executor._convert_a2a_parts_to_content_blocks([])

    assert result == []


def test_convert_a2a_parts_to_content_blocks_file_part_no_name():
    """Test conversion of FilePart with no file name."""
    from a2a.types import FilePart

    executor = StrandsA2AExecutor(MagicMock())

    # Mock file object without name
    file_obj = MagicMock()
    delattr(file_obj, "name")  # Remove name attribute
    file_obj.mime_type = "text/plain"
    file_obj.bytes = b"test content"
    file_obj.uri = None

    # Mock FilePart with proper spec
    file_part = MagicMock(spec=FilePart)
    file_part.file = file_obj

    # Mock Part with FilePart root
    part = MagicMock()
    part.root = file_part

    result = executor._convert_a2a_parts_to_content_blocks([part])

    assert len(result) == 1
    content_block = result[0]
    assert "document" in content_block
    assert content_block["document"]["name"] == "FileNameNotProvided"  # Should use default


def test_convert_a2a_parts_to_content_blocks_file_part_no_mime_type():
    """Test conversion of FilePart with no MIME type."""
    from a2a.types import FilePart

    executor = StrandsA2AExecutor(MagicMock())

    # Mock file object without MIME type
    file_obj = MagicMock()
    file_obj.name = "test_file"
    delattr(file_obj, "mime_type")
    file_obj.bytes = b"test content"
    file_obj.uri = None

    # Mock FilePart with proper spec
    file_part = MagicMock(spec=FilePart)
    file_part.file = file_obj

    # Mock Part with FilePart root
    part = MagicMock()
    part.root = file_part

    result = executor._convert_a2a_parts_to_content_blocks([part])

    assert len(result) == 1
    content_block = result[0]
    assert "document" in content_block  # Should default to document with unknown type
    assert content_block["document"]["format"] == "txt"  # Should use default format for unknown file type


def test_convert_a2a_parts_to_content_blocks_file_part_no_bytes_no_uri():
    """Test conversion of FilePart with neither bytes nor URI."""
    from a2a.types import FilePart

    executor = StrandsA2AExecutor(MagicMock())

    # Mock file object without bytes or URI
    file_obj = MagicMock()
    file_obj.name = "test_file.txt"
    file_obj.mime_type = "text/plain"
    file_obj.bytes = None
    file_obj.uri = None

    # Mock FilePart with proper spec
    file_part = MagicMock(spec=FilePart)
    file_part.file = file_obj

    # Mock Part with FilePart root
    part = MagicMock()
    part.root = file_part

    result = executor._convert_a2a_parts_to_content_blocks([part])

    # Should return empty list since no fallback case exists
    assert len(result) == 0


def test_convert_a2a_parts_to_content_blocks_data_part_serialization_error():
    """Test conversion of DataPart with non-serializable data."""
    from a2a.types import DataPart

    executor = StrandsA2AExecutor(MagicMock())

    # Create non-serializable data (e.g., a function)
    def non_serializable():
        pass

    # Mock DataPart with proper spec
    data_part = MagicMock(spec=DataPart)
    data_part.data = {"function": non_serializable}  # This will cause JSON serialization to fail

    # Mock Part with DataPart root
    part = MagicMock()
    part.root = data_part

    # Should not raise an exception, should handle gracefully
    result = executor._convert_a2a_parts_to_content_blocks([part])

    # The error handling should result in an empty list or the part being skipped
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_execute_with_mixed_part_types(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test execute with a message containing mixed A2A part types."""
    from a2a.types import DataPart, FilePart, TextPart

    async def mock_stream(content_blocks):
        """Mock streaming function."""
        yield {"data": "Processing mixed content"}
        yield {"result": MagicMock(spec=SAAgentResult)}

    # Setup mock agent streaming
    mock_strands_agent.stream_async = MagicMock(return_value=mock_stream([]))

    # Create executor
    executor = StrandsA2AExecutor(mock_strands_agent)

    # Mock the task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.context_id = "test-context-id"
    mock_request_context.current_task = mock_task

    # Create mixed parts
    text_part = MagicMock(spec=TextPart)
    text_part.text = "Hello"
    text_part_mock = MagicMock()
    text_part_mock.root = text_part

    # File part with bytes
    file_obj = MagicMock()
    file_obj.name = "image.png"
    file_obj.mime_type = "image/png"
    file_obj.bytes = b"fake_image"
    file_obj.uri = None
    file_part = MagicMock(spec=FilePart)
    file_part.file = file_obj
    file_part_mock = MagicMock()
    file_part_mock.root = file_part

    # Data part
    data_part = MagicMock(spec=DataPart)
    data_part.data = {"key": "value"}
    data_part_mock = MagicMock()
    data_part_mock.root = data_part

    # Mock message with mixed parts
    mock_message = MagicMock()
    mock_message.parts = [text_part_mock, file_part_mock, data_part_mock]
    mock_request_context.message = mock_message

    await executor.execute(mock_request_context, mock_event_queue)

    # Verify agent was called with ContentBlock list containing all types
    mock_strands_agent.stream_async.assert_called_once()
    call_args = mock_strands_agent.stream_async.call_args[0][0]
    assert isinstance(call_args, list)
    assert len(call_args) == 3  # Should have converted all 3 parts

    # Check that we have text, image, and structured data
    has_text = any("text" in block for block in call_args)
    has_image = any("image" in block for block in call_args)
    has_structured_data = any("text" in block and "[Structured Data]" in block.get("text", "") for block in call_args)

    assert has_text
    assert has_image
    assert has_structured_data


def test_integration_example():
    """Integration test example showing how A2A Parts are converted to ContentBlocks.

    This test serves as documentation for the conversion functionality.
    """
    from a2a.types import DataPart, FilePart, TextPart

    executor = StrandsA2AExecutor(MagicMock())

    # Example 1: Text content
    text_part = MagicMock(spec=TextPart)
    text_part.text = "Hello, this is a text message"
    text_part_mock = MagicMock()
    text_part_mock.root = text_part

    # Example 2: Image file
    image_bytes = b"fake_image_content"
    image_file = MagicMock()
    image_file.name = "photo.jpg"
    image_file.mime_type = "image/jpeg"
    image_file.bytes = image_bytes
    image_file.uri = None

    image_part = MagicMock(spec=FilePart)
    image_part.file = image_file
    image_part_mock = MagicMock()
    image_part_mock.root = image_part

    # Example 3: Document file
    doc_bytes = b"PDF document content"
    doc_file = MagicMock()
    doc_file.name = "report.pdf"
    doc_file.mime_type = "application/pdf"
    doc_file.bytes = doc_bytes
    doc_file.uri = None

    doc_part = MagicMock(spec=FilePart)
    doc_part.file = doc_file
    doc_part_mock = MagicMock()
    doc_part_mock.root = doc_part

    # Example 4: Structured data
    data_part = MagicMock(spec=DataPart)
    data_part.data = {"user": "john_doe", "action": "upload_file", "timestamp": "2023-12-01T10:00:00Z"}
    data_part_mock = MagicMock()
    data_part_mock.root = data_part

    # Convert all parts to ContentBlocks
    parts = [text_part_mock, image_part_mock, doc_part_mock, data_part_mock]
    content_blocks = executor._convert_a2a_parts_to_content_blocks(parts)

    # Verify conversion results
    assert len(content_blocks) == 4

    # Text part becomes text ContentBlock
    assert content_blocks[0]["text"] == "Hello, this is a text message"

    # Image part becomes image ContentBlock with proper format and bytes
    assert "image" in content_blocks[1]
    assert content_blocks[1]["image"]["format"] == "jpeg"
    assert content_blocks[1]["image"]["source"]["bytes"] == image_bytes

    # Document part becomes document ContentBlock
    assert "document" in content_blocks[2]
    assert content_blocks[2]["document"]["format"] == "pdf"
    assert content_blocks[2]["document"]["name"] == "report"  # Extension stripped
    assert content_blocks[2]["document"]["source"]["bytes"] == doc_bytes

    # Data part becomes text ContentBlock with JSON representation
    assert "text" in content_blocks[3]
    assert "[Structured Data]" in content_blocks[3]["text"]
    assert "john_doe" in content_blocks[3]["text"]
    assert "upload_file" in content_blocks[3]["text"]


def test_default_formats_modularization():
    """Test that DEFAULT_FORMATS mapping works correctly for modular format defaults."""
    executor = StrandsA2AExecutor(MagicMock())

    # Test that DEFAULT_FORMATS contains expected mappings
    assert hasattr(executor, "DEFAULT_FORMATS")
    assert executor.DEFAULT_FORMATS["document"] == "txt"
    assert executor.DEFAULT_FORMATS["image"] == "png"
    assert executor.DEFAULT_FORMATS["video"] == "mp4"
    assert executor.DEFAULT_FORMATS["unknown"] == "txt"

    # Test format selection with None mime_type
    assert executor._get_file_format_from_mime_type(None, "document") == "txt"
    assert executor._get_file_format_from_mime_type(None, "image") == "png"
    assert executor._get_file_format_from_mime_type(None, "video") == "mp4"
    assert executor._get_file_format_from_mime_type(None, "unknown") == "txt"
    assert executor._get_file_format_from_mime_type(None, "nonexistent") == "txt"  # fallback

    # Test format selection with empty mime_type
    assert executor._get_file_format_from_mime_type("", "document") == "txt"
    assert executor._get_file_format_from_mime_type("", "image") == "png"
    assert executor._get_file_format_from_mime_type("", "video") == "mp4"
