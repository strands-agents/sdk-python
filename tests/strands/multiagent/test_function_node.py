"""Unit tests for FunctionNode implementation."""

from unittest.mock import Mock, patch

import pytest

from strands.multiagent.base import Status
from strands.multiagent.function_node import FunctionNode
from strands.types.content import ContentBlock, Message


@pytest.fixture
def mock_tracer():
    """Create a mock tracer for testing."""
    tracer = Mock()
    span = Mock()
    span.__enter__ = Mock(return_value=span)
    span.__exit__ = Mock(return_value=None)
    tracer.start_multiagent_span.return_value = span
    return tracer


@pytest.mark.asyncio
async def test_invoke_async_string_input_success(mock_tracer):
    """Test successful function execution with string input."""

    def test_function(task, invocation_state=None, **kwargs):
        return f"Processed: {task}"

    node = FunctionNode(test_function, "string_test")

    with patch.object(node, "tracer", mock_tracer):
        result = await node.invoke_async("test input")

        assert result.status == Status.COMPLETED
        assert "string_test" in result.results
        assert result.results["string_test"].status == Status.COMPLETED
        assert result.accumulated_usage["inputTokens"] == 0
        assert result.accumulated_usage["outputTokens"] == 0


@pytest.mark.asyncio
async def test_invoke_async_content_block_input_success(mock_tracer):
    """Test successful function execution with ContentBlock input."""

    def test_function(task, invocation_state=None, **kwargs):
        return "ContentBlock processed"

    node = FunctionNode(test_function, "content_block_test")
    content_blocks = [ContentBlock(text="First block"), ContentBlock(text="Second block")]

    with patch.object(node, "tracer", mock_tracer):
        result = await node.invoke_async(content_blocks)

        assert result.status == Status.COMPLETED
        assert "content_block_test" in result.results
        node_result = result.results["content_block_test"]
        assert node_result.status == Status.COMPLETED


@pytest.mark.asyncio
async def test_invoke_async_with_kwargs(mock_tracer):
    """Test function execution with additional kwargs."""

    def test_function(task, invocation_state=None, **kwargs):
        extra_param = kwargs.get("extra_param", "none")
        return f"Extra: {extra_param}"

    node = FunctionNode(test_function, "kwargs_test")

    with patch.object(node, "tracer", mock_tracer):
        result = await node.invoke_async("test", None, extra_param="test_value")

        assert result.status == Status.COMPLETED


@pytest.mark.asyncio
async def test_invoke_async_function_exception(mock_tracer):
    """Test proper exception handling when function raises an error."""

    def failing_function(task, invocation_state=None, **kwargs):
        raise ValueError("Test exception")

    node = FunctionNode(failing_function, "exception_test")

    with patch.object(node, "tracer", mock_tracer):
        result = await node.invoke_async("test input")

        assert result.status == Status.FAILED
        assert "exception_test" in result.results
        node_result = result.results["exception_test"]
        assert node_result.status == Status.FAILED
        assert isinstance(node_result.result, ValueError)
        assert str(node_result.result) == "Test exception"


@pytest.mark.asyncio
async def test_function_returns_string(mock_tracer):
    """Test function returning string."""

    def string_function(task, invocation_state=None, **kwargs):
        return "Hello World"

    node = FunctionNode(string_function, "string_node")

    with patch.object(node, "tracer", mock_tracer):
        result = await node.invoke_async("test")

        agent_result = result.results["string_node"].result
        assert agent_result.message["content"][0]["text"] == "Hello World"


@pytest.mark.asyncio
async def test_function_returns_content_blocks(mock_tracer):
    """Test function returning list of ContentBlocks."""

    def content_block_function(task, invocation_state=None, **kwargs):
        return [ContentBlock(text="Block 1"), ContentBlock(text="Block 2")]

    node = FunctionNode(content_block_function, "content_node")

    with patch.object(node, "tracer", mock_tracer):
        result = await node.invoke_async("test")

        agent_result = result.results["content_node"].result
        assert len(agent_result.message["content"]) == 2
        assert agent_result.message["content"][0]["text"] == "Block 1"
        assert agent_result.message["content"][1]["text"] == "Block 2"


@pytest.mark.asyncio
async def test_function_returns_message(mock_tracer):
    """Test function returning Message."""

    def message_function(task, invocation_state=None, **kwargs):
        return Message(role="user", content=[ContentBlock(text="Custom message")])

    node = FunctionNode(message_function, "message_node")

    with patch.object(node, "tracer", mock_tracer):
        result = await node.invoke_async("test")

        agent_result = result.results["message_node"].result
        assert agent_result.message["role"] == "user"
        assert agent_result.message["content"][0]["text"] == "Custom message"
