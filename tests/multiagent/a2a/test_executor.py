"""Tests for the StrandsA2AExecutor class."""

import pytest
from a2a.types import UnsupportedOperationError
from a2a.utils.errors import ServerError

from strands.agent.agent_result import AgentResult
from strands.multiagent.a2a.executor import StrandsA2AExecutor
from strands.telemetry.metrics import EventLoopMetrics


class MockAgent:
    """Mock Strands Agent for testing."""

    def __init__(self, response_text="Test response"):
        """Initialize the mock agent with a predefined response."""
        self.response_text = response_text
        self.called_with = None

    def __call__(self, input_text):
        """Mock the agent call method."""
        self.called_with = input_text
        return AgentResult(
            stop_reason="end_turn",
            message={"content": [{"text": self.response_text}]},
            metrics=EventLoopMetrics(),
            state={},
        )


class MockEventQueue:
    """Mock EventQueue for testing."""

    def __init__(self):
        """Initialize the mock event queue."""
        self.events = []

    async def enqueue_event(self, event):
        """Mock the enqueue_event method."""
        self.events.append(event)
        return None


class MockRequestContext:
    """Mock RequestContext for testing."""

    def __init__(self, user_input="Test input"):
        """Initialize the mock request context."""
        self.user_input = user_input

    def get_user_input(self):
        """Mock the get_user_input method."""
        return self.user_input


@pytest.fixture
def mock_agent():
    """Create a mock Strands agent for testing."""
    return MockAgent()


@pytest.fixture
def executor(mock_agent):
    """Create a StrandsA2AExecutor for testing."""
    return StrandsA2AExecutor(mock_agent)


@pytest.fixture
def event_queue():
    """Create a mock event queue for testing."""
    return MockEventQueue()


@pytest.fixture
def request_context():
    """Create a mock request context for testing."""
    return MockRequestContext()


@pytest.mark.asyncio
async def test_execute(executor, event_queue, request_context):
    """Test that the execute method works correctly."""
    await executor.execute(request_context, event_queue)

    # Check that the agent was called with the correct input
    assert executor.agent.called_with == "Test input"

    # Check that an event was enqueued (we can't check the content directly)
    assert len(event_queue.events) == 1


@pytest.mark.asyncio
async def test_cancel(executor, event_queue, request_context):
    """Test that the cancel method raises the expected error."""
    with pytest.raises(ServerError) as excinfo:
        await executor.cancel(request_context, event_queue)

    # Check that the error contains an UnsupportedOperationError
    assert isinstance(excinfo.value.error, UnsupportedOperationError)
