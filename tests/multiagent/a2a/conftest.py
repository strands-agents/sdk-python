"""Common fixtures for A2A module tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue

from strands.agent.agent import Agent as SAAgent
from strands.agent.agent_result import AgentResult as SAAgentResult


@pytest.fixture
def mock_strands_agent():
    """Create a mock Strands Agent for testing."""
    agent = MagicMock(spec=SAAgent)
    agent.name = "Test Agent"
    agent.description = "A test agent for unit testing"

    # Setup default response
    mock_result = MagicMock(spec=SAAgentResult)
    mock_result.message = {"content": [{"text": "Test response"}]}
    agent.return_value = mock_result

    return agent


@pytest.fixture
def mock_request_context():
    """Create a mock RequestContext for testing."""
    context = MagicMock(spec=RequestContext)
    context.get_user_input.return_value = "Test input"
    return context


@pytest.fixture
def mock_event_queue():
    """Create a mock EventQueue for testing."""
    queue = MagicMock(spec=EventQueue)
    queue.enqueue_event = AsyncMock()
    return queue
