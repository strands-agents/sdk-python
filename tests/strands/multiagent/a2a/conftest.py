"""Common fixtures for A2A module tests."""

from unittest.mock import AsyncMock, MagicMock, PropertyMock

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

    # Setup async methods
    agent.invoke_async = AsyncMock(return_value=mock_result)
    agent.stream_async = AsyncMock(return_value=iter([]))

    # Setup mock tool registry
    mock_tool_registry = MagicMock()
    mock_tool_registry.get_all_tools_config.return_value = {}
    agent.tool_registry = mock_tool_registry

    # Setup with_session_manager to return a copy of the agent
    def mock_with_session_manager(session_manager=None, request_metadata=None):
        """Create a copy of the agent with session manager."""
        agent_copy = MagicMock(spec=SAAgent)
        agent_copy.name = agent.name
        agent_copy.description = agent.description
        agent_copy.invoke_async = agent.invoke_async
        agent_copy.stream_async = agent.stream_async
        agent_copy.tool_registry = agent.tool_registry
        return agent_copy

    agent.with_session_manager = MagicMock(side_effect=mock_with_session_manager)

    return agent


@pytest.fixture
def mock_request_context():
    """Create a mock RequestContext for testing."""
    context = MagicMock(spec=RequestContext)
    context.get_user_input.return_value = "Test input"
    type(context).context_id = PropertyMock(return_value="test-context-id")
    return context


@pytest.fixture
def mock_event_queue():
    """Create a mock EventQueue for testing."""
    queue = MagicMock(spec=EventQueue)
    queue.enqueue_event = AsyncMock()
    return queue
