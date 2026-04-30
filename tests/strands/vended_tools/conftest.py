"""Shared fixtures for vended tools tests."""

import uuid
from unittest.mock import MagicMock

import pytest

from strands.agent.state import AgentState
from strands.sandbox.base import StreamChunk
from strands.sandbox.host import HostSandbox
from strands.types.tools import ToolContext, ToolUse


@pytest.fixture
def sandbox(tmp_path):
    """Create a HostSandbox for testing."""
    return HostSandbox(working_dir=str(tmp_path))


@pytest.fixture
def agent_state():
    """Create a fresh AgentState."""
    return AgentState()


@pytest.fixture
def mock_agent(sandbox, agent_state):
    """Create a mock agent with sandbox and state."""
    agent = MagicMock()
    agent.sandbox = sandbox
    agent.state = agent_state
    return agent


@pytest.fixture
def tool_use():
    """Create a mock tool use."""
    return ToolUse(
        toolUseId=str(uuid.uuid4()),
        name="test_tool",
        input={},
    )


@pytest.fixture
def tool_context(mock_agent, tool_use):
    """Create a ToolContext for testing."""
    return ToolContext(
        tool_use=tool_use,
        agent=mock_agent,
        invocation_state={},
    )


async def collect_generator(gen):
    """Collect all values from an async generator.

    Returns (stream_chunks, final_result) where stream_chunks are all
    StreamChunk objects yielded, and final_result is the last non-StreamChunk
    value (the formatted result string).
    """
    chunks = []
    final = None
    async for item in gen:
        if isinstance(item, StreamChunk):
            chunks.append(item)
        else:
            final = item
    return chunks, final
