"""Tests for AgentAsTool - the agent-as-tool adapter."""

import logging
from unittest.mock import MagicMock

import pytest

from strands.agent.agent_as_tool import AgentAsTool
from strands.agent.agent_result import AgentResult
from strands.telemetry.metrics import EventLoopMetrics
from strands.types._events import ToolResultEvent


async def _mock_stream_async(result, intermediate_events=None):
    """Helper that yields intermediate events then the final result event."""
    for event in intermediate_events or []:
        yield event
    yield {"result": result}


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.name = "test_agent"
    agent.description = "A test agent"
    return agent


@pytest.fixture
def tool(mock_agent):
    return AgentAsTool(mock_agent, name="test_agent", description="A test agent")


@pytest.fixture
def tool_use():
    return {
        "toolUseId": "tool-123",
        "name": "test_agent",
        "input": {"input": "hello"},
    }


@pytest.fixture
def agent_result():
    return AgentResult(
        stop_reason="end_turn",
        message={"role": "assistant", "content": [{"text": "response text"}]},
        metrics=EventLoopMetrics(),
        state={},
    )


# --- init ---


def test_init_sets_name(mock_agent):
    tool = AgentAsTool(mock_agent, name="my_tool", description="desc")
    assert tool.tool_name == "my_tool"


def test_init_sets_description(mock_agent):
    tool = AgentAsTool(mock_agent, name="my_tool", description="custom desc")
    assert tool._description == "custom desc"


def test_init_stores_agent_reference(mock_agent, tool):
    assert tool.agent is mock_agent


# --- properties ---


def test_tool_name(tool):
    assert tool.tool_name == "test_agent"


def test_tool_type(tool):
    assert tool.tool_type == "agent"


def test_tool_spec_name(tool):
    assert tool.tool_spec["name"] == "test_agent"


def test_tool_spec_description(tool):
    assert tool.tool_spec["description"] == "A test agent"


def test_tool_spec_input_schema(tool):
    schema = tool.tool_spec["inputSchema"]["json"]
    assert schema["type"] == "object"
    assert "input" in schema["properties"]
    assert schema["properties"]["input"]["type"] == "string"
    assert "preserve_context" in schema["properties"]
    assert schema["properties"]["preserve_context"]["type"] == "boolean"
    assert schema["required"] == ["input"]


def test_display_properties(tool):
    props = tool.get_display_properties()
    assert props["Agent"] == "test_agent"
    assert props["Type"] == "agent"


# --- stream ---


@pytest.mark.asyncio
async def test_stream_success(tool, mock_agent, tool_use, agent_result):
    mock_agent.stream_async.return_value = _mock_stream_async(agent_result)

    events = [event async for event in tool.stream(tool_use, {})]

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(result_events) == 1
    assert result_events[0]["tool_result"]["status"] == "success"
    assert result_events[0]["tool_result"]["content"][0]["text"] == "response text\n"


@pytest.mark.asyncio
async def test_stream_passes_input_to_agent(tool, mock_agent, tool_use, agent_result):
    mock_agent.stream_async.return_value = _mock_stream_async(agent_result)

    async for _ in tool.stream(tool_use, {}):
        pass

    mock_agent.stream_async.assert_called_once_with("hello")


@pytest.mark.asyncio
async def test_stream_empty_input(tool, mock_agent, agent_result):
    empty_tool_use = {
        "toolUseId": "tool-123",
        "name": "test_agent",
        "input": {},
    }
    mock_agent.stream_async.return_value = _mock_stream_async(agent_result)

    async for _ in tool.stream(empty_tool_use, {}):
        pass

    mock_agent.stream_async.assert_called_once_with("")


@pytest.mark.asyncio
async def test_stream_error(tool, mock_agent, tool_use):
    mock_agent.stream_async.side_effect = RuntimeError("boom")

    events = [event async for event in tool.stream(tool_use, {})]

    assert len(events) == 1
    assert events[0]["tool_result"]["status"] == "error"
    assert "boom" in events[0]["tool_result"]["content"][0]["text"]


@pytest.mark.asyncio
async def test_stream_propagates_tool_use_id(tool, mock_agent, tool_use, agent_result):
    mock_agent.stream_async.return_value = _mock_stream_async(agent_result)

    events = [event async for event in tool.stream(tool_use, {})]

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert result_events[0]["tool_result"]["toolUseId"] == "tool-123"


@pytest.mark.asyncio
async def test_stream_forwards_intermediate_events(tool, mock_agent, tool_use, agent_result):
    intermediate = [{"data": "partial"}, {"data": "more"}]
    mock_agent.stream_async.return_value = _mock_stream_async(agent_result, intermediate)

    events = [event async for event in tool.stream(tool_use, {})]

    # Intermediate events are yielded as-is (raw dicts); wrapping in ToolStreamEvent happens in the caller
    non_result_events = [e for e in events if not isinstance(e, ToolResultEvent)]
    assert len(non_result_events) == 2
    assert non_result_events[0]["data"] == "partial"
    assert non_result_events[1]["data"] == "more"


@pytest.mark.asyncio
async def test_stream_no_result_yields_error(tool, mock_agent, tool_use):
    async def _empty_stream():
        return
        yield  # noqa: RET504 - make it an async generator

    mock_agent.stream_async.return_value = _empty_stream()

    events = [event async for event in tool.stream(tool_use, {})]

    assert len(events) == 1
    assert events[0]["tool_result"]["status"] == "error"
    assert "did not produce a result" in events[0]["tool_result"]["content"][0]["text"]


@pytest.mark.asyncio
async def test_stream_structured_output(tool, mock_agent, tool_use):
    from pydantic import BaseModel

    class MyOutput(BaseModel):
        answer: str

    structured = MyOutput(answer="42")
    result = AgentResult(
        stop_reason="end_turn",
        message={"role": "assistant", "content": [{"text": "ignored"}]},
        metrics=EventLoopMetrics(),
        state={},
        structured_output=structured,
    )
    mock_agent.stream_async.return_value = _mock_stream_async(result)

    events = [event async for event in tool.stream(tool_use, {})]

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert result_events[0]["tool_result"]["status"] == "success"
    assert result_events[0]["tool_result"]["content"][0]["json"] == {"answer": "42"}


@pytest.mark.asyncio
async def test_stream_string_input(tool, mock_agent, agent_result):
    """When tool_use input is a plain string rather than a dict."""
    tool_use = {
        "toolUseId": "tool-123",
        "name": "test_agent",
        "input": "direct string",
    }
    mock_agent.stream_async.return_value = _mock_stream_async(agent_result)

    async for _ in tool.stream(tool_use, {}):
        pass

    mock_agent.stream_async.assert_called_once_with("direct string")


# --- preserve_context ---


class _FakeAgent:
    """Minimal fake agent with a real messages list for preserve_context tests."""

    def __init__(self):
        self.name = "fake_agent"
        self.messages: list = []

    async def invoke_async(self, prompt=None, **kwargs):
        pass

    def __call__(self, prompt=None, **kwargs):
        pass

    def stream_async(self, prompt=None, **kwargs):
        return _mock_stream_async(
            AgentResult(
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "ok"}]},
                metrics=EventLoopMetrics(),
                state={},
            )
        )


@pytest.mark.asyncio
async def test_stream_clears_context_when_preserve_context_false():
    agent = _FakeAgent()
    agent.messages = [{"role": "user", "content": [{"text": "old"}]}]
    tool = AgentAsTool(agent, name="fake_agent", description="desc")

    tool_use = {
        "toolUseId": "tool-123",
        "name": "fake_agent",
        "input": {"input": "hello", "preserve_context": False},
    }

    async for _ in tool.stream(tool_use, {}):
        pass

    assert agent.messages == []


@pytest.mark.asyncio
async def test_stream_preserves_context_by_default():
    agent = _FakeAgent()
    agent.messages = [{"role": "user", "content": [{"text": "old"}]}]
    tool = AgentAsTool(agent, name="fake_agent", description="desc")

    tool_use = {
        "toolUseId": "tool-123",
        "name": "fake_agent",
        "input": {"input": "hello"},
    }

    async for _ in tool.stream(tool_use, {}):
        pass

    assert len(agent.messages) >= 1


@pytest.mark.asyncio
async def test_stream_preserves_context_when_explicitly_true():
    agent = _FakeAgent()
    agent.messages = [{"role": "user", "content": [{"text": "old"}]}]
    tool = AgentAsTool(agent, name="fake_agent", description="desc")

    tool_use = {
        "toolUseId": "tool-123",
        "name": "fake_agent",
        "input": {"input": "hello", "preserve_context": True},
    }

    async for _ in tool.stream(tool_use, {}):
        pass

    assert len(agent.messages) >= 1


@pytest.mark.asyncio
async def test_stream_preserve_context_false_warns_when_no_messages_attr(caplog):
    """Agent without a messages attribute should log a warning."""

    class _NoMessagesAgent:
        name = "bare_agent"

        async def invoke_async(self, prompt=None, **kwargs):
            pass

        def __call__(self, prompt=None, **kwargs):
            pass

        def stream_async(self, prompt=None, **kwargs):
            return _mock_stream_async(
                AgentResult(
                    stop_reason="end_turn",
                    message={"role": "assistant", "content": [{"text": "ok"}]},
                    metrics=EventLoopMetrics(),
                    state={},
                )
            )

    agent = _NoMessagesAgent()
    tool = AgentAsTool(agent, name="bare_agent", description="desc")

    tool_use = {
        "toolUseId": "tool-123",
        "name": "bare_agent",
        "input": {"input": "hello", "preserve_context": False},
    }

    with caplog.at_level(logging.WARNING, logger="strands.agent.agent_as_tool"):
        async for _ in tool.stream(tool_use, {}):
            pass

    assert "preserve_context=false requested" in caplog.text
