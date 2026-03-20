"""Tests for AgentAsTool - the agent-as-tool adapter."""

from unittest.mock import MagicMock

import pytest

from strands.agent import AgentAsTool
from strands.agent.agent_result import AgentResult
from strands.telemetry.metrics import EventLoopMetrics
from strands.types._events import AgentAsToolStreamEvent, ToolResultEvent, ToolStreamEvent


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


def test_init(mock_agent):
    tool = AgentAsTool(mock_agent, name="my_tool", description="custom desc")
    assert tool.tool_name == "my_tool"
    assert tool._description == "custom desc"
    assert tool.agent is mock_agent


def test_init_preserve_context_defaults_true(mock_agent):
    tool = AgentAsTool(mock_agent, name="t", description="d")
    assert tool._preserve_context is True


def test_init_preserve_context_false(fake_agent):
    tool = AgentAsTool(fake_agent, name="t", description="d", preserve_context=False)
    assert tool._preserve_context is False


# --- properties ---


def test_tool_properties(tool):
    assert tool.tool_name == "test_agent"
    assert tool.tool_type == "agent"

    spec = tool.tool_spec
    assert spec["name"] == "test_agent"
    assert spec["description"] == "A test agent"

    schema = spec["inputSchema"]["json"]
    assert schema["type"] == "object"
    assert "input" in schema["properties"]
    assert schema["properties"]["input"]["type"] == "string"
    assert schema["required"] == ["input"]

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
async def test_stream_string_input(tool, mock_agent, agent_result):
    tool_use = {
        "toolUseId": "tool-123",
        "name": "test_agent",
        "input": "direct string",
    }
    mock_agent.stream_async.return_value = _mock_stream_async(agent_result)

    async for _ in tool.stream(tool_use, {}):
        pass

    mock_agent.stream_async.assert_called_once_with("direct string")


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

    stream_events = [e for e in events if isinstance(e, AgentAsToolStreamEvent)]
    assert len(stream_events) == 2
    assert stream_events[0]["tool_stream_event"]["data"]["data"] == "partial"
    assert stream_events[1]["tool_stream_event"]["data"]["data"] == "more"
    assert stream_events[0].agent_as_tool is tool
    assert stream_events[0].tool_use_id == "tool-123"


@pytest.mark.asyncio
async def test_stream_events_not_double_wrapped_by_executor(tool, mock_agent, tool_use, agent_result):
    """AgentAsToolStreamEvent is a ToolStreamEvent subclass, so the executor should pass it through directly."""
    intermediate = [{"data": "chunk"}]
    mock_agent.stream_async.return_value = _mock_stream_async(agent_result, intermediate)

    events = [event async for event in tool.stream(tool_use, {})]

    stream_events = [e for e in events if isinstance(e, AgentAsToolStreamEvent)]
    assert len(stream_events) == 1

    event = stream_events[0]
    # It's a ToolStreamEvent (so the executor yields it directly)
    assert isinstance(event, ToolStreamEvent)
    # But it's specifically an AgentAsToolStreamEvent (not re-wrapped)
    assert type(event) is AgentAsToolStreamEvent
    # And it references the originating AgentAsTool
    assert event.agent_as_tool is tool


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


# --- preserve_context ---


@pytest.fixture
def fake_agent():
    """A real Agent instance for preserve_context tests."""
    from strands.agent.agent import Agent

    return Agent(name="fake_agent", callback_handler=None)


@pytest.mark.asyncio
async def test_stream_resets_to_initial_state_when_preserve_context_false(fake_agent):
    fake_agent.messages = [{"role": "user", "content": [{"text": "initial"}]}]
    fake_agent.state.set("counter", 0)

    tool = AgentAsTool(fake_agent, name="fake_agent", description="desc", preserve_context=False)

    # Mutate agent state as if a previous invocation happened
    fake_agent.messages.append({"role": "assistant", "content": [{"text": "reply"}]})
    fake_agent.state.set("counter", 5)

    # Mock stream_async so we don't need a real model
    fake_agent.stream_async = lambda prompt, **kw: _mock_stream_async(
        AgentResult(
            stop_reason="end_turn",
            message={"role": "assistant", "content": [{"text": "ok"}]},
            metrics=EventLoopMetrics(),
            state={},
        )
    )

    tool_use = {
        "toolUseId": "tool-123",
        "name": "fake_agent",
        "input": {"input": "hello"},
    }

    async for _ in tool.stream(tool_use, {}):
        pass

    assert fake_agent.messages == [{"role": "user", "content": [{"text": "initial"}]}]
    assert fake_agent.state.get("counter") == 0


@pytest.mark.asyncio
async def test_stream_resets_on_every_invocation(fake_agent):
    """Each call should reset to the same initial snapshot, not to the previous call's state."""
    fake_agent.messages = [{"role": "user", "content": [{"text": "seed"}]}]
    fake_agent.state.set("count", 1)

    tool = AgentAsTool(fake_agent, name="fake_agent", description="desc", preserve_context=False)

    fake_agent.stream_async = lambda prompt, **kw: _mock_stream_async(
        AgentResult(
            stop_reason="end_turn",
            message={"role": "assistant", "content": [{"text": "ok"}]},
            metrics=EventLoopMetrics(),
            state={},
        )
    )

    tool_use = {
        "toolUseId": "tool-1",
        "name": "fake_agent",
        "input": {"input": "first"},
    }

    async for _ in tool.stream(tool_use, {}):
        pass
    fake_agent.messages.append({"role": "assistant", "content": [{"text": "added"}]})
    fake_agent.state.set("count", 99)

    tool_use["toolUseId"] = "tool-2"
    async for _ in tool.stream(tool_use, {}):
        pass

    assert fake_agent.messages == [{"role": "user", "content": [{"text": "seed"}]}]
    assert fake_agent.state.get("count") == 1


@pytest.mark.asyncio
async def test_stream_initial_snapshot_is_deep_copy(fake_agent):
    """Mutating the agent's messages after construction should not affect the snapshot."""
    fake_agent.messages = [{"role": "user", "content": [{"text": "original"}]}]

    tool = AgentAsTool(fake_agent, name="fake_agent", description="desc", preserve_context=False)

    fake_agent.messages[0]["content"][0]["text"] = "mutated"
    fake_agent.messages.append({"role": "assistant", "content": [{"text": "extra"}]})

    fake_agent.stream_async = lambda prompt, **kw: _mock_stream_async(
        AgentResult(
            stop_reason="end_turn",
            message={"role": "assistant", "content": [{"text": "ok"}]},
            metrics=EventLoopMetrics(),
            state={},
        )
    )

    tool_use = {
        "toolUseId": "tool-123",
        "name": "fake_agent",
        "input": {"input": "hello"},
    }

    async for _ in tool.stream(tool_use, {}):
        pass

    assert fake_agent.messages == [{"role": "user", "content": [{"text": "original"}]}]


@pytest.mark.asyncio
async def test_stream_resets_empty_initial_state_when_preserve_context_false(fake_agent):
    tool = AgentAsTool(fake_agent, name="fake_agent", description="desc", preserve_context=False)

    fake_agent.messages = [{"role": "user", "content": [{"text": "old"}]}]
    fake_agent.state.set("key", "value")

    fake_agent.stream_async = lambda prompt, **kw: _mock_stream_async(
        AgentResult(
            stop_reason="end_turn",
            message={"role": "assistant", "content": [{"text": "ok"}]},
            metrics=EventLoopMetrics(),
            state={},
        )
    )

    tool_use = {
        "toolUseId": "tool-123",
        "name": "fake_agent",
        "input": {"input": "hello"},
    }

    async for _ in tool.stream(tool_use, {}):
        pass

    assert fake_agent.messages == []
    assert fake_agent.state.get() == {}


@pytest.mark.asyncio
async def test_stream_preserves_context_by_default(fake_agent):
    fake_agent.messages = [{"role": "user", "content": [{"text": "old"}]}]
    fake_agent.state.set("key", "value")
    tool = AgentAsTool(fake_agent, name="fake_agent", description="desc")

    fake_agent.stream_async = lambda prompt, **kw: _mock_stream_async(
        AgentResult(
            stop_reason="end_turn",
            message={"role": "assistant", "content": [{"text": "ok"}]},
            metrics=EventLoopMetrics(),
            state={},
        )
    )

    tool_use = {
        "toolUseId": "tool-123",
        "name": "fake_agent",
        "input": {"input": "hello"},
    }

    async for _ in tool.stream(tool_use, {}):
        pass

    assert len(fake_agent.messages) >= 1
    assert fake_agent.state.get("key") == "value"


def test_preserve_context_false_requires_agent_instance():
    """preserve_context=False should raise TypeError for non-Agent instances."""

    class _NotAnAgent:
        name = "not_agent"

        async def invoke_async(self, prompt=None, **kwargs):
            pass

        def __call__(self, prompt=None, **kwargs):
            pass

        def stream_async(self, prompt=None, **kwargs):
            pass

    with pytest.raises(TypeError, match="requires an Agent instance"):
        AgentAsTool(_NotAnAgent(), name="bad", description="desc", preserve_context=False)
