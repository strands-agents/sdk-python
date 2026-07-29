"""Tests for _AgentAsTool - the agent-as-tool adapter."""

import json
from unittest.mock import MagicMock

import pytest

from strands import tool as strands_tool
from strands.agent._agent_as_tool import _AgentAsTool
from strands.agent.agent import Agent
from strands.agent.agent_result import AgentResult
from strands.agent.state import AgentState
from strands.hooks import BeforeToolCallEvent
from strands.interrupt import Interrupt, _InterruptState
from strands.telemetry.metrics import EventLoopMetrics
from strands.types._events import AgentAsToolStreamEvent, ToolInterruptEvent, ToolResultEvent, ToolStreamEvent
from tests.fixtures.mocked_model_provider import MockedModelProvider


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
    agent._interrupt_state = _InterruptState()
    # Real (serializable) values so resume-snapshot deepcopy works; a bare MagicMock cannot be copied.
    agent.messages = []
    agent.state = AgentState()
    return agent


@pytest.fixture
def fake_agent():
    """A real Agent instance for tests that need Agent-specific features."""
    from strands.agent.agent import Agent

    return Agent(name="fake_agent", callback_handler=None)


@pytest.fixture
def tool(mock_agent):
    return _AgentAsTool(mock_agent, name="test_agent", description="A test agent", preserve_context=True)


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
    tool = _AgentAsTool(mock_agent, name="my_tool", description="custom desc", preserve_context=True)
    assert tool.tool_name == "my_tool"
    assert tool._description == "custom desc"
    assert tool.agent is mock_agent


def test_init_description_defaults_to_agent_description(fake_agent):
    fake_agent.description = "Agent that researches topics"
    tool = _AgentAsTool(fake_agent, name="researcher", preserve_context=True)
    assert tool._description == "Agent that researches topics"


def test_init_description_defaults_to_generic_when_agent_has_none(fake_agent):
    tool = _AgentAsTool(fake_agent, name="researcher", preserve_context=True)
    assert tool._description == "Use the researcher agent as a tool by providing a natural language input"


def test_init_description_explicit_overrides_agent_description(fake_agent):
    fake_agent.description = "Agent that researches topics"
    tool = _AgentAsTool(fake_agent, name="researcher", description="custom", preserve_context=True)
    assert tool._description == "custom"


def test_init_preserve_context_defaults_false(fake_agent):
    tool = _AgentAsTool(fake_agent, name="t", description="d")
    assert tool._preserve_context is False


def test_init_preserve_context_true(mock_agent):
    tool = _AgentAsTool(mock_agent, name="t", description="d", preserve_context=True)
    assert tool._preserve_context is True


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
    # And it references the originating _AgentAsTool
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


@pytest.mark.asyncio
async def test_stream_resets_to_initial_state_when_preserve_context_false(fake_agent):
    fake_agent.messages = [{"role": "user", "content": [{"text": "initial"}]}]
    fake_agent.state.set("counter", 0)

    tool = _AgentAsTool(fake_agent, name="fake_agent", description="desc", preserve_context=False)

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

    tool = _AgentAsTool(fake_agent, name="fake_agent", description="desc", preserve_context=False)

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

    tool = _AgentAsTool(fake_agent, name="fake_agent", description="desc", preserve_context=False)

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
    tool = _AgentAsTool(fake_agent, name="fake_agent", description="desc", preserve_context=False)

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
async def test_stream_resets_context_by_default(fake_agent):
    """Default preserve_context=False means each invocation starts fresh."""
    fake_agent.messages = [{"role": "user", "content": [{"text": "old"}]}]
    fake_agent.state.set("key", "value")
    tool = _AgentAsTool(fake_agent, name="fake_agent", description="desc")

    # Mutate after construction
    fake_agent.messages.append({"role": "assistant", "content": [{"text": "extra"}]})
    fake_agent.state.set("key", "changed")

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

    # Should reset to construction-time snapshot
    assert fake_agent.messages == [{"role": "user", "content": [{"text": "old"}]}]
    assert fake_agent.state.get("key") == "value"


@pytest.mark.asyncio
async def test_stream_preserves_context_when_explicitly_true(fake_agent):
    fake_agent.messages = [{"role": "user", "content": [{"text": "old"}]}]
    fake_agent.state.set("key", "value")
    tool = _AgentAsTool(fake_agent, name="fake_agent", description="desc", preserve_context=True)

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


def test_preserve_context_false_rejects_session_manager(fake_agent):
    """preserve_context=False should raise ValueError when agent has a session manager."""
    fake_agent._session_manager = MagicMock()

    with pytest.raises(ValueError, match="cannot be used with an agent that has a session manager"):
        _AgentAsTool(fake_agent, name="t", description="d", preserve_context=False)


# --- interrupt propagation ---


@pytest.fixture
def interrupt_result():
    interrupt = Interrupt(id="interrupt-1", name="approval", reason="need approval")
    return AgentResult(
        stop_reason="interrupt",
        message={"role": "assistant", "content": [{"text": "pending"}]},
        metrics=EventLoopMetrics(),
        state={},
        interrupts=[interrupt],
    )


@pytest.mark.asyncio
async def test_stream_interrupt_yields_tool_interrupt_event(tool, mock_agent, tool_use, interrupt_result):
    """When the sub-agent returns an interrupt result, _AgentAsTool should yield ToolInterruptEvent."""
    mock_agent.stream_async.return_value = _mock_stream_async(interrupt_result)

    events = [event async for event in tool.stream(tool_use, {})]

    assert len(events) == 1
    assert isinstance(events[0], ToolInterruptEvent)
    # Interrupt IDs are namespaced by the outer tool_use_id to avoid collisions.
    assert len(events[0].interrupts) == 1
    assert events[0].interrupts[0].id == f"tool-123:{interrupt_result.interrupts[0].id}"
    assert events[0].interrupts[0].name == interrupt_result.interrupts[0].name
    assert events[0].interrupts[0].reason == interrupt_result.interrupts[0].reason
    assert events[0].tool_use_id == "tool-123"


@pytest.mark.asyncio
async def test_stream_interrupt_no_tool_result_appended(tool, mock_agent, tool_use, interrupt_result):
    """ToolInterruptEvent should not produce a ToolResultEvent."""
    mock_agent.stream_async.return_value = _mock_stream_async(interrupt_result)

    events = [event async for event in tool.stream(tool_use, {})]

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert result_events == []


@pytest.mark.asyncio
async def test_stream_interrupt_forwards_intermediate_events(tool, mock_agent, tool_use, interrupt_result):
    """Intermediate events should still be yielded before the interrupt."""
    intermediate = [{"data": "partial"}]
    mock_agent.stream_async.return_value = _mock_stream_async(interrupt_result, intermediate)

    events = [event async for event in tool.stream(tool_use, {})]

    stream_events = [e for e in events if isinstance(e, AgentAsToolStreamEvent)]
    interrupt_events = [e for e in events if isinstance(e, ToolInterruptEvent)]
    assert len(stream_events) == 1
    assert len(interrupt_events) == 1


# --- sub-agent interrupt resume across rehydration ---


@pytest.mark.asyncio
async def test_stream_interrupt_attaches_sub_agent_snapshot(fake_agent):
    """The propagated ToolInterruptEvent carries a serializable snapshot of the interrupted turn."""
    sub_tool_use_message = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "sub-1", "name": "get_my_roles", "input": {}}}],
    }
    interrupt = Interrupt(id="interrupt-1", name="approval", reason="need approval")
    interrupt_result = AgentResult(
        stop_reason="interrupt",
        message={"role": "assistant", "content": [{"text": "pending"}]},
        metrics=EventLoopMetrics(),
        state={},
        interrupts=[interrupt],
    )

    async def fake_stream(prompt, **kwargs):
        # Mirror the event loop: the sub-agent records its interrupt state before returning.
        fake_agent.messages = [{"role": "user", "content": [{"text": "do it"}]}, sub_tool_use_message]
        fake_agent.state.set("k", "v")
        fake_agent._interrupt_state.context = {"tool_use_message": sub_tool_use_message, "tool_results": []}
        fake_agent._interrupt_state.interrupts["interrupt-1"] = interrupt
        fake_agent._interrupt_state.activate()
        yield {"result": interrupt_result}

    fake_agent.stream_async = fake_stream

    tool = _AgentAsTool(fake_agent, name="fake_agent", description="desc", preserve_context=True)
    tool_use = {"toolUseId": "orch-1", "name": "fake_agent", "input": {"input": "do it"}}

    events = [event async for event in tool.stream(tool_use, {})]

    interrupt_events = [event for event in events if isinstance(event, ToolInterruptEvent)]
    assert len(interrupt_events) == 1
    snapshot = interrupt_events[0].sub_agent_snapshot
    assert snapshot is not None
    # The snapshot uses the versioned session_snapshot path for complete state.
    assert "session_snapshot" in snapshot
    session_data = snapshot["session_snapshot"]["data"]
    assert session_data["messages"] == [{"role": "user", "content": [{"text": "do it"}]}, sub_tool_use_message]
    assert session_data["state"] == {"k": "v"}
    assert session_data["interrupt_state"]["activated"] is True
    # The original sub-agent toolUseId is preserved, which is what keeps the interrupt id stable.
    persisted_tool_use = session_data["interrupt_state"]["context"]["tool_use_message"]["content"][0]["toolUse"]
    assert persisted_tool_use["toolUseId"] == "sub-1"
    assert "interrupt-1" in session_data["interrupt_state"]["interrupts"]
    # The interrupt_id_map maps parent-namespaced IDs back to sub-agent-local IDs.
    assert snapshot["interrupt_id_map"] == {"orch-1:interrupt-1": "interrupt-1"}


@pytest.mark.asyncio
async def test_stream_data_routed_resume_restores_snapshot_and_forwards_responses(fake_agent):
    """A fresh sub-agent resumes from a serialized snapshot routed via invocation_state.

    This is the stateless/distributed case: the sub-agent was rebuilt from scratch (no live
    interrupt state), so the in-process object-identity path cannot apply. The response must
    reach the sub-agent purely as data.

    Uses non-default model_state and conversation_manager_state to lock in that the restore
    half of the round trip actually applies these fields (not just messages/interrupt_state).
    """
    sub_tool_use_message = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "sub-1", "name": "get_my_roles", "input": {}}}],
    }
    snapshot = {
        "session_snapshot": {
            "scope": "agent",
            "schema_version": "1.0",
            "created_at": "2025-01-01T00:00:00+00:00",
            "data": {
                "messages": [{"role": "user", "content": [{"text": "what are my roles"}]}, sub_tool_use_message],
                "state": {"k": "v"},
                "conversation_manager_state": {
                    "__name__": "SlidingWindowConversationManager",
                    "removed_message_count": 5,
                    "model_call_count": 3,
                },
                "interrupt_state": {
                    "interrupts": {
                        "interrupt-1": {
                            "id": "interrupt-1",
                            "name": "approval",
                            "reason": "need approval",
                            "response": None,
                        }
                    },
                    "context": {"tool_use_message": sub_tool_use_message, "tool_results": []},
                    "activated": True,
                },
                "model_state": {"response_id": "resp_x"},
            },
            "app_data": {},
        },
        "interrupt_id_map": {"orch-1:interrupt-1": "interrupt-1"},
    }
    # Round-trip through JSON to simulate restoration from a session store.
    snapshot = json.loads(json.dumps(snapshot))

    invocation_state = {
        "_sub_agent_interrupt_resume": {
            "responses": [{"interruptResponse": {"interruptId": "orch-1:interrupt-1", "response": "APPROVE"}}],
            "snapshots": {"orch-1": snapshot},
        }
    }

    # The sub-agent is freshly built: not currently interrupted, default conversation manager state.
    assert fake_agent._interrupt_state.activated is False
    assert fake_agent._model_state == {}
    assert fake_agent.conversation_manager.removed_message_count == 0

    normal_result = AgentResult(
        stop_reason="end_turn",
        message={"role": "assistant", "content": [{"text": "approved"}]},
        metrics=EventLoopMetrics(),
        state={},
    )
    fake_agent.stream_async = MagicMock(return_value=_mock_stream_async(normal_result))

    tool = _AgentAsTool(fake_agent, name="fake_agent", description="desc", preserve_context=True)
    tool_use = {"toolUseId": "orch-1", "name": "fake_agent", "input": {"input": "ignored on resume"}}

    events = [event async for event in tool.stream(tool_use, invocation_state)]

    # Sub-agent state was rebuilt from the snapshot.
    assert fake_agent.messages == snapshot["session_snapshot"]["data"]["messages"]
    assert fake_agent.state.get() == {"k": "v"}
    assert fake_agent._interrupt_state.activated is True
    assert "interrupt-1" in fake_agent._interrupt_state.interrupts

    # model_state was restored from the snapshot (not left as the default empty dict).
    assert fake_agent._model_state == {"response_id": "resp_x"}

    # conversation_manager_state was restored (non-default values prove the restore path ran).
    assert fake_agent.conversation_manager.removed_message_count == 5
    assert fake_agent.conversation_manager._model_call_count == 3

    # The user's response was forwarded as the resume prompt, not the original tool input.
    agent_input = fake_agent.stream_async.call_args[0][0]
    assert agent_input == [{"interruptResponse": {"interruptId": "interrupt-1", "response": "APPROVE"}}]

    result_events = [event for event in events if isinstance(event, ToolResultEvent)]
    assert len(result_events) == 1
    assert result_events[0]["tool_result"]["status"] == "success"


@pytest.mark.asyncio
async def test_stream_data_routed_resume_ignores_snapshot_for_other_tool_use_id(fake_agent):
    """A snapshot keyed to a different tool call must not hijack this invocation."""
    snapshot = {
        "messages": [{"role": "user", "content": [{"text": "other"}]}],
        "state": {},
        "interrupt_state": {"interrupts": {}, "context": {}, "activated": False},
    }
    invocation_state = {
        "_sub_agent_interrupt_resume": {"responses": [], "snapshots": {"some-other-tool-use": snapshot}}
    }

    normal_result = AgentResult(
        stop_reason="end_turn",
        message={"role": "assistant", "content": [{"text": "ok"}]},
        metrics=EventLoopMetrics(),
        state={},
    )
    fake_agent.stream_async = MagicMock(return_value=_mock_stream_async(normal_result))

    tool = _AgentAsTool(fake_agent, name="fake_agent", description="desc", preserve_context=True)
    tool_use = {"toolUseId": "orch-1", "name": "fake_agent", "input": {"input": "the input"}}

    async for _ in tool.stream(tool_use, invocation_state):
        pass

    # Falls through to the normal path: the original tool input is sent as the prompt.
    assert fake_agent.stream_async.call_args[0][0] == "the input"


def _build_confirm_hook():
    """Hook that interrupts before get_my_roles and proceeds once a response is provided."""

    def before_tool(event):
        if event.tool_use["name"] == "get_my_roles":
            event.interrupt("confirm_roles", reason="Confirm reading roles?")

    return before_tool


@pytest.mark.asyncio
async def test_sub_agent_interrupt_resumes_deterministically_after_rehydration():
    """End-to-end: a sub-agent interrupt resumes across a simulated process restart.

    Reproduces the stateless-Lambda scenario where the orchestrator and sub-agent are rebuilt
    from storage between turns, so they no longer share in-memory Interrupt objects. Guarantees
    the sub-agent resumes the original pending tool call (same toolUseId) instead of re-prompting.
    """
    roles_calls = []

    @strands_tool(name="get_my_roles")
    def get_my_roles() -> str:
        roles_calls.append(True)
        return "admin, billing"

    # --- Turn 1: raise the sub-agent interrupt ---
    sub_model_1 = MockedModelProvider(
        [
            {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "sub-roles-1", "name": "get_my_roles", "input": {}}}],
            },
            {"role": "assistant", "content": [{"text": "Your roles are admin, billing."}]},
        ]
    )
    sub_1 = Agent(name="user_management", model=sub_model_1, tools=[get_my_roles], callback_handler=None)
    sub_1.hooks.add_callback(BeforeToolCallEvent, _build_confirm_hook())

    orch_model_1 = MockedModelProvider(
        [
            {
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": "orch-um-1", "name": "user_management", "input": {"input": "roles?"}}}
                ],
            },
            {"role": "assistant", "content": [{"text": "Done."}]},
        ]
    )
    orch_1 = Agent(
        name="orchestrator",
        model=orch_model_1,
        tools=[sub_1.as_tool(preserve_context=False)],
        callback_handler=None,
    )

    result_1 = orch_1("what are my roles")

    assert result_1.stop_reason == "interrupt"
    assert roles_calls == []  # tool blocked pending confirmation
    # The sub-agent snapshot is persisted inside the orchestrator's own interrupt record.
    snapshots = orch_1._interrupt_state.context["sub_agent_snapshots"]
    assert "orch-um-1" in snapshots
    session_data = snapshots["orch-um-1"]["session_snapshot"]["data"]
    persisted_tool_use = session_data["interrupt_state"]["context"]["tool_use_message"]["content"][0]["toolUse"]
    assert persisted_tool_use["toolUseId"] == "sub-roles-1"
    interrupt_id = next(iter(orch_1._interrupt_state.interrupts))

    # --- Simulate rehydration: serialize, then rebuild fresh objects from storage ---
    serialized_interrupt_state = json.loads(json.dumps(orch_1._interrupt_state.to_dict()))
    restored_orch_messages = json.loads(json.dumps(orch_1.messages))

    # Fresh sub-agent. Its model only needs the post-resume continuation, since resume skips
    # the model call that originally produced the tool use.
    sub_model_2 = MockedModelProvider([{"role": "assistant", "content": [{"text": "Your roles are admin, billing."}]}])
    sub_2 = Agent(name="user_management", model=sub_model_2, tools=[get_my_roles], callback_handler=None)
    sub_2.hooks.add_callback(BeforeToolCallEvent, _build_confirm_hook())

    orch_model_2 = MockedModelProvider([{"role": "assistant", "content": [{"text": "Done."}]}])
    orch_2 = Agent(
        name="orchestrator",
        model=orch_model_2,
        tools=[sub_2.as_tool(preserve_context=False)],
        callback_handler=None,
    )
    orch_2.messages = restored_orch_messages
    orch_2._interrupt_state = _InterruptState.from_dict(serialized_interrupt_state)

    # --- Turn 2: resume with the user's approval ---
    result_2 = orch_2([{"interruptResponse": {"interruptId": interrupt_id, "response": "APPROVE"}}])

    assert result_2.stop_reason == "end_turn"
    # The sub-agent ran the pending tool exactly once rather than re-prompting.
    assert roles_calls == [True]
    # And it resumed the original tool call: a result for the original toolUseId exists.
    sub_tool_result_ids = [
        content["toolResult"]["toolUseId"]
        for message in sub_2.messages
        for content in message["content"]
        if "toolResult" in content
    ]
    assert "sub-roles-1" in sub_tool_result_ids


# --- concurrency ---


@pytest.mark.asyncio
async def test_stream_rejects_concurrent_call(tool, mock_agent, tool_use, agent_result):
    """A second concurrent call should get an error ToolResultEvent."""
    mock_agent.stream_async.return_value = _mock_stream_async(agent_result)

    # Simulate the lock already being held by another invocation
    tool._lock.acquire()
    try:
        events = [event async for event in tool.stream(tool_use, {})]

        assert len(events) == 1
        assert isinstance(events[0], ToolResultEvent)
        assert events[0]["tool_result"]["status"] == "error"
        assert "already processing" in events[0]["tool_result"]["content"][0]["text"]
        mock_agent.stream_async.assert_not_called()
    finally:
        tool._lock.release()


@pytest.mark.asyncio
async def test_stream_releases_lock_after_completion(tool, mock_agent, tool_use, agent_result):
    """Lock should be released after stream completes, allowing subsequent calls."""
    mock_agent.stream_async.return_value = _mock_stream_async(agent_result)

    async for _ in tool.stream(tool_use, {}):
        pass

    assert not tool._lock.locked()

    # A second call should succeed
    mock_agent.stream_async.return_value = _mock_stream_async(agent_result)
    events = [event async for event in tool.stream(tool_use, {})]

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(result_events) == 1
    assert result_events[0]["tool_result"]["status"] == "success"


@pytest.mark.asyncio
async def test_stream_releases_lock_after_error(tool, mock_agent, tool_use):
    """Lock should be released even when the agent raises an exception."""
    mock_agent.stream_async.side_effect = RuntimeError("boom")

    async for _ in tool.stream(tool_use, {}):
        pass

    assert not tool._lock.locked()


# --- Agent-as-tool sugar (passing agents directly in tools list) ---


def test_agent_passed_directly_in_tools_list():
    """Test that an Agent can be passed directly in another Agent's tools list."""
    from strands.agent.agent import Agent

    sub_agent = Agent(name="research_agent", description="Does research", callback_handler=None)

    # This should work without calling .as_tool() explicitly
    parent_agent = Agent(name="orchestrator", tools=[sub_agent], callback_handler=None)

    assert "research_agent" in parent_agent.tool_names


def test_multiple_agents_passed_directly_in_tools_list():
    """Test that multiple Agents can be passed directly in another Agent's tools list."""
    from strands.agent.agent import Agent

    agent_a = Agent(name="agent_a", callback_handler=None)
    agent_b = Agent(name="agent_b", callback_handler=None)

    parent = Agent(name="parent", tools=[agent_a, agent_b], callback_handler=None)

    assert "agent_a" in parent.tool_names
    assert "agent_b" in parent.tool_names


def test_agent_mixed_with_regular_tools_in_tools_list():
    """Test that Agents can be mixed with regular tools in the tools list."""
    from strands import tool as tool_decorator
    from strands.agent.agent import Agent

    @tool_decorator
    def my_tool(x: str) -> str:
        """A regular tool."""
        return x

    sub_agent = Agent(name="helper_agent", callback_handler=None)

    parent = Agent(name="parent", tools=[my_tool, sub_agent], callback_handler=None)

    assert "my_tool" in parent.tool_names
    assert "helper_agent" in parent.tool_names


@pytest.mark.asyncio
async def test_concurrent_sub_agents_with_same_local_interrupt_id_are_disambiguated():
    """Two sub-agents producing the same local interrupt ID must not collide at the parent."""
    interrupt_a = Interrupt(id="same-local-id", name="confirm", reason="reason A")
    interrupt_b = Interrupt(id="same-local-id", name="confirm", reason="reason B")

    namespaced_a, id_map_a = _AgentAsTool._namespace_interrupts("outer-tool-A", [interrupt_a])
    namespaced_b, id_map_b = _AgentAsTool._namespace_interrupts("outer-tool-B", [interrupt_b])

    assert namespaced_a[0].id != namespaced_b[0].id
    assert namespaced_a[0].id == "outer-tool-A:same-local-id"
    assert namespaced_b[0].id == "outer-tool-B:same-local-id"

    parent_interrupts: dict[str, Interrupt] = {}
    parent_interrupts.setdefault(namespaced_a[0].id, namespaced_a[0])
    parent_interrupts.setdefault(namespaced_b[0].id, namespaced_b[0])
    assert len(parent_interrupts) == 2

    assert set(id_map_a.keys()) == {"outer-tool-A:same-local-id"}
    assert set(id_map_b.keys()) == {"outer-tool-B:same-local-id"}


def test_two_concurrent_nested_interrupts_through_event_loop():
    """Two sub-agents interrupt in a single orchestrator cycle and both resume correctly.

    This exercises the actual accumulation site in the event loop (sub_agent_snapshots dict
    update at event_loop.py) rather than calling _namespace_interrupts in isolation.
    A mutation that rebinds sub_agent_snapshots instead of updating it would fail this test
    because only the last snapshot would survive.
    """
    calls_a: list[str] = []
    calls_b: list[str] = []

    @strands_tool(name="sensitive_op_a")
    def sensitive_op_a() -> str:
        calls_a.append("executed")
        return "result A"

    @strands_tool(name="sensitive_op_b")
    def sensitive_op_b() -> str:
        calls_b.append("executed")
        return "result B"

    def _interrupt_hook(event):
        """Hook that interrupts before any sensitive_op_* tool."""
        tool_name = event.tool_use["name"]
        if tool_name.startswith("sensitive_op_"):
            event.interrupt("confirm", reason=f"Confirm {tool_name}?")

    # --- Turn 1: orchestrator calls both sub-agents, both interrupt ---
    sub_model_a = MockedModelProvider(
        [
            {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "sub-a-1", "name": "sensitive_op_a", "input": {}}}],
            },
            {"role": "assistant", "content": [{"text": "A done."}]},
        ]
    )
    sub_agent_a = Agent(name="agent_a", model=sub_model_a, tools=[sensitive_op_a], callback_handler=None)
    sub_agent_a.hooks.add_callback(BeforeToolCallEvent, _interrupt_hook)

    sub_model_b = MockedModelProvider(
        [
            {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "sub-b-1", "name": "sensitive_op_b", "input": {}}}],
            },
            {"role": "assistant", "content": [{"text": "B done."}]},
        ]
    )
    sub_agent_b = Agent(name="agent_b", model=sub_model_b, tools=[sensitive_op_b], callback_handler=None)
    sub_agent_b.hooks.add_callback(BeforeToolCallEvent, _interrupt_hook)

    # Orchestrator model produces a single message with two tool_use blocks (concurrent invocation).
    orch_model_1 = MockedModelProvider(
        [
            {
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": "orch-a", "name": "agent_a", "input": {"input": "do A"}}},
                    {"toolUse": {"toolUseId": "orch-b", "name": "agent_b", "input": {"input": "do B"}}},
                ],
            },
            {"role": "assistant", "content": [{"text": "All done."}]},
        ]
    )
    orch = Agent(
        name="orchestrator",
        model=orch_model_1,
        tools=[sub_agent_a.as_tool(preserve_context=False), sub_agent_b.as_tool(preserve_context=False)],
        callback_handler=None,
    )

    result_1 = orch("do both tasks")

    # Both sub-agents interrupted (neither tool actually ran).
    assert result_1.stop_reason == "interrupt"
    assert calls_a == []
    assert calls_b == []

    # The orchestrator accumulated both snapshots.
    sub_agent_snapshots = orch._interrupt_state.context.get("sub_agent_snapshots")
    assert sub_agent_snapshots is not None
    assert "orch-a" in sub_agent_snapshots
    assert "orch-b" in sub_agent_snapshots

    # Both interrupt IDs are distinct at the parent level.
    interrupt_ids = list(orch._interrupt_state.interrupts.keys())
    assert len(interrupt_ids) == 2
    assert interrupt_ids[0] != interrupt_ids[1]

    # --- Turn 2: resume with approval for both ---
    serialized_interrupt_state = json.loads(json.dumps(orch._interrupt_state.to_dict()))
    restored_messages = json.loads(json.dumps(orch.messages))

    # Fresh sub-agents for turn 2 (simulating stateless rebuild).
    sub_model_a2 = MockedModelProvider([{"role": "assistant", "content": [{"text": "A done."}]}])
    sub_agent_a2 = Agent(name="agent_a", model=sub_model_a2, tools=[sensitive_op_a], callback_handler=None)
    sub_agent_a2.hooks.add_callback(BeforeToolCallEvent, _interrupt_hook)

    sub_model_b2 = MockedModelProvider([{"role": "assistant", "content": [{"text": "B done."}]}])
    sub_agent_b2 = Agent(name="agent_b", model=sub_model_b2, tools=[sensitive_op_b], callback_handler=None)
    sub_agent_b2.hooks.add_callback(BeforeToolCallEvent, _interrupt_hook)

    orch_model_2 = MockedModelProvider([{"role": "assistant", "content": [{"text": "All done."}]}])
    orch_2 = Agent(
        name="orchestrator",
        model=orch_model_2,
        tools=[sub_agent_a2.as_tool(preserve_context=False), sub_agent_b2.as_tool(preserve_context=False)],
        callback_handler=None,
    )
    orch_2.messages = restored_messages
    orch_2._interrupt_state = _InterruptState.from_dict(serialized_interrupt_state)

    # Approve all interrupts.
    responses = [
        {"interruptResponse": {"interruptId": iid, "response": "APPROVE"}} for iid in interrupt_ids
    ]
    result_2 = orch_2(responses)

    assert result_2.stop_reason == "end_turn"
    # Both pending tools executed exactly once.
    assert calls_a == ["executed"]
    assert calls_b == ["executed"]

    # Both sub-agents resumed the correct original tool call IDs.
    sub_a_tool_result_ids = [
        content["toolResult"]["toolUseId"]
        for msg in sub_agent_a2.messages
        for content in msg.get("content", [])
        if "toolResult" in content
    ]
    sub_b_tool_result_ids = [
        content["toolResult"]["toolUseId"]
        for msg in sub_agent_b2.messages
        for content in msg.get("content", [])
        if "toolResult" in content
    ]
    assert "sub-a-1" in sub_a_tool_result_ids
    assert "sub-b-1" in sub_b_tool_result_ids


@pytest.mark.asyncio
async def test_snapshot_preserves_model_state_and_conversation_manager_state(fake_agent):
    """The sub-agent snapshot captures model_state and conversation_manager_state."""
    sub_tool_use_message = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "sub-1", "name": "do_thing", "input": {}}}],
    }
    interrupt = Interrupt(id="int-1", name="approval", reason="check")
    interrupt_result = AgentResult(
        stop_reason="interrupt",
        message={"role": "assistant", "content": [{"text": "pending"}]},
        metrics=EventLoopMetrics(),
        state={},
        interrupts=[interrupt],
    )

    async def fake_stream(prompt, **kwargs):
        fake_agent.messages = [{"role": "user", "content": [{"text": "go"}]}, sub_tool_use_message]
        fake_agent._model_state = {"response_id": "resp_abc123"}
        fake_agent._interrupt_state.interrupts["int-1"] = interrupt
        fake_agent._interrupt_state.activate()
        yield {"result": interrupt_result}

    fake_agent.stream_async = fake_stream

    tool = _AgentAsTool(fake_agent, name="fake_agent", description="desc", preserve_context=True)
    tool_use = {"toolUseId": "orch-1", "name": "fake_agent", "input": {"input": "go"}}

    events = [event async for event in tool.stream(tool_use, {})]

    interrupt_events = [e for e in events if isinstance(e, ToolInterruptEvent)]
    assert len(interrupt_events) == 1
    snapshot = interrupt_events[0].sub_agent_snapshot
    assert snapshot is not None

    session_data = snapshot["session_snapshot"]["data"]
    assert session_data["model_state"] == {"response_id": "resp_abc123"}
    assert "conversation_manager_state" in session_data
    assert "__name__" in session_data["conversation_manager_state"]


def test_tool_interrupt_event_snapshot_not_in_dict():
    """sub_agent_snapshot must not appear in the event's dict or as_dict() output."""
    interrupt = Interrupt(id="int-1", name="test", reason="why")
    tool_use = {"toolUseId": "tu-1", "name": "tool", "input": {}}
    snapshot_data = {"session_snapshot": {"data": {"messages": [{"role": "user", "content": [{"text": "secret"}]}]}}}

    event = ToolInterruptEvent(tool_use, [interrupt], sub_agent_snapshot=snapshot_data)

    assert event.sub_agent_snapshot is snapshot_data
    assert "sub_agent_snapshot" not in event
    assert "sub_agent_snapshot" not in event.get("tool_interrupt_event", {})
    assert "sub_agent_snapshot" not in str(event.as_dict())
    assert "secret" not in str(event.as_dict())

    event_no_snapshot = ToolInterruptEvent(tool_use, [interrupt])
    assert event_no_snapshot.sub_agent_snapshot is None
    assert "sub_agent_snapshot" not in event_no_snapshot.get("tool_interrupt_event", {})
