"""Tests for AgentDelegation plugin — verifies delegation semantics with agent-as-tool."""

import json as json_module
from datetime import datetime
from unittest.mock import MagicMock, PropertyMock

import pytest

from strands._middleware.stages import AgentStreamContext, ExecuteToolContext
from strands.agent._agent_as_tool import DELEGATION_DESCRIPTION_SUFFIX, _AgentAsTool
from strands.agent._agent_delegation import AgentDelegation, _DelegationState, _to_content_blocks
from strands.agent.agent import Agent
from strands.agent.agent_result import AgentResult
from strands.hooks import (
    AfterToolCallEvent,
    AfterToolsEvent,
    BeforeToolsEvent,
    MessageAddedEvent,
)
from strands.interrupt import _InterruptState
from strands.telemetry.metrics import EventLoopMetrics
from strands.types._events import EventLoopStopEvent, ToolResultEvent, TypedEvent
from tests.fixtures.mocked_model_provider import MockedModelProvider

# --- Helpers ---


def _mock_sub_agent(name="sub"):
    agent = MagicMock()
    agent.name = name
    agent._interrupt_state = _InterruptState()
    return agent


def _make_delegation_tool(name="sub", **kwargs):
    return _AgentAsTool(_mock_sub_agent(name), name=name, delegate=True, preserve_context=True, **kwargs)


def _get_plugin(agent):
    return agent._plugin_registry._plugins["strands:agent-delegation"]


def _fire_after_tool_call(parent, plugin, tool, tool_use_id, status, selected_tool=None):
    return AfterToolCallEvent(
        agent=parent,
        selected_tool=selected_tool or tool,
        tool_use={"toolUseId": tool_use_id, "name": "sub", "input": {}},
        invocation_state={"request_state": {}},
        result={"toolUseId": tool_use_id, "status": status, "content": [{"text": "x"}]},
    )


def _fire_after_tools(parent, plugin, status="success"):
    message = {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": "t1", "status": status, "content": [{"text": "x"}]}}],
    }
    event = AfterToolsEvent(agent=parent, message=message, invocation_state={"request_state": {}})
    plugin._on_after_tools(event)
    return event


# --- Delegation flag ---


def test_as_tool_delegate_defaults_false():
    tru_delegate = Agent(name="sub", callback_handler=None).as_tool().delegate
    assert tru_delegate is False


def test_as_tool_delegate_true_adds_suffix():
    tool = Agent(name="sub", description="Billing", callback_handler=None).as_tool(delegate=True)
    assert tool.delegate is True
    assert tool._description == "Billing" + DELEGATION_DESCRIPTION_SUFFIX


def test_as_tool_delegate_false_no_suffix():
    tool = Agent(name="sub", description="Billing", callback_handler=None).as_tool()
    assert DELEGATION_DESCRIPTION_SUFFIX not in tool._description


def test_as_tool_custom_description_with_delegate():
    tool = Agent(name="sub", callback_handler=None).as_tool(delegate=True, description="Custom")
    exp_description = "Custom" + DELEGATION_DESCRIPTION_SUFFIX
    assert tool._description == exp_description


# --- Tool does not own stop ---


@pytest.mark.asyncio
@pytest.mark.parametrize("delegate", [True, False])
async def test_tool_stream_never_sets_stop_event_loop(delegate):
    """_AgentAsTool defers all stop decisions to the plugin."""
    mock_agent = _mock_sub_agent()
    result = AgentResult(
        stop_reason="end_turn",
        message={"role": "assistant", "content": [{"text": "ok"}]},
        metrics=EventLoopMetrics(),
        state={},
    )

    async def mock_stream(prompt):
        yield {"result": result}

    mock_agent.stream_async = mock_stream
    tool = _AgentAsTool(mock_agent, name="sub", delegate=delegate, preserve_context=True)

    invocation_state = {"request_state": {}}
    async for _ in tool.stream({"toolUseId": "t1", "name": "sub", "input": {"input": "hi"}}, invocation_state):
        pass

    assert "stop_event_loop" not in invocation_state["request_state"]


@pytest.mark.asyncio
async def test_tool_stream_error_does_not_set_stop():
    mock_agent = _mock_sub_agent()

    async def failing(prompt):
        raise RuntimeError("crash")
        yield  # noqa: RET503

    mock_agent.stream_async = failing
    tool = _AgentAsTool(mock_agent, name="sub", delegate=True, preserve_context=True)

    invocation_state = {"request_state": {}}
    async for _ in tool.stream({"toolUseId": "t1", "name": "sub", "input": {"input": "hi"}}, invocation_state):
        pass

    assert invocation_state["request_state"].get("stop_event_loop") is not True


# --- Single-call constraint (BeforeToolsEvent) ---


@pytest.mark.parametrize(
    "tool_names,expect_cancel",
    [
        (["specialist", "other_tool"], True),
        (["billing", "tech"], True),
        (["specialist"], False),
        (["tool_a", "tool_b"], False),
    ],
    ids=["delegate+other", "two-delegates", "single-delegate", "no-delegates"],
)
def test_on_before_tools_cancel_decision(tool_names, expect_cancel):
    sub = Agent(name="specialist", description="S", callback_handler=None)
    tools = [sub.as_tool(delegate=True)]
    if "tech" in tool_names or "billing" in tool_names:
        sub2 = Agent(name="billing" if "billing" in tool_names else "tech", callback_handler=None)
        tools.append(sub2.as_tool(delegate=True))

    parent = Agent(name="parent", tools=tools, callback_handler=None)
    message = {
        "role": "assistant",
        "content": [
            {"toolUse": {"toolUseId": f"t{i}", "name": name, "input": {}}} for i, name in enumerate(tool_names)
        ],
    }

    plugin = AgentDelegation()
    event = BeforeToolsEvent(agent=parent, message=message, invocation_state={})
    plugin._on_before_tools(event)

    assert bool(event.cancel) == expect_cancel


# --- AfterToolCallEvent state tracking ---


def test_on_after_tool_call_marks_on_success():
    tool = _make_delegation_tool()
    parent = Agent(name="p", tools=[tool], callback_handler=None)
    plugin = _get_plugin(parent)
    plugin._state[parent] = _DelegationState(tool_use_count=1)

    event = _fire_after_tool_call(parent, plugin, tool, "t1", "success")
    plugin._on_after_tool_call(event)
    assert plugin._state[parent].tool_use_id == "t1"


def test_on_after_tool_call_clears_on_error():
    tool = _make_delegation_tool()
    parent = Agent(name="p", tools=[tool], callback_handler=None)
    plugin = _get_plugin(parent)
    plugin._state[parent] = _DelegationState(tool_use_id="t1", tool_use_count=1)

    event = _fire_after_tool_call(parent, plugin, tool, "t1", "error")
    plugin._on_after_tool_call(event)
    assert plugin._state[parent].tool_use_id is None


def test_on_after_tool_call_retry_swap_clears_matching_id():
    tool = _make_delegation_tool()
    parent = Agent(name="p", tools=[tool], callback_handler=None)
    plugin = _get_plugin(parent)
    plugin._state[parent] = _DelegationState(tool_use_id="t1", tool_use_count=1)

    ordinary = MagicMock()
    ordinary.delegate = False
    event = _fire_after_tool_call(parent, plugin, tool, "t1", "success", selected_tool=ordinary)
    plugin._on_after_tool_call(event)
    assert plugin._state[parent].tool_use_id is None


def test_on_after_tool_call_retry_swap_preserves_different_id():
    tool = _make_delegation_tool()
    parent = Agent(name="p", tools=[tool], callback_handler=None)
    plugin = _get_plugin(parent)
    plugin._state[parent] = _DelegationState(tool_use_id="t1", tool_use_count=1)

    ordinary = MagicMock()
    ordinary.delegate = False
    event = _fire_after_tool_call(parent, plugin, tool, "t2", "success", selected_tool=ordinary)
    plugin._on_after_tool_call(event)
    assert plugin._state[parent].tool_use_id == "t1"


def test_on_after_tool_call_foreign_stop_not_touched():
    """Delegation never clears stop_event_loop it didn't set."""
    tool = _make_delegation_tool()
    parent = Agent(name="p", tools=[tool], callback_handler=None)
    plugin = _get_plugin(parent)
    plugin._state[parent] = _DelegationState(tool_use_id="other", tool_use_count=1)

    invocation_state = {"request_state": {"stop_event_loop": True}}
    event = AfterToolCallEvent(
        agent=parent,
        selected_tool=tool,
        tool_use={"toolUseId": "t99", "name": "sub", "input": {}},
        invocation_state=invocation_state,
        result={"toolUseId": "t99", "status": "success", "content": [{"text": "ok"}]},
    )
    plugin._on_after_tool_call(event)
    assert invocation_state["request_state"]["stop_event_loop"] is True


# --- AfterToolsEvent end_turn ---


def test_on_after_tools_sets_end_turn_on_success():
    tool = _make_delegation_tool()
    parent = Agent(name="p", tools=[tool], callback_handler=None)
    plugin = _get_plugin(parent)
    plugin._state[parent] = _DelegationState(tool_use_id="t1", tool_use_count=1)

    tru_event = _fire_after_tools(parent, plugin)
    assert tru_event.end_turn is True


def test_on_after_tools_skips_when_result_is_error():
    tool = _make_delegation_tool()
    parent = Agent(name="p", tools=[tool], callback_handler=None)
    plugin = _get_plugin(parent)
    plugin._state[parent] = _DelegationState(tool_use_id="t1", tool_use_count=1)

    tru_event = _fire_after_tools(parent, plugin, status="error")
    assert not tru_event.end_turn
    assert plugin._state[parent].tool_use_id is None


def test_on_after_tools_suppressed_with_structured_output():
    from pydantic import BaseModel

    from strands.tools.structured_output.structured_output_tool import StructuredOutputTool

    class R(BaseModel):
        x: int

    tool = _make_delegation_tool()
    parent = Agent(name="p", tools=[tool], callback_handler=None)
    plugin = _get_plugin(parent)
    plugin._state[parent] = _DelegationState(tool_use_id="t1", tool_use_count=1)
    parent.tool_registry.register_dynamic_tool(StructuredOutputTool(R))

    tru_event = _fire_after_tools(parent, plugin)
    assert not tru_event.end_turn


@pytest.mark.parametrize(
    "content",
    [
        [{"toolResult": {"toolUseId": "t_other", "status": "success", "content": [{"text": "x"}]}}],
        [],
    ],
    ids=["mismatched-id", "empty-content"],
)
def test_on_after_tools_skips_when_result_absent(content):
    """When the message has no toolResult matching state.tool_use_id, end_turn is not set."""
    tool = _make_delegation_tool()
    parent = Agent(name="p", tools=[tool], callback_handler=None)
    plugin = _get_plugin(parent)
    plugin._state[parent] = _DelegationState(tool_use_id="t1", tool_use_count=1)

    message = {"role": "user", "content": content}
    event = AfterToolsEvent(agent=parent, message=message, invocation_state={"request_state": {}})
    plugin._on_after_tools(event)

    assert not event.end_turn
    assert plugin._state[parent].tool_use_id is None


# --- _handle_stream ordering ---


@pytest.mark.asyncio
async def test_handle_stream_non_delegation_preserves_trailing():
    parent = Agent(name="p", callback_handler=None)
    plugin = _get_plugin(parent)
    ctx = AgentStreamContext(agent=parent, messages=[], invocation_state={"request_state": {}}, _interrupts={})

    a = TypedEvent({"a": 1})
    stop = EventLoopStopEvent("end_turn", {"role": "assistant", "content": []}, parent.event_loop_metrics, {})
    b = TypedEvent({"b": 2})

    async def inner(c):
        yield a
        yield stop
        yield b

    tru_events = [e async for e in plugin._handle_stream(ctx, inner)]
    exp_events = [a, stop, b]
    assert tru_events == exp_events


@pytest.mark.asyncio
async def test_handle_stream_non_delegation_multiple_stops_preserved():
    parent = Agent(name="p", callback_handler=None)
    plugin = _get_plugin(parent)
    ctx = AgentStreamContext(agent=parent, messages=[], invocation_state={"request_state": {}}, _interrupts={})

    s1 = EventLoopStopEvent("end_turn", {"role": "assistant", "content": []}, parent.event_loop_metrics, {})
    mid = TypedEvent({"mid": 1})
    s2 = EventLoopStopEvent("end_turn", {"role": "assistant", "content": []}, parent.event_loop_metrics, {})

    async def inner(c):
        yield s1
        yield mid
        yield s2

    tru_events = [e async for e in plugin._handle_stream(ctx, inner)]
    exp_events = [s1, mid, s2]
    assert tru_events == exp_events


@pytest.mark.asyncio
async def test_handle_stream_delegation_replaces_stop_with_trailing():
    """Delegation replaces stop; trailing events keep position; exactly one terminal."""
    tool = _make_delegation_tool()
    parent = Agent(name="p", tools=[tool], callback_handler=None)
    plugin = _get_plugin(parent)

    parent.messages.extend(
        [
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "sub", "input": {}}}]},
            {
                "role": "user",
                "content": [{"toolResult": {"toolUseId": "t1", "status": "success", "content": [{"text": "answer"}]}}],
            },
            {"role": "assistant", "content": [{"text": "placeholder"}]},
        ]
    )

    ctx = AgentStreamContext(agent=parent, messages=[], invocation_state={"request_state": {}}, _interrupts={})
    pre = TypedEvent({"pre": 1})
    stop = EventLoopStopEvent("end_turn", parent.messages[-1], parent.event_loop_metrics, {})
    trail = TypedEvent({"trail": 1})

    async def inner(c):
        plugin._state[parent] = _DelegationState(tool_use_id="t1", end_turn_via_delegation=True, tool_use_count=1)
        yield pre
        yield stop
        yield trail

    tru_events = [e async for e in plugin._handle_stream(ctx, inner)]

    assert tru_events[0] is pre
    tru_stops = [e for e in tru_events if isinstance(e, EventLoopStopEvent)]
    assert len(tru_stops) == 1
    assert tru_stops[0]["stop"][1]["content"][0]["text"] == "answer"
    assert tru_events[-1] is trail
    assert "tracking_id" in parent.messages[-1]


@pytest.mark.asyncio
async def test_handle_stream_reverify_failure_replays_original():
    """If the tool result is mutated to error after _on_after_tools, original stop replays."""
    tool = _make_delegation_tool()
    parent = Agent(name="p", tools=[tool], callback_handler=None)
    plugin = _get_plugin(parent)

    parent.messages.extend(
        [
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "sub", "input": {}}}]},
            {
                "role": "user",
                "content": [{"toolResult": {"toolUseId": "t1", "status": "error", "content": [{"text": "failed"}]}}],
            },
            {"role": "assistant", "content": [{"text": "placeholder"}]},
        ]
    )

    ctx = AgentStreamContext(agent=parent, messages=[], invocation_state={"request_state": {}}, _interrupts={})
    original_stop = EventLoopStopEvent("end_turn", parent.messages[-1], parent.event_loop_metrics, {})

    async def inner(c):
        plugin._state[parent] = _DelegationState(tool_use_id="t1", end_turn_via_delegation=True, tool_use_count=1)
        yield original_stop

    tru_events = [e async for e in plugin._handle_stream(ctx, inner)]
    assert len(tru_events) == 1
    assert tru_events[0] is original_stop


@pytest.mark.asyncio
async def test_handle_stream_absent_tool_result_skips_delegation():
    """When the tool result for the delegation toolUseId is absent from history, delegation is skipped."""
    tool = _make_delegation_tool()
    parent = Agent(name="p", tools=[tool], callback_handler=None)
    plugin = _get_plugin(parent)

    parent.messages.extend(
        [
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "sub", "input": {}}}]},
            {"role": "assistant", "content": [{"text": "placeholder"}]},
        ]
    )

    ctx = AgentStreamContext(agent=parent, messages=[], invocation_state={"request_state": {}}, _interrupts={})
    original_stop = EventLoopStopEvent("end_turn", parent.messages[-1], parent.event_loop_metrics, {})

    async def inner(c):
        plugin._state[parent] = _DelegationState(tool_use_id="t1", end_turn_via_delegation=True, tool_use_count=1)
        yield original_stop

    tru_events = [e async for e in plugin._handle_stream(ctx, inner)]
    assert len(tru_events) == 1
    assert tru_events[0] is original_stop


# --- Stateful model rejection ---


def test_init_raises_with_delegation_on_stateful():
    tool = _make_delegation_tool()
    stateful = MagicMock()
    type(stateful).stateful = PropertyMock(return_value=True)

    with pytest.raises(ValueError, match="not supported with stateful models"):
        Agent(name="p", model=stateful, tools=[tool], callback_handler=None)


def test_init_ok_without_delegation_on_stateful():
    stateful = MagicMock()
    type(stateful).stateful = PropertyMock(return_value=True)
    Agent(name="p", model=stateful, callback_handler=None)


@pytest.mark.asyncio
async def test_runtime_stateful_delegate_returns_error():
    tool = _make_delegation_tool()
    parent = Agent(name="p", callback_handler=None)
    plugin = _get_plugin(parent)
    type(parent.model).stateful = PropertyMock(return_value=True)

    try:
        ctx = ExecuteToolContext(
            agent=parent,
            tool=tool,
            tool_use={"toolUseId": "t1", "name": "sub", "input": {"input": "hi"}},
            invocation_state={"request_state": {}},
            _interrupt_state=parent._interrupt_state,
        )

        async def unreachable(c):
            yield ToolResultEvent({"toolUseId": "t1", "status": "success", "content": [{"text": "nope"}]})

        tru_events = [e async for e in plugin._handle_tool_execution(ctx, unreachable)]
        assert len(tru_events) == 1
        assert tru_events[0].tool_result["status"] == "error"
        assert "stateful" in tru_events[0].tool_result["content"][0]["text"].lower()
    finally:
        del type(parent.model).stateful


# --- Auto-registration ---


def test_auto_registered():
    assert "strands:agent-delegation" in Agent(name="t", callback_handler=None)._plugin_registry._plugins


def test_auto_registration_not_duplicated():
    p = AgentDelegation()
    agent = Agent(name="t", plugins=[p], callback_handler=None)
    assert agent._plugin_registry._plugins.get("strands:agent-delegation") is p


# --- Content conversion ---


def test_to_content_blocks_converts_text_json_and_passthrough():
    tru_blocks = _to_content_blocks(
        {
            "content": [
                {"text": "hi"},
                {"json": {"k": 1}},
                {"image": {"format": "png", "source": {"bytes": b"x"}}},
            ]
        }
    )
    assert tru_blocks[0] == {"text": "hi"}
    assert json_module.loads(tru_blocks[1]["text"]) == {"k": 1}
    assert tru_blocks[2] == {"image": {"format": "png", "source": {"bytes": b"x"}}}


# --- Child structured output serialization ---


@pytest.mark.asyncio
async def test_child_structured_output_datetime_serializes_cleanly():
    from pydantic import BaseModel

    class S(BaseModel):
        at: datetime

    mock_agent = _mock_sub_agent("sched")
    result = AgentResult(
        stop_reason="end_turn",
        message={"role": "assistant", "content": [{"text": "done"}]},
        metrics=EventLoopMetrics(),
        state={},
        structured_output=S(at=datetime(2025, 1, 15, 9, 0)),
    )

    async def stream(prompt):
        yield {"result": result}

    mock_agent.stream_async = stream
    tool = _AgentAsTool(mock_agent, name="sched", delegate=True, preserve_context=True)

    tru_events = [e async for e in tool.stream({"toolUseId": "t1", "name": "sched", "input": {"input": "x"}}, {})]
    tru_results = [e for e in tru_events if isinstance(e, ToolResultEvent)]
    assert tru_results[0].tool_result["status"] == "success"
    assert "2025-01-15" in json_module.dumps(tru_results[0].tool_result["content"][0]["json"])


# --- Full delegation flow ---


@pytest.mark.asyncio
async def test_full_delegation_routes_to_specialist():
    sub = Agent(
        model=MockedModelProvider([{"role": "assistant", "content": [{"text": "Balance: $42"}]}]),
        name="billing",
        callback_handler=None,
    )
    orch = Agent(
        model=MockedModelProvider(
            [
                {
                    "role": "assistant",
                    "content": [{"toolUse": {"toolUseId": "c1", "name": "billing", "input": {"input": "check"}}}],
                }
            ]
        ),
        name="orch",
        tools=[sub.as_tool(delegate=True)],
        callback_handler=None,
    )

    tru_result = await orch.invoke_async("Check balance")
    assert tru_result.stop_reason == "end_turn"
    assert any("$42" in str(b.get("text", "")) for b in tru_result.message["content"])

    # The placeholder must not remain stranded in history
    for msg in orch.messages:
        for block in msg.get("content", []):
            if isinstance(block, dict) and "text" in block:
                assert "Turn ended early" not in block["text"]


@pytest.mark.asyncio
async def test_full_delegation_error_recovery():
    err = _mock_sub_agent("failing")

    async def fail(prompt):
        raise RuntimeError("crash")
        yield  # noqa: RET503

    err.stream_async = fail
    tool = _AgentAsTool(err, name="failing", delegate=True, preserve_context=True)

    orch = Agent(
        model=MockedModelProvider(
            [
                {
                    "role": "assistant",
                    "content": [{"toolUse": {"toolUseId": "c1", "name": "failing", "input": {"input": "x"}}}],
                },
                {"role": "assistant", "content": [{"text": "I recovered."}]},
            ]
        ),
        name="orch",
        tools=[tool],
        callback_handler=None,
    )

    tru_result = await orch.invoke_async("Do it")
    assert tru_result.stop_reason == "end_turn"
    assert "recovered" in str(tru_result.message["content"]).lower()


# --- Session persistence ---


@pytest.mark.asyncio
async def test_persisted_matches_in_memory_after_delegation():
    from strands.session.repository_session_manager import RepositorySessionManager
    from tests.fixtures.mock_session_repository import MockedSessionRepository

    repo = MockedSessionRepository()
    session_mgr = RepositorySessionManager(session_id="s1", session_repository=repo)

    sub_model = MockedModelProvider([{"role": "assistant", "content": [{"text": "Balance: $42"}]}])
    sub = Agent(model=sub_model, name="billing", callback_handler=None)

    orch_model = MockedModelProvider(
        [
            {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "c1", "name": "billing", "input": {"input": "check"}}}],
            }
        ]
    )
    orch = Agent(
        model=orch_model,
        name="orchestrator",
        tools=[sub.as_tool(delegate=True)],
        session_manager=session_mgr,
        callback_handler=None,
    )

    await orch.invoke_async("Check balance")

    session_mgr_2 = RepositorySessionManager(session_id="s1", session_repository=repo)
    orch_2 = Agent(model=orch_model, name="orchestrator", session_manager=session_mgr_2, callback_handler=None)

    assert len(orch_2.messages) == len(orch.messages)
    for restored, live in zip(orch_2.messages, orch.messages, strict=True):
        assert restored["role"] == live["role"]
        assert restored["content"] == live["content"]

    assert orch_2.messages[-1]["role"] == "assistant"
    assert any("$42" in str(b.get("text", "")) for b in orch_2.messages[-1]["content"])


# --- Context offloader integration ---


@pytest.mark.asyncio
async def test_large_delegation_result_not_offloaded():
    """A delegation result exceeding the offloader threshold stays in context."""
    from strands.vended_plugins.context_offloader import ContextOffloader, InMemoryStorage

    storage = InMemoryStorage()
    offloader = ContextOffloader(storage=storage, max_result_tokens=25, preview_tokens=10)

    large_answer = "x" * 500
    sub = Agent(
        model=MockedModelProvider([{"role": "assistant", "content": [{"text": large_answer}]}]),
        name="billing",
        callback_handler=None,
    )

    orch = Agent(
        model=MockedModelProvider(
            [
                {
                    "role": "assistant",
                    "content": [{"toolUse": {"toolUseId": "c1", "name": "billing", "input": {"input": "check"}}}],
                }
            ]
        ),
        name="orch",
        tools=[sub.as_tool(delegate=True)],
        plugins=[offloader],
        callback_handler=None,
    )

    tru_result = await orch.invoke_async("Check balance")
    assert tru_result.stop_reason == "end_turn"
    tru_text = "".join(block.get("text", "") for block in tru_result.message["content"] if isinstance(block, dict))
    assert large_answer in tru_text
    assert len(storage._store) == 0


# --- AfterToolsEvent ordering guard ---


@pytest.mark.asyncio
async def test_ordering_guard_late_hook_flipping_result_prevents_end_turn():
    """A default-order AfterToolsEvent hook that flips the result to error prevents delegation end_turn.

    Order priority always wins: DEFAULT (0) runs before SDK_LAST (100). Reverse ordering only flips
    registration order *within* one order tier, so delegation's SDK_LAST hook runs last and reads the
    already-mutated committed message. Its own check in _on_after_tools is what declines to set
    end_turn here; _handle_stream's re-verification is a second line of defence that does not fire in
    this scenario.
    """
    sub = Agent(
        model=MockedModelProvider([{"role": "assistant", "content": [{"text": "answer"}]}]),
        name="specialist",
        callback_handler=None,
    )

    orch = Agent(
        model=MockedModelProvider(
            [
                {
                    "role": "assistant",
                    "content": [{"toolUse": {"toolUseId": "c1", "name": "specialist", "input": {"input": "go"}}}],
                },
                {"role": "assistant", "content": [{"text": "I continued after flip."}]},
            ]
        ),
        name="orch",
        tools=[sub.as_tool(delegate=True)],
        callback_handler=None,
    )

    def flip_result_to_error(event: AfterToolsEvent):
        content = event.message.get("content", [])
        for block in content:
            if isinstance(block, dict) and "toolResult" in block:
                block["toolResult"]["status"] = "error"

    orch.add_hook(flip_result_to_error, AfterToolsEvent)

    tru_result = await orch.invoke_async("Do it")
    assert tru_result.stop_reason == "end_turn"
    assert "continued" in str(tru_result.message["content"]).lower()


# --- MessageAddedEvent during delegation ---


@pytest.mark.asyncio
async def test_message_added_event_fires_for_delegation_message():
    """MessageAddedEvent fires for the placeholder and then the real delegation content."""
    sub = Agent(
        model=MockedModelProvider([{"role": "assistant", "content": [{"text": "Balance: $42"}]}]),
        name="billing",
        callback_handler=None,
    )

    orch = Agent(
        model=MockedModelProvider(
            [
                {
                    "role": "assistant",
                    "content": [{"toolUse": {"toolUseId": "c1", "name": "billing", "input": {"input": "check"}}}],
                }
            ]
        ),
        name="orch",
        tools=[sub.as_tool(delegate=True)],
        callback_handler=None,
    )

    received_messages = []

    def on_message_added(event: MessageAddedEvent):
        received_messages.append(event.message)

    orch.add_hook(on_message_added, MessageAddedEvent)
    await orch.invoke_async("Check balance")

    # Expected: user prompt, assistant tool_use, tool_result, end_turn placeholder, delegation content
    assert len(received_messages) == 5

    tru_placeholders = [
        m
        for m in received_messages
        if m["role"] == "assistant" and any("Turn ended early" in str(b.get("text", "")) for b in m.get("content", []))
    ]
    assert len(tru_placeholders) == 1

    tru_delegation = [
        m
        for m in received_messages
        if m["role"] == "assistant" and any("$42" in str(b.get("text", "")) for b in m.get("content", []))
    ]
    assert len(tru_delegation) == 1

    # Placeholder comes before delegation content
    assert received_messages.index(tru_placeholders[0]) < received_messages.index(tru_delegation[0])


@pytest.mark.asyncio
async def test_session_manager_suppresses_delegation_message_added_event():
    """With a session manager, subscribers see the placeholder but not the delegation content event."""
    from strands.session.repository_session_manager import RepositorySessionManager
    from tests.fixtures.mock_session_repository import MockedSessionRepository

    repo = MockedSessionRepository()
    session_mgr = RepositorySessionManager(session_id="s1", session_repository=repo)

    sub = Agent(
        model=MockedModelProvider([{"role": "assistant", "content": [{"text": "Balance: $42"}]}]),
        name="billing",
        callback_handler=None,
    )

    orch = Agent(
        model=MockedModelProvider(
            [
                {
                    "role": "assistant",
                    "content": [{"toolUse": {"toolUseId": "c1", "name": "billing", "input": {"input": "check"}}}],
                }
            ]
        ),
        name="orch",
        tools=[sub.as_tool(delegate=True)],
        session_manager=session_mgr,
        callback_handler=None,
    )

    received_messages = []

    def on_message_added(event: MessageAddedEvent):
        received_messages.append(event.message)

    orch.add_hook(on_message_added, MessageAddedEvent)
    await orch.invoke_async("Check balance")

    tru_delegation = [
        m
        for m in received_messages
        if m["role"] == "assistant" and any("$42" in str(b.get("text", "")) for b in m.get("content", []))
    ]
    assert len(tru_delegation) == 0

    # But agent.messages still reflects the delegation content
    assert any("$42" in str(b.get("text", "")) for b in orch.messages[-1].get("content", []))


# --- Middleware single-call guard ---


@pytest.mark.asyncio
async def test_middleware_rejects_delegation_when_batch_count_exceeds_one():
    tool = _make_delegation_tool()
    parent = Agent(name="p", tools=[tool], callback_handler=None)
    plugin = _get_plugin(parent)
    plugin._state[parent] = _DelegationState(tool_use_count=2)

    ctx = ExecuteToolContext(
        agent=parent,
        tool=tool,
        tool_use={"toolUseId": "t1", "name": "sub", "input": {"input": "hi"}},
        invocation_state={"request_state": {}},
        _interrupt_state=parent._interrupt_state,
    )

    async def unreachable(c):
        yield ToolResultEvent({"toolUseId": "t1", "status": "success", "content": [{"text": "nope"}]})

    tru_events = [e async for e in plugin._handle_tool_execution(ctx, unreachable)]
    assert len(tru_events) == 1
    assert tru_events[0].tool_result["status"] == "error"
    assert "only tool" in tru_events[0].tool_result["content"][0]["text"].lower()


# --- Byte-verbatim fidelity ---


@pytest.mark.asyncio
async def test_delegation_preserves_content_verbatim_across_hops():
    """Delegated content must be byte-identical across a multi-level chain."""
    exact_payload = '{"name": "User", "roles": ["admin"]}'

    leaf = Agent(
        model=MockedModelProvider([{"role": "assistant", "content": [{"text": exact_payload}]}]),
        name="leaf",
        callback_handler=None,
    )
    mid = Agent(
        model=MockedModelProvider(
            [
                {
                    "role": "assistant",
                    "content": [{"toolUse": {"toolUseId": "m1", "name": "leaf", "input": {"input": "go"}}}],
                }
            ]
        ),
        name="mid",
        tools=[leaf.as_tool(delegate=True)],
        callback_handler=None,
    )
    top = Agent(
        model=MockedModelProvider(
            [
                {
                    "role": "assistant",
                    "content": [{"toolUse": {"toolUseId": "t1", "name": "mid", "input": {"input": "go"}}}],
                }
            ]
        ),
        name="top",
        tools=[mid.as_tool(delegate=True)],
        callback_handler=None,
    )

    result = await top.invoke_async("run")

    assert len(result.message["content"]) == 1
    delivered_text = result.message["content"][0]["text"]
    assert delivered_text == exact_payload, f"Expected verbatim {exact_payload!r}, got {delivered_text!r}"
