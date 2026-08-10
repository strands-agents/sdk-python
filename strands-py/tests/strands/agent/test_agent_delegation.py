"""Tests for AgentDelegation plugin — verifies delegation semantics with agent-as-tool."""

from unittest.mock import MagicMock, PropertyMock

import pytest

from strands.agent._agent_as_tool import DELEGATION_DESCRIPTION_SUFFIX, _AgentAsTool
from strands.agent._agent_delegation import AgentDelegation, _DelegationState, _to_content_blocks
from strands.agent.agent import Agent
from strands.agent.agent_result import AgentResult
from strands.interrupt import _InterruptState
from strands.telemetry.metrics import EventLoopMetrics


def _mock_sub_agent(name="sub"):
    agent = MagicMock()
    agent.name = name
    agent._interrupt_state = _InterruptState()
    return agent


def _make_delegation_tool(name="sub", **kwargs):
    return _AgentAsTool(_mock_sub_agent(name), name=name, delegate=True, preserve_context=True, **kwargs)


def _get_plugin(agent):
    return agent._plugin_registry._plugins["strands:agent-delegation"]


class TestDelegationFlag:
    def test_delegate_defaults_false(self):
        tool = Agent(name="sub", callback_handler=None).as_tool()
        assert tool.delegate is False

    def test_delegate_true_adds_suffix(self):
        tool = Agent(name="sub", description="Billing", callback_handler=None).as_tool(delegate=True)
        assert tool.delegate is True
        assert tool._description == "Billing" + DELEGATION_DESCRIPTION_SUFFIX

    def test_delegate_false_no_suffix(self):
        tool = Agent(name="sub", description="Billing", callback_handler=None).as_tool()
        assert DELEGATION_DESCRIPTION_SUFFIX not in tool._description

    def test_custom_description_with_delegate(self):
        tool = Agent(name="sub", callback_handler=None).as_tool(delegate=True, description="Custom")
        assert tool._description == "Custom" + DELEGATION_DESCRIPTION_SUFFIX


class TestToolDoesNotOwnStop:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("delegate", [True, False])
    async def test_tool_never_sets_stop_event_loop(self, delegate):
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
    async def test_tool_error_does_not_set_stop(self):
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


class TestSingleCallConstraint:
    """BeforeToolsEvent cancels the batch when a delegate is mixed with other tools."""

    @pytest.mark.parametrize(
        "tool_names,expect_cancel",
        [
            (["specialist", "other_tool"], True),
            (["billing", "tech"], True),  # two delegates
            (["specialist"], False),  # single delegate OK
            (["tool_a", "tool_b"], False),  # no delegates
        ],
        ids=["delegate+other", "two-delegates", "single-delegate", "no-delegates"],
    )
    def test_cancel_decision(self, tool_names, expect_cancel):
        from strands.hooks import BeforeToolsEvent

        sub = Agent(name="specialist", description="S", callback_handler=None)
        tools = [sub.as_tool(delegate=True)]
        # Add a second delegate if needed
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


class TestAfterToolCallState:
    """_on_after_tool_call marks/clears tool_use_id correctly."""

    def _fire(self, parent, plugin, tool, tool_use_id, status, selected_tool=None):
        from strands.hooks import AfterToolCallEvent

        return AfterToolCallEvent(
            agent=parent,
            selected_tool=selected_tool or tool,
            tool_use={"toolUseId": tool_use_id, "name": "sub", "input": {}},
            invocation_state={"request_state": {}},
            result={"toolUseId": tool_use_id, "status": status, "content": [{"text": "x"}]},
        )

    def test_marks_on_success(self):
        tool = _make_delegation_tool()
        parent = Agent(name="p", tools=[tool], callback_handler=None)
        plugin = _get_plugin(parent)
        plugin._state[parent] = _DelegationState(tool_use_count=1)

        event = self._fire(parent, plugin, tool, "t1", "success")
        plugin._on_after_tool_call(event)
        assert plugin._state[parent].tool_use_id == "t1"

    def test_clears_on_error(self):
        tool = _make_delegation_tool()
        parent = Agent(name="p", tools=[tool], callback_handler=None)
        plugin = _get_plugin(parent)
        plugin._state[parent] = _DelegationState(tool_use_id="t1", tool_use_count=1)

        event = self._fire(parent, plugin, tool, "t1", "error")
        plugin._on_after_tool_call(event)
        assert plugin._state[parent].tool_use_id is None

    def test_retry_swap_clears_matching_id(self):
        tool = _make_delegation_tool()
        parent = Agent(name="p", tools=[tool], callback_handler=None)
        plugin = _get_plugin(parent)
        plugin._state[parent] = _DelegationState(tool_use_id="t1", tool_use_count=1)

        ordinary = MagicMock()
        ordinary.delegate = False
        event = self._fire(parent, plugin, tool, "t1", "success", selected_tool=ordinary)
        plugin._on_after_tool_call(event)
        assert plugin._state[parent].tool_use_id is None

    def test_retry_swap_preserves_different_id(self):
        tool = _make_delegation_tool()
        parent = Agent(name="p", tools=[tool], callback_handler=None)
        plugin = _get_plugin(parent)
        plugin._state[parent] = _DelegationState(tool_use_id="t1", tool_use_count=1)

        ordinary = MagicMock()
        ordinary.delegate = False
        event = self._fire(parent, plugin, tool, "t2", "success", selected_tool=ordinary)
        plugin._on_after_tool_call(event)
        assert plugin._state[parent].tool_use_id == "t1"

    def test_foreign_stop_not_touched(self):
        """Delegation never clears stop_event_loop it didn't set."""
        tool = _make_delegation_tool()
        parent = Agent(name="p", tools=[tool], callback_handler=None)
        plugin = _get_plugin(parent)
        plugin._state[parent] = _DelegationState(tool_use_id="other", tool_use_count=1)

        invocation_state = {"request_state": {"stop_event_loop": True}}
        from strands.hooks import AfterToolCallEvent

        event = AfterToolCallEvent(
            agent=parent,
            selected_tool=tool,
            tool_use={"toolUseId": "t99", "name": "sub", "input": {}},
            invocation_state=invocation_state,
            result={"toolUseId": "t99", "status": "success", "content": [{"text": "ok"}]},
        )
        plugin._on_after_tool_call(event)
        assert invocation_state["request_state"]["stop_event_loop"] is True


class TestAfterToolsEndTurn:
    """_on_after_tools sets end_turn only when delegation result is valid."""

    def _fire(self, parent, plugin, status="success"):
        from strands.hooks import AfterToolsEvent

        message = {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "t1", "status": status, "content": [{"text": "x"}]}}],
        }
        event = AfterToolsEvent(agent=parent, message=message, invocation_state={"request_state": {}})
        plugin._on_after_tools(event)
        return event

    def test_sets_end_turn_on_success(self):
        tool = _make_delegation_tool()
        parent = Agent(name="p", tools=[tool], callback_handler=None)
        plugin = _get_plugin(parent)
        plugin._state[parent] = _DelegationState(tool_use_id="t1", tool_use_count=1)

        event = self._fire(parent, plugin)
        assert event.end_turn is True

    def test_skips_when_result_is_error(self):
        tool = _make_delegation_tool()
        parent = Agent(name="p", tools=[tool], callback_handler=None)
        plugin = _get_plugin(parent)
        plugin._state[parent] = _DelegationState(tool_use_id="t1", tool_use_count=1)

        event = self._fire(parent, plugin, status="error")
        assert not event.end_turn
        assert plugin._state[parent].tool_use_id is None

    def test_suppressed_with_structured_output(self):
        from pydantic import BaseModel

        from strands.tools.structured_output.structured_output_tool import StructuredOutputTool

        class R(BaseModel):
            x: int

        tool = _make_delegation_tool()
        parent = Agent(name="p", tools=[tool], callback_handler=None)
        plugin = _get_plugin(parent)
        plugin._state[parent] = _DelegationState(tool_use_id="t1", tool_use_count=1)
        parent.tool_registry.register_dynamic_tool(StructuredOutputTool(R))

        event = self._fire(parent, plugin)
        assert not event.end_turn


class TestStreamOrdering:
    """_handle_stream preserves ordering and emits exactly one terminal."""

    @pytest.mark.asyncio
    async def test_non_delegation_preserves_trailing_after_stop(self):
        from strands._middleware.stages import AgentStreamContext
        from strands.types._events import EventLoopStopEvent, TypedEvent

        parent = Agent(name="p", callback_handler=None)
        plugin = _get_plugin(parent)
        ctx = AgentStreamContext(
            agent=parent,
            messages=[],
            invocation_state={"request_state": {}},
            _interrupts={},
        )

        a = TypedEvent({"a": 1})
        stop = EventLoopStopEvent("end_turn", {"role": "assistant", "content": []}, parent.event_loop_metrics, {})
        b = TypedEvent({"b": 2})

        async def inner(c):
            yield a
            yield stop
            yield b

        events = [e async for e in plugin._handle_stream(ctx, inner)]
        assert events == [a, stop, b]

    @pytest.mark.asyncio
    async def test_non_delegation_multiple_stops_preserved(self):
        from strands._middleware.stages import AgentStreamContext
        from strands.types._events import EventLoopStopEvent, TypedEvent

        parent = Agent(name="p", callback_handler=None)
        plugin = _get_plugin(parent)
        ctx = AgentStreamContext(
            agent=parent,
            messages=[],
            invocation_state={"request_state": {}},
            _interrupts={},
        )

        s1 = EventLoopStopEvent("end_turn", {"role": "assistant", "content": []}, parent.event_loop_metrics, {})
        mid = TypedEvent({"mid": 1})
        s2 = EventLoopStopEvent("end_turn", {"role": "assistant", "content": []}, parent.event_loop_metrics, {})

        async def inner(c):
            yield s1
            yield mid
            yield s2

        events = [e async for e in plugin._handle_stream(ctx, inner)]
        assert events == [s1, mid, s2]

    @pytest.mark.asyncio
    async def test_delegation_single_terminal_with_trailing(self):
        """Delegation replaces stop; trailing events keep position; exactly one terminal."""
        from strands._middleware.stages import AgentStreamContext
        from strands.types._events import EventLoopStopEvent, TypedEvent

        tool = _make_delegation_tool()
        parent = Agent(name="p", tools=[tool], callback_handler=None)
        plugin = _get_plugin(parent)

        parent.messages.extend(
            [
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "sub", "input": {}}}]},
                {
                    "role": "user",
                    "content": [
                        {"toolResult": {"toolUseId": "t1", "status": "success", "content": [{"text": "answer"}]}}
                    ],
                },
                {"role": "assistant", "content": [{"text": "placeholder"}]},
            ]
        )

        ctx = AgentStreamContext(
            agent=parent,
            messages=[],
            invocation_state={"request_state": {}},
            _interrupts={},
        )

        pre = TypedEvent({"pre": 1})
        stop = EventLoopStopEvent("end_turn", parent.messages[-1], parent.event_loop_metrics, {})
        trail = TypedEvent({"trail": 1})

        async def inner(c):
            plugin._state[parent] = _DelegationState(tool_use_id="t1", end_turn_via_delegation=True, tool_use_count=1)
            yield pre
            yield stop
            yield trail

        events = [e async for e in plugin._handle_stream(ctx, inner)]

        assert events[0] is pre
        stops = [e for e in events if isinstance(e, EventLoopStopEvent)]
        assert len(stops) == 1
        assert stops[0]["stop"][1]["content"][0]["text"] == "answer"
        assert events[-1] is trail
        assert "tracking_id" in parent.messages[-1]

    @pytest.mark.asyncio
    async def test_delegation_reverify_failure_replays_original(self):
        """If the tool result is mutated to error after _on_after_tools, original stop replays."""
        from strands._middleware.stages import AgentStreamContext
        from strands.types._events import EventLoopStopEvent

        tool = _make_delegation_tool()
        parent = Agent(name="p", tools=[tool], callback_handler=None)
        plugin = _get_plugin(parent)

        # History where the tool result is an error (simulating late mutation)
        parent.messages.extend(
            [
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "sub", "input": {}}}]},
                {
                    "role": "user",
                    "content": [
                        {"toolResult": {"toolUseId": "t1", "status": "error", "content": [{"text": "failed"}]}}
                    ],
                },
                {"role": "assistant", "content": [{"text": "placeholder"}]},
            ]
        )

        ctx = AgentStreamContext(
            agent=parent,
            messages=[],
            invocation_state={"request_state": {}},
            _interrupts={},
        )

        original_stop = EventLoopStopEvent("end_turn", parent.messages[-1], parent.event_loop_metrics, {})

        async def inner(c):
            # Delegation state was set (as if _on_after_tools saw success), but history
            # was mutated to error by a late hook before _handle_stream runs.
            plugin._state[parent] = _DelegationState(tool_use_id="t1", end_turn_via_delegation=True, tool_use_count=1)
            yield original_stop

        events = [e async for e in plugin._handle_stream(ctx, inner)]

        # Re-verify fails → original stop replays unchanged, no delegation transformation
        assert len(events) == 1
        assert events[0] is original_stop

    @pytest.mark.asyncio
    async def test_absent_tool_result_skips_delegation(self):
        """When the tool result for the delegation toolUseId is absent from history, delegation is skipped."""
        from strands._middleware.stages import AgentStreamContext
        from strands.types._events import EventLoopStopEvent

        tool = _make_delegation_tool()
        parent = Agent(name="p", tools=[tool], callback_handler=None)
        plugin = _get_plugin(parent)

        # History has no tool result at all for the delegation toolUseId
        parent.messages.extend(
            [
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "sub", "input": {}}}]},
                {"role": "assistant", "content": [{"text": "placeholder"}]},
            ]
        )

        ctx = AgentStreamContext(
            agent=parent,
            messages=[],
            invocation_state={"request_state": {}},
            _interrupts={},
        )

        original_stop = EventLoopStopEvent("end_turn", parent.messages[-1], parent.event_loop_metrics, {})

        async def inner(c):
            plugin._state[parent] = _DelegationState(tool_use_id="t1", end_turn_via_delegation=True, tool_use_count=1)
            yield original_stop

        events = [e async for e in plugin._handle_stream(ctx, inner)]

        # Result absent → original stop replays unchanged
        assert len(events) == 1
        assert events[0] is original_stop


class TestStatefulModel:
    """Delegation tools are rejected on stateful models at init and runtime."""

    def test_init_raises_with_delegation_on_stateful(self):
        tool = _make_delegation_tool()
        stateful = MagicMock()
        type(stateful).stateful = PropertyMock(return_value=True)

        with pytest.raises(ValueError, match="not supported with stateful models"):
            Agent(name="p", model=stateful, tools=[tool], callback_handler=None)

    def test_init_ok_without_delegation_on_stateful(self):
        stateful = MagicMock()
        type(stateful).stateful = PropertyMock(return_value=True)
        Agent(name="p", model=stateful, callback_handler=None)

    @pytest.mark.asyncio
    async def test_runtime_stateful_delegate_returns_error(self):
        from strands._middleware.stages import ExecuteToolContext
        from strands.types._events import ToolResultEvent

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

            events = [e async for e in plugin._handle_tool_execution(ctx, unreachable)]
            assert len(events) == 1
            assert events[0].tool_result["status"] == "error"
            assert "stateful" in events[0].tool_result["content"][0]["text"].lower()
        finally:
            del type(parent.model).stateful


class TestAutoRegistration:
    def test_auto_registered(self):
        assert "strands:agent-delegation" in Agent(name="t", callback_handler=None)._plugin_registry._plugins

    def test_not_duplicated(self):
        p = AgentDelegation()
        agent = Agent(name="t", plugins=[p], callback_handler=None)
        assert agent._plugin_registry._plugins.get("strands:agent-delegation") is p


class TestJsonContentConversion:
    def test_converts_text_json_and_passthrough(self):
        import json

        blocks = _to_content_blocks(
            {
                "content": [
                    {"text": "hi"},
                    {"json": {"k": 1}},
                    {"image": {"format": "png", "source": {"bytes": b"x"}}},
                ]
            }
        )
        assert blocks[0] == {"text": "hi"}
        assert json.loads(blocks[1]["text"]) == {"k": 1}
        assert blocks[2] == {"image": {"format": "png", "source": {"bytes": b"x"}}}


class TestChildStructuredOutputSerialization:
    @pytest.mark.asyncio
    async def test_datetime_serializes_cleanly(self):
        import json
        from datetime import datetime

        from pydantic import BaseModel

        from strands.types._events import ToolResultEvent

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

        events = [e async for e in tool.stream({"toolUseId": "t1", "name": "sched", "input": {"input": "x"}}, {})]
        results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert results[0].tool_result["status"] == "success"
        assert "2025-01-15" in json.dumps(results[0].tool_result["content"][0]["json"])


class TestFullDelegationFlow:
    @pytest.mark.asyncio
    async def test_routes_to_specialist(self):
        from tests.fixtures.mocked_model_provider import MockedModelProvider

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

        result = await orch.invoke_async("Check balance")
        assert result.stop_reason == "end_turn"
        assert any("$42" in str(b.get("text", "")) for b in result.message["content"])

        # The placeholder must not remain stranded in history
        for msg in orch.messages:
            for block in msg.get("content", []):
                if isinstance(block, dict) and "text" in block:
                    assert "Turn ended early" not in block["text"]

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        from tests.fixtures.mocked_model_provider import MockedModelProvider

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

        result = await orch.invoke_async("Do it")
        assert result.stop_reason == "end_turn"
        assert "recovered" in str(result.message["content"]).lower()


class TestSessionPersistence:
    """Persisted history after delegation matches in-memory history."""

    @pytest.mark.asyncio
    async def test_persisted_matches_in_memory_after_delegation(self):
        from strands.session.repository_session_manager import RepositorySessionManager
        from tests.fixtures.mock_session_repository import MockedSessionRepository
        from tests.fixtures.mocked_model_provider import MockedModelProvider

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

        # Restore from the same repo into a fresh agent
        session_mgr_2 = RepositorySessionManager(session_id="s1", session_repository=repo)
        orch_2 = Agent(model=orch_model, name="orchestrator", session_manager=session_mgr_2, callback_handler=None)

        # Persisted messages must match in-memory messages exactly
        assert len(orch_2.messages) == len(orch.messages)
        for restored, live in zip(orch_2.messages, orch.messages, strict=True):
            assert restored["role"] == live["role"]
            # Content comparison (ignore metadata/tracking_id differences)
            assert restored["content"] == live["content"]

        # The final message is the delegation content, not the placeholder
        assert orch_2.messages[-1]["role"] == "assistant"
        assert any("$42" in str(b.get("text", "")) for b in orch_2.messages[-1]["content"])


class TestDelegationWithContextOffloader:
    """ContextOffloader must not offload delegation tool results."""

    @pytest.mark.asyncio
    async def test_large_delegation_result_not_offloaded(self):
        """A delegation result exceeding the offloader threshold stays in context."""
        from strands.vended_plugins.context_offloader import ContextOffloader, InMemoryStorage
        from tests.fixtures.mocked_model_provider import MockedModelProvider

        storage = InMemoryStorage()
        offloader = ContextOffloader(storage=storage, max_result_tokens=25, preview_tokens=10)

        large_answer = "x" * 500  # well above 25-token threshold
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

        result = await orch.invoke_async("Check balance")

        # The delegation result must appear in the final message unmodified
        assert result.stop_reason == "end_turn"
        final_text = "".join(block.get("text", "") for block in result.message["content"] if isinstance(block, dict))
        assert large_answer in final_text

        # Nothing was stored in the offloader
        assert len(storage._store) == 0


class TestAfterToolsOrderingGuard:
    """SDK_LAST on AfterToolsEvent makes delegation read the result after other hooks have mutated it."""

    @pytest.mark.asyncio
    async def test_late_hook_flipping_result_prevents_end_turn(self):
        """A default-order AfterToolsEvent hook that flips the result to error prevents delegation end_turn.

        Order priority always wins: DEFAULT (0) runs before SDK_LAST (100). Reverse ordering only flips
        registration order *within* one order tier, so delegation's SDK_LAST hook runs last and reads the
        already-mutated committed message. Its own check in _on_after_tools is what declines to set
        end_turn here; _handle_stream's re-verification is a second line of defence that does not fire in
        this scenario. This test verifies the agent loop continues rather than ending the turn.
        """
        from tests.fixtures.mocked_model_provider import MockedModelProvider

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
                    # After recovery from the flipped result, the model responds with text
                    {"role": "assistant", "content": [{"text": "I continued after flip."}]},
                ]
            ),
            name="orch",
            tools=[sub.as_tool(delegate=True)],
            callback_handler=None,
        )

        from strands.hooks import AfterToolsEvent

        def flip_result_to_error(event: AfterToolsEvent):
            """Simulate a hook that mutates the tool result to error after delegation saw it."""
            content = event.message.get("content", [])
            for block in content:
                if isinstance(block, dict) and "toolResult" in block:
                    block["toolResult"]["status"] = "error"

        # Register at DEFAULT order — in reverse ordering this runs AFTER SDK_LAST (delegation)
        orch.add_hook(flip_result_to_error, AfterToolsEvent)

        result = await orch.invoke_async("Do it")

        # The agent must have continued (not stopped at the flipped delegation) and produced a second reply
        assert result.stop_reason == "end_turn"
        assert "continued" in str(result.message["content"]).lower()


class TestMessageAddedEventDuringDelegation:
    """Delegation fires MessageAddedEvent for the delegation message."""

    @pytest.mark.asyncio
    async def test_message_added_event_fires_for_delegation_message(self):
        """MessageAddedEvent fires for the placeholder and then the real delegation content."""
        from strands.hooks import MessageAddedEvent
        from tests.fixtures.mocked_model_provider import MockedModelProvider

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

        # Expected events: user prompt, assistant tool_use, tool_result, end_turn placeholder, delegation content
        assert len(received_messages) == 5

        # The placeholder fires before the delegation content
        placeholder_msgs = [
            m
            for m in received_messages
            if m["role"] == "assistant"
            and any("Turn ended early" in str(b.get("text", "")) for b in m.get("content", []))
        ]
        assert len(placeholder_msgs) == 1

        # The delegation message (containing the sub-agent's answer) fires exactly once
        delegation_msgs = [
            m
            for m in received_messages
            if m["role"] == "assistant" and any("$42" in str(b.get("text", "")) for b in m.get("content", []))
        ]
        assert len(delegation_msgs) == 1

        # Placeholder comes before delegation content
        placeholder_idx = received_messages.index(placeholder_msgs[0])
        delegation_idx = received_messages.index(delegation_msgs[0])
        assert placeholder_idx < delegation_idx

    @pytest.mark.asyncio
    async def test_session_manager_suppresses_delegation_event(self):
        """With a session manager, subscribers see the placeholder but not the delegation content event."""
        from strands.hooks import MessageAddedEvent
        from strands.session.repository_session_manager import RepositorySessionManager
        from tests.fixtures.mock_session_repository import MockedSessionRepository
        from tests.fixtures.mocked_model_provider import MockedModelProvider

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

        # With session manager: no MessageAddedEvent fires for the delegation content
        delegation_msgs = [
            m
            for m in received_messages
            if m["role"] == "assistant" and any("$42" in str(b.get("text", "")) for b in m.get("content", []))
        ]
        assert len(delegation_msgs) == 0

        # But agent.messages still reflects the delegation content
        assert any("$42" in str(b.get("text", "")) for b in orch.messages[-1].get("content", []))


class TestMiddlewareSingleCallGuard:
    """ExecuteToolStage middleware rejects delegation in multi-tool batch."""

    @pytest.mark.asyncio
    async def test_middleware_rejects_when_batch_count_exceeds_one(self):
        from strands._middleware.stages import ExecuteToolContext
        from strands.types._events import ToolResultEvent

        tool = _make_delegation_tool()
        parent = Agent(name="p", tools=[tool], callback_handler=None)
        plugin = _get_plugin(parent)

        # Simulate BeforeToolsEvent having already set batch count > 1
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

        events = [e async for e in plugin._handle_tool_execution(ctx, unreachable)]

        assert len(events) == 1
        assert events[0].tool_result["status"] == "error"
        assert "only tool" in events[0].tool_result["content"][0]["text"].lower()
