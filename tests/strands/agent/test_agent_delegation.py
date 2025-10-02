"""Tests for Agent delegation functionality.

These tests exercise actual delegation behavior against the implementation
in ``strands.event_loop.event_loop`` rather than re-implementing the logic
inside the tests. This keeps the suite aligned with production behavior.
"""

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from strands import Agent, tool
from strands.agent.agent_result import AgentResult
from strands.agent.state import AgentState
from strands.event_loop.event_loop import _handle_delegation
from strands.telemetry.metrics import EventLoopMetrics, Trace
from strands.types.exceptions import AgentDelegationException
from tests.fixtures.mocked_model_provider import MockedModelProvider


def _make_agent(name: str) -> Agent:
    return Agent(name=name, model=MockedModelProvider([]))


def _make_agent_result(text: str = "delegated") -> AgentResult:
    return AgentResult(
        stop_reason="end_turn",
        message={"role": "assistant", "content": [{"text": text}]},
        metrics=EventLoopMetrics(),
        state={}
    )


async def _run_delegation(
    orchestrator: Agent,
    sub_agent: Agent,
    exception: AgentDelegationException,
    invocation_state: dict[str, Any] | None = None,
) -> tuple[AgentResult, Trace]:
    orchestrator._sub_agents[sub_agent.name] = sub_agent
    cycle_trace = Trace("cycle")
    result = await _handle_delegation(
        agent=orchestrator,
        delegation_exception=exception,
        invocation_state=invocation_state or {},
        cycle_trace=cycle_trace,
        cycle_span=None,
    )
    return result, cycle_trace


class DummySessionManager:
    """Minimal session manager used to validate nested session creation."""

    def __init__(self, session_id: str, *_: Any, **__: Any) -> None:
        self.session_id = session_id
        self.saved_agents: list[Agent] = []

    async def save_agent(self, agent: Agent) -> None:  # pragma: no cover - simple helper
        self.saved_agents.append(agent)

    async def sync_agent(self, agent: Agent) -> None:  # pragma: no cover - compatibility stub
        return None

    def redact_latest_message(self, *_: Any, **__: Any) -> None:  # pragma: no cover - compatibility stub
        return None


class TestAgentDelegationValidation:
    """Validation rules should reflect actual Agent behavior."""

    def test_unique_names_enforcement(self) -> None:
        orchestrator = _make_agent("Orchestrator")
        sub_agent1 = _make_agent("SubAgent")
        sub_agent2 = _make_agent("SubAgent")

        with pytest.raises(ValueError, match="Sub-agent names must be unique"):
            orchestrator._validate_sub_agents([sub_agent1, sub_agent2])

    def test_circular_reference_prevention(self) -> None:
        orchestrator = _make_agent("Orchestrator")

        with pytest.raises(ValueError, match="Agent cannot delegate to itself"):
            orchestrator._validate_sub_agents([orchestrator])

    def test_tool_name_conflict_detection(self) -> None:
        @tool
        def handoff_to_subagent(message: str) -> dict:
            return {"content": [{"text": message}]}

        orchestrator = Agent(
            name="Orchestrator",
            model=MockedModelProvider([]),
            tools=[handoff_to_subagent],
        )
        sub_agent = _make_agent("SubAgent")

        with pytest.raises(ValueError, match="Tool name conflict"):
            orchestrator._validate_sub_agents([sub_agent])

    def test_cross_provider_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING)

        orchestrator = _make_agent("Orchestrator")
        orchestrator.model = Mock()
        orchestrator.model.config = {"provider": "anthropic"}

        sub_agent = _make_agent("SubAgent")
        sub_agent.model = Mock()
        sub_agent.model.config = {"provider": "openai"}

        orchestrator._validate_sub_agents([sub_agent])

        assert "Model provider mismatch" in caplog.text
        assert "anthropic" in caplog.text
        assert "openai" in caplog.text


class TestDynamicSubAgentManagement:
    """Ensure dynamic sub-agent changes update tools and registry."""

    def test_add_sub_agent_registers_tool(self) -> None:
        orchestrator = _make_agent("Orchestrator")
        sub_agent = _make_agent("NewAgent")

        orchestrator.add_sub_agent(sub_agent)

        assert orchestrator._sub_agents["NewAgent"] is sub_agent
        assert "handoff_to_newagent" in orchestrator.tool_registry.registry

    def test_remove_sub_agent_cleans_up(self) -> None:
        orchestrator = _make_agent("Orchestrator")
        sub_agent = _make_agent("TestAgent")
        orchestrator.add_sub_agent(sub_agent)

        removed = orchestrator.remove_sub_agent("TestAgent")

        assert removed is True
        assert "TestAgent" not in orchestrator._sub_agents
        assert "handoff_to_testagent" not in orchestrator.tool_registry.registry

    def test_remove_nonexistent_sub_agent(self) -> None:
        orchestrator = _make_agent("Orchestrator")
        assert orchestrator.remove_sub_agent("Missing") is False

    def test_add_duplicate_name_preserves_existing(self) -> None:
        orchestrator = _make_agent("Orchestrator")
        existing = _make_agent("Existing")
        orchestrator.add_sub_agent(existing)

        duplicate = _make_agent("Existing")
        with pytest.raises(ValueError, match="Tool name conflict"):
            orchestrator.add_sub_agent(duplicate)

        assert orchestrator._sub_agents["Existing"] is existing
        assert len(orchestrator._sub_agents) == 1

    def test_sub_agents_property_returns_copy(self) -> None:
        orchestrator = _make_agent("Orchestrator")
        orchestrator.add_sub_agent(_make_agent("Primary"))

        sub_agents_copy = orchestrator.sub_agents
        sub_agents_copy["Injected"] = _make_agent("Injected")

        assert "Injected" not in orchestrator._sub_agents
        assert len(orchestrator._sub_agents) == 1


def _delegation_trace(cycle_trace: Trace) -> Trace:
    assert cycle_trace.children, "delegation trace not recorded"
    return cycle_trace.children[0]


class AttrDict(dict):
    def __getattr__(self, item: str) -> Any:
        if item in self:
            return self[item]
        raise AttributeError(item)


class AsyncStream:
    def __init__(self, events: list[Any]) -> None:
        self._events = iter(events)

    def __aiter__(self) -> "AsyncStream":
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._events)
        except StopIteration as exc:  # pragma: no cover - completion path
            raise StopAsyncIteration from exc

    def __await__(self):
        async def _ready() -> "AsyncStream":
            return self

        return _ready().__await__()


@pytest.mark.asyncio
class TestDelegationTransferBehavior:
    async def test_state_transferred_as_deepcopy(self) -> None:
        orchestrator = _make_agent("Orch")
        orchestrator.state = {"shared": {"count": 1}}

        sub_agent = _make_agent("Sub")
        sub_agent.state = {"previous": True}
        sub_agent.invoke_async = AsyncMock(return_value=_make_agent_result("done"))

        exception = AgentDelegationException(
            target_agent=sub_agent.name,
            message="Handle request",
            context={},
            delegation_chain=[],
            transfer_state=True,
            transfer_messages=False,
        )

        result, trace = await _run_delegation(orchestrator, sub_agent, exception)

        assert result.message["content"][0]["text"] == "done"
        assert sub_agent.state == orchestrator.state
        assert sub_agent.state is not orchestrator.state

        sub_agent.state["shared"]["count"] = 2
        assert orchestrator.state["shared"]["count"] == 1

    async def test_state_transfer_respects_flag(self) -> None:
        orchestrator = _make_agent("Orch")
        orchestrator.state = {"shared": "value"}

        sub_agent = _make_agent("Sub")
        original_state = {"keep": True}
        sub_agent.state = original_state.copy()
        sub_agent.invoke_async = AsyncMock(return_value=_make_agent_result())

        exception = AgentDelegationException(
            target_agent=sub_agent.name,
            message="No state please",
            context={},
            delegation_chain=[],
            transfer_state=False,
            transfer_messages=False,
        )

        await _run_delegation(orchestrator, sub_agent, exception)

        assert sub_agent.state == original_state

    async def test_message_filtering_applies_engine_rules(self) -> None:
        orchestrator = _make_agent("Orch")
        orchestrator.messages = [
            {"role": "system", "content": [{"type": "text", "text": "System prompt"}]},
            {"role": "user", "content": [{"type": "text", "text": "User question"}]},
            {"role": "assistant", "content": [{"type": "toolUse", "name": "internal_tool"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Relevant reply"}]},
        ]

        sub_agent = _make_agent("Sub")
        sub_agent.messages = []
        sub_agent.invoke_async = AsyncMock(return_value=_make_agent_result())

        exception = AgentDelegationException(
            target_agent=sub_agent.name,
            message="Process",
            context={},
            delegation_chain=[],
            transfer_state=False,
            transfer_messages=True,
        )

        _, trace = await _run_delegation(orchestrator, sub_agent, exception)

        filtered_history = sub_agent.messages[:-1]  # last message is delegation context
        assert [msg["role"] for msg in filtered_history] == ["system", "user", "assistant"]
        assert all(
            all("toolUse" not in block for block in msg.get("content", []))
            for msg in filtered_history
        )

        delegation_msg = sub_agent.messages[-1]
        assert "Delegated from Orch" in delegation_msg["content"][0]["text"]

        metadata = _delegation_trace(trace).metadata["message_filtering_applied"]
        assert metadata["original_message_count"] == 4
        assert metadata["filtered_message_count"] == 3
        assert metadata["noise_removed"] == 1
        assert metadata["compression_ratio"] == "75.0%"

    async def test_message_transfer_disabled_skips_history(self) -> None:
        orchestrator = _make_agent("Orch")
        orchestrator.messages = [
            {"role": "system", "content": [{"text": "Root"}]},
            {"role": "user", "content": [{"text": "Request"}]},
        ]

        sub_agent = _make_agent("Sub")
        sub_agent.messages = [{"role": "assistant", "content": [{"text": "pre-existing"}]}]
        sub_agent.invoke_async = AsyncMock(return_value=_make_agent_result())

        exception = AgentDelegationException(
            target_agent=sub_agent.name,
            message="Handle locally",
            context={},
            delegation_chain=[],
            transfer_state=False,
            transfer_messages=False,
        )

        await _run_delegation(orchestrator, sub_agent, exception)

        assert len(sub_agent.messages) == 1
        assert "Delegated from Orch" in sub_agent.messages[0]["content"][0]["text"]

    async def test_additional_context_appended(self) -> None:
        orchestrator = _make_agent("Orch")
        sub_agent = _make_agent("Sub")
        sub_agent.invoke_async = AsyncMock(return_value=_make_agent_result())

        context_payload = {"user_id": "123", "priority": "high"}
        exception = AgentDelegationException(
            target_agent=sub_agent.name,
            message="Need context",
            context=context_payload,
            delegation_chain=[],
            transfer_state=False,
            transfer_messages=False,
        )

        await _run_delegation(orchestrator, sub_agent, exception)

        assert len(sub_agent.messages) == 2
        assert "Additional context" in sub_agent.messages[-1]["content"][0]["text"]
        assert "user_id" in sub_agent.messages[-1]["content"][0]["text"]


@pytest.mark.asyncio
class TestDelegationStateSerializer:
    async def test_custom_serializer_used(self) -> None:
        class CustomState:
            def __init__(self, data: str) -> None:
                self.data = data

        orchestrator = _make_agent("Orch")
        orchestrator.state = CustomState("payload")

        def serializer(state: Any) -> Any:
            assert isinstance(state, CustomState)
            return {"serialized": True, "data": state.data}

        orchestrator.delegation_state_serializer = serializer

        sub_agent = _make_agent("Sub")
        sub_agent.invoke_async = AsyncMock(return_value=_make_agent_result())

        exception = AgentDelegationException(
            target_agent=sub_agent.name,
            message="serialize",
            context={},
            delegation_chain=[],
            transfer_state=True,
            transfer_messages=False,
        )

        await _run_delegation(orchestrator, sub_agent, exception)

        assert sub_agent.state == {"serialized": True, "data": "payload"}

    async def test_serializer_error_falls_back_to_deepcopy(self) -> None:
        orchestrator = _make_agent("Orch")
        orchestrator.state = {"data": [1, 2, 3]}

        def failing_serializer(_: Any) -> Any:
            raise ValueError("Serialization failed")

        orchestrator.delegation_state_serializer = failing_serializer

        sub_agent = _make_agent("Sub")
        sub_agent.invoke_async = AsyncMock(return_value=_make_agent_result())

        exception = AgentDelegationException(
            target_agent=sub_agent.name,
            message="serialize",
            context={},
            delegation_chain=[],
            transfer_state=True,
            transfer_messages=False,
        )

        _, trace = await _run_delegation(orchestrator, sub_agent, exception)

        assert sub_agent.state == orchestrator.state
        assert sub_agent.state is not orchestrator.state

        metadata = _delegation_trace(trace).metadata["state_serialization_error"]
        assert metadata["fallback_to_deepcopy"] is True
        assert metadata["error"] == "Serialization failed"


@pytest.mark.asyncio
class TestDelegationSessionAndTimeouts:
    async def test_creates_nested_session_manager(self) -> None:
        orchestrator = _make_agent("Orch")
        orchestrator._session_manager = DummySessionManager("root-session")

        sub_agent = _make_agent("Sub")
        sub_agent.invoke_async = AsyncMock(return_value=_make_agent_result())

        exception = AgentDelegationException(
            target_agent=sub_agent.name,
            message="sess",
            context={},
            delegation_chain=[],
            transfer_state=False,
            transfer_messages=False,
        )

        await _run_delegation(orchestrator, sub_agent, exception)

        assert orchestrator._session_manager.session_id == "root-session"
        assert sub_agent._session_manager.session_id.startswith("root-session/delegation/")
        assert sub_agent._session_manager.saved_agents == [sub_agent]

    async def test_timeout_raises_structured_error(self) -> None:
        orchestrator = _make_agent("Orch")
        orchestrator.delegation_timeout = 0.01

        sub_agent = _make_agent("Slow")

        async def slow_invoke() -> AgentResult:
            await asyncio.sleep(0.1)
            return _make_agent_result("late")

        sub_agent.invoke_async = slow_invoke

        exception = AgentDelegationException(
            target_agent=sub_agent.name,
            message="timeout",
            context={},
            delegation_chain=[],
            transfer_state=False,
            transfer_messages=False,
        )

        with pytest.raises(TimeoutError, match="timed out"):
            await _run_delegation(orchestrator, sub_agent, exception)


@pytest.mark.asyncio
class TestDelegationStreaming:
    async def test_streaming_proxy_returns_result(self) -> None:
        orchestrator = _make_agent("Orch")
        orchestrator.delegation_streaming_proxy = True

        sub_agent = _make_agent("StreamSub")
        sub_agent.invoke_async = AsyncMock(return_value=_make_agent_result("fallback"))
        sub_agent.stream_async = Mock()

        exception = AgentDelegationException(
            target_agent=sub_agent.name,
            message="stream",
            context={},
            delegation_chain=[],
            transfer_state=False,
            transfer_messages=False,
        )

        result, trace = await _run_delegation(orchestrator, sub_agent, exception)

        assert result.message["content"][0]["text"] == "fallback"
        sub_agent.invoke_async.assert_awaited_once()
        sub_agent.stream_async.assert_not_called()

        metadata = _delegation_trace(trace).metadata["delegation_complete"]
        assert metadata["streaming_proxied"] is True