"""Tests for the ``use_agent`` vended tool.

The tool builds a fresh :class:`~strands.Agent` per invocation using a mocked
model provider so the tests never touch a real provider. Coverage focuses on
the security surface first (allowlist, caps, recursion, model inheritance)
and only exercises the happy path once end-to-end.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any
from unittest.mock import patch

import pytest

from strands.agent.agent import Agent
from strands.tools.decorator import tool
from strands.types.tools import ToolContext
from strands.vended_tools.use_agent import (
    MultiagentDepthExceeded,
    make_use_agent,
    use_agent,
)
from strands.vended_tools.use_agent.use_agent import (
    _MAX_DEPTH,
    _MAX_SYSTEM_PROMPT_BYTES,
    _MAX_TASK_BYTES,
    _MAX_TOOL_ALLOWLIST,
)
from tests.fixtures.mocked_model_provider import MockedModelProvider


@tool
def _fake_search(query: str) -> str:
    """A stub tool that a parent might expose to a child.

    Args:
        query: The query string to look up.
    """
    return f"results for {query}"


@tool
def _fake_second(payload: str) -> str:
    """A second stub tool used for allowlist tests.

    Args:
        payload: An arbitrary payload string.
    """
    return payload


def _parent_agent(tools: list[Any] | None = None, name: str = "parent") -> Agent:
    model = MockedModelProvider([{"role": "assistant", "content": [{"text": "unused"}]}])
    return Agent(model=model, name=name, tools=tools or [], callback_handler=None)


def _tool_context(agent: Agent, invocation_state: dict[str, Any] | None = None) -> ToolContext:
    return ToolContext(
        tool_use={"name": "use_agent", "toolUseId": "test-id", "input": {}},
        agent=agent,
        invocation_state=invocation_state or {},
    )


def _fake_result(text: str, stop_reason: str = "end_turn") -> Any:
    from strands.agent.agent_result import AgentResult
    from strands.telemetry.metrics import EventLoopMetrics

    return AgentResult(
        stop_reason=stop_reason,
        message={"role": "assistant", "content": [{"text": text}]},
        metrics=EventLoopMetrics(),
        state={},
    )


class TestInputValidation:
    @pytest.mark.asyncio
    async def test_rejects_empty_system_prompt(self):
        parent = _parent_agent()
        with pytest.raises(ValueError, match="system_prompt must be non-empty"):
            await use_agent(
                system_prompt="   ",
                task="do the thing",
                tool_context=_tool_context(parent),
            )

    @pytest.mark.asyncio
    async def test_rejects_oversized_system_prompt(self):
        parent = _parent_agent()
        with pytest.raises(ValueError, match="exceeds size cap"):
            await use_agent(
                system_prompt="a" * (_MAX_SYSTEM_PROMPT_BYTES + 1),
                task="do the thing",
                tool_context=_tool_context(parent),
            )

    @pytest.mark.asyncio
    async def test_rejects_empty_task(self):
        parent = _parent_agent()
        with pytest.raises(ValueError, match="task must be non-empty"):
            await use_agent(
                system_prompt="be helpful",
                task="",
                tool_context=_tool_context(parent),
            )

    @pytest.mark.asyncio
    async def test_rejects_oversized_task(self):
        parent = _parent_agent()
        with pytest.raises(ValueError, match="exceeds size cap"):
            await use_agent(
                system_prompt="be helpful",
                task="a" * (_MAX_TASK_BYTES + 1),
                tool_context=_tool_context(parent),
            )


class TestToolAllowlist:
    @pytest.mark.asyncio
    async def test_rejects_wildcard_entry(self):
        parent = _parent_agent(tools=[_fake_search])
        with pytest.raises(ValueError, match="wildcard"):
            await use_agent(
                system_prompt="be helpful",
                task="do it",
                tool_context=_tool_context(parent),
                tools=["*"],
            )

    @pytest.mark.asyncio
    async def test_rejects_unknown_tool_name(self):
        parent = _parent_agent(tools=[_fake_search])
        with pytest.raises(ValueError, match="not present in the parent agent's tool registry"):
            await use_agent(
                system_prompt="be helpful",
                task="do it",
                tool_context=_tool_context(parent),
                tools=["_fake_search", "nope_not_a_tool"],
            )

    @pytest.mark.asyncio
    async def test_rejects_non_string_entry(self):
        parent = _parent_agent(tools=[_fake_search])
        with pytest.raises(ValueError, match="tools entries must be strings"):
            await use_agent(
                system_prompt="be helpful",
                task="do it",
                tool_context=_tool_context(parent),
                tools=[123],  # type: ignore[list-item]
            )

    @pytest.mark.asyncio
    async def test_rejects_non_list_tools(self):
        parent = _parent_agent()
        with pytest.raises(ValueError, match="tools must be a list"):
            await use_agent(
                system_prompt="be helpful",
                task="do it",
                tool_context=_tool_context(parent),
                tools="_fake_search",  # type: ignore[arg-type]
            )

    @pytest.mark.asyncio
    async def test_rejects_multiagent_tool_names(self):
        # A parent registry might contain these; we still refuse to nest them.
        parent = _parent_agent(tools=[_fake_search])
        for name in ("use_agent", "swarm", "graph", "a2a_client"):
            with pytest.raises(ValueError, match="multi-agent tool"):
                await use_agent(
                    system_prompt="be helpful",
                    task="do it",
                    tool_context=_tool_context(parent),
                    tools=[name],
                )

    @pytest.mark.asyncio
    async def test_rejects_oversized_allowlist(self):
        parent = _parent_agent(tools=[_fake_search])
        with pytest.raises(ValueError, match=f"allowlist exceeds cap of {_MAX_TOOL_ALLOWLIST}"):
            await use_agent(
                system_prompt="be helpful",
                task="do it",
                tool_context=_tool_context(parent),
                tools=["_fake_search"] * (_MAX_TOOL_ALLOWLIST + 1),
            )

    @pytest.mark.asyncio
    async def test_deduplicates_and_preserves_order(self):
        parent = _parent_agent(tools=[_fake_search, _fake_second])
        captured: dict[str, Any] = {}

        # Capture the Agent constructor call so we can inspect the resolved
        # tool list and its order.
        original_init = Agent.__init__

        def spy_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            if "tools" in kwargs and "use_agent" in kwargs.get("name", ""):
                captured["tools"] = list(kwargs["tools"])
            return original_init(self, *args, **kwargs)

        async def fake_invoke_async(self, prompt, *, invocation_state=None, **kwargs):
            return _fake_result("done")

        with (
            patch.object(Agent, "__init__", spy_init),
            patch.object(Agent, "invoke_async", fake_invoke_async),
        ):
            await use_agent(
                system_prompt="be helpful",
                task="do it",
                tool_context=_tool_context(parent),
                tools=["_fake_search", "_fake_second", "_fake_search"],
            )

        # Order-preserving, deduplicated.
        assert [t.tool_name for t in captured["tools"]] == ["_fake_search", "_fake_second"]


class TestRecursion:
    @pytest.mark.asyncio
    async def test_refuses_at_cap(self):
        parent = _parent_agent()
        ctx = _tool_context(parent, invocation_state={"multiagent_depth": _MAX_DEPTH})
        with pytest.raises(MultiagentDepthExceeded, match="recursion depth cap"):
            await use_agent(
                system_prompt="be helpful",
                task="do it",
                tool_context=ctx,
            )

    @pytest.mark.asyncio
    async def test_forwards_incremented_depth_to_child(self):
        parent = _parent_agent()
        captured: dict[str, Any] = {}

        async def fake_invoke_async(self, prompt, *, invocation_state=None, **kwargs):
            captured["invocation_state"] = invocation_state
            return _fake_result("done")

        with patch.object(Agent, "invoke_async", fake_invoke_async):
            await use_agent(
                system_prompt="be helpful",
                task="do it",
                tool_context=_tool_context(parent, invocation_state={"multiagent_depth": 1}),
            )
        assert captured["invocation_state"] == {"multiagent_depth": 2}

    @pytest.mark.asyncio
    async def test_preserves_parent_invocation_state_and_overrides_depth(self):
        parent = _parent_agent()
        captured: dict[str, Any] = {}

        async def fake_invoke_async(self, prompt, *, invocation_state=None, **kwargs):
            captured["invocation_state"] = invocation_state
            return _fake_result("done")

        parent_state = {"trace_id": "abc-123", "run_id": "xyz", "multiagent_depth": 0}
        with patch.object(Agent, "invoke_async", fake_invoke_async):
            await use_agent(
                system_prompt="be helpful",
                task="do it",
                tool_context=_tool_context(parent, invocation_state=parent_state),
            )
        assert captured["invocation_state"] == {
            "trace_id": "abc-123",
            "run_id": "xyz",
            "multiagent_depth": 1,
        }

    def test_factory_rejects_non_finite_max_depth(self):
        for bad in (math.inf, math.nan, -math.inf):
            with pytest.raises(ValueError):
                make_use_agent(max_depth=bad)  # type: ignore[arg-type]

    def test_factory_rejects_boolean_max_depth(self):
        # ``bool`` is an ``int`` subclass; explicit rejection prevents
        # ``True`` silently coercing to a cap of one.
        with pytest.raises(ValueError):
            make_use_agent(max_depth=True)  # type: ignore[arg-type]

    def test_factory_rejects_non_positive_max_depth(self):
        for bad in (0, -1):
            with pytest.raises(ValueError):
                make_use_agent(max_depth=bad)

    @pytest.mark.asyncio
    async def test_boolean_depth_in_invocation_state_treated_as_zero(self):
        # ``bool`` is an ``int`` subclass; if the counter accepted ``True``
        # as depth 1, an attacker could set it to False and reset the counter.
        parent = _parent_agent()
        captured: dict[str, Any] = {}

        async def fake_invoke_async(self, prompt, *, invocation_state=None, **kwargs):
            captured["invocation_state"] = invocation_state
            return _fake_result("done")

        with patch.object(Agent, "invoke_async", fake_invoke_async):
            await use_agent(
                system_prompt="be helpful",
                task="do it",
                tool_context=_tool_context(parent, invocation_state={"multiagent_depth": True}),
            )
        # ``True`` treated as depth 0, child sees depth 1.
        assert captured["invocation_state"]["multiagent_depth"] == 1


class TestModelInheritance:
    @pytest.mark.asyncio
    async def test_child_inherits_parent_model(self):
        parent = _parent_agent()
        captured: dict[str, Any] = {}

        original_init = Agent.__init__

        def spy_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            if "use_agent" in kwargs.get("name", ""):
                captured["model"] = kwargs.get("model")
            return original_init(self, *args, **kwargs)

        async def fake_invoke_async(self, prompt, *, invocation_state=None, **kwargs):
            return _fake_result("done")

        with (
            patch.object(Agent, "__init__", spy_init),
            patch.object(Agent, "invoke_async", fake_invoke_async),
        ):
            await use_agent(
                system_prompt="be helpful",
                task="do it",
                tool_context=_tool_context(parent),
            )

        assert captured["model"] is parent.model


class TestCancellation:
    @pytest.mark.asyncio
    async def test_parent_cancel_propagates_and_returns_cancelled(self):
        parent = _parent_agent()

        async def slow_invoke(self, prompt, *, invocation_state=None, **kwargs):
            while not self._cancel_signal.is_set():
                await asyncio.sleep(0.01)
            return _fake_result("partial", stop_reason="cancelled")

        with patch.object(Agent, "invoke_async", slow_invoke):
            task = asyncio.create_task(
                use_agent(
                    system_prompt="be helpful",
                    task="a long-running task",
                    tool_context=_tool_context(parent),
                )
            )
            await asyncio.sleep(0.05)
            parent._cancel_signal.set()

            result = await task

        assert result["status"] == "cancelled"
        assert isinstance(result["output"], str)
        assert isinstance(result["execution_time_ms"], int)


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_returns_child_final_text(self):
        parent = _parent_agent(tools=[_fake_search])

        async def fake_invoke_async(self, prompt, *, invocation_state=None, **kwargs):
            return _fake_result("hello from the child")

        with patch.object(Agent, "invoke_async", fake_invoke_async):
            result = await use_agent(
                system_prompt="be a helper",
                task="say hi",
                tool_context=_tool_context(parent),
                tools=["_fake_search"],
            )
        assert result["status"] == "completed"
        assert "hello from the child" in result["output"]
        assert isinstance(result["execution_time_ms"], int)


class TestStopReasonMapping:
    """Non-happy stop reasons must surface as ``failed``, not ``completed``.

    A child that hit ``limit_turns`` or a content filter did not deliver on
    the delegated task; presenting that as a completed delegation would let
    an adversarial or misaligned child hide failures from the parent.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stop_reason",
        [
            "limit_turns",
            "content_filtered",
            "max_tokens",
            "guardrail_intervened",
            "limit_output_tokens",
            "limit_total_tokens",
        ],
    )
    async def test_non_end_turn_stop_reasons_are_failed(self, stop_reason: str):
        parent = _parent_agent()

        async def fake_invoke_async(self, prompt, *, invocation_state=None, **kwargs):
            return _fake_result("partial output", stop_reason=stop_reason)

        with patch.object(Agent, "invoke_async", fake_invoke_async):
            result = await use_agent(
                system_prompt="be a helper",
                task="do it",
                tool_context=_tool_context(parent),
            )
        assert result["status"] == "failed"
        assert "partial output" in result["output"]
        assert isinstance(result["execution_time_ms"], int)

    @pytest.mark.asyncio
    async def test_end_turn_maps_to_completed(self):
        parent = _parent_agent()

        async def fake_invoke_async(self, prompt, *, invocation_state=None, **kwargs):
            return _fake_result("done", stop_reason="end_turn")

        with patch.object(Agent, "invoke_async", fake_invoke_async):
            result = await use_agent(
                system_prompt="be a helper",
                task="do it",
                tool_context=_tool_context(parent),
            )
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_interrupt_maps_to_interrupted(self):
        parent = _parent_agent()

        async def fake_invoke_async(self, prompt, *, invocation_state=None, **kwargs):
            return _fake_result("paused", stop_reason="interrupt")

        with patch.object(Agent, "invoke_async", fake_invoke_async):
            result = await use_agent(
                system_prompt="be a helper",
                task="do it",
                tool_context=_tool_context(parent),
            )
        assert result["status"] == "interrupted"


class TestToolSpec:
    def test_name_and_required_fields(self):
        spec = use_agent.tool_spec
        assert spec["name"] == "use_agent"
        properties = spec["inputSchema"]["json"]["properties"]
        for required in ("system_prompt", "task"):
            assert required in properties
        assert "tools" in properties
        # The credential-injection surface is deliberately absent.
        assert "model_provider" not in properties
        assert "model_settings" not in properties
