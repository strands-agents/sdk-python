"""Tests for the swarm vended tool.

Focus: the tool boundary — input validation and result mapping — since the
swarm's execution semantics themselves are covered by the SDK's own
``multiagent/test_swarm.py``. The happy-path test patches the ``Swarm`` class
so we exercise the shim without spinning up model traffic.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strands.multiagent.base import NodeResult, Status
from strands.multiagent.swarm import SwarmResult
from strands.types.event_loop import Metrics, Usage
from strands.types.tools import ToolContext
from strands.vended_tools.swarm import MultiagentDepthExceeded, make_swarm, swarm

# The package's `__init__` re-exports `swarm` (the tool object), which shadows
# the like-named submodule. Grab the module object via importlib so we can patch
# its `Swarm` attribute directly without wrestling with the dotted-path lookup.
_swarm_module = importlib.import_module("strands.vended_tools.swarm.swarm")


class _FakeTool:
    def __init__(self, name: str) -> None:
        self.tool_name = name


class _FakeRegistry:
    def __init__(self, registry: dict[str, Any] | None = None) -> None:
        self.registry = registry or {}


class _FakeCancelSignal:
    def __init__(self, set_: bool = False) -> None:
        self._set = set_

    def is_set(self) -> bool:
        return self._set


class _FakeModel:
    """Model stub with just enough surface to survive Agent.__init__."""

    stateful = False
    context_window_limit = None

    def update_config(self, **_: Any) -> None:
        pass

    def get_config(self) -> dict[str, Any]:
        return {}

    async def structured_output(self, *_args: Any, **_kwargs: Any):
        yield {}

    async def stream(self, *_args: Any, **_kwargs: Any):
        yield {}


def _spec(name: str = "a", **overrides: Any) -> dict[str, Any]:
    """Build a minimal valid child spec.

    Per the shared multi-agent dialect (`_multiagent_conventions.md`), `name`,
    `system_prompt`, and `tools` are all required. `tools` may be empty.
    Individual tests override fields to exercise specific validation branches.
    """
    base: dict[str, Any] = {"name": name, "system_prompt": "you are helpful", "tools": []}
    base.update(overrides)
    return base


def _tool_context(
    *,
    parent_tools: dict[str, Any] | None = None,
    cancelled: bool = False,
    invocation_state: dict[str, Any] | None = None,
) -> ToolContext:
    """Build a ToolContext with a stub parent agent."""
    agent = SimpleNamespace(
        model=_FakeModel(),
        tool_registry=_FakeRegistry(parent_tools),
        _cancel_signal=_FakeCancelSignal(cancelled),
    )
    return ToolContext(
        tool_use={"name": "swarm", "toolUseId": "tid", "input": {}},
        agent=agent,
        invocation_state=invocation_state or {},
    )


class _FakeSwarmNode:
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id

    def __str__(self) -> str:
        return self.node_id


def _build_result(
    node_history: list[str],
    final_text: str = "",
    status: Status = Status.COMPLETED,
    execution_time: int = 42,
) -> SwarmResult:
    """Build a SwarmResult that _map_result can read back."""
    from strands.agent import AgentResult

    swarm_nodes = [_FakeSwarmNode(nid) for nid in node_history]

    results: dict[str, NodeResult] = {}
    if node_history:
        agent_result = AgentResult(
            message={"role": "assistant", "content": [{"text": final_text}]},
            stop_reason="end_turn",
            state={},
            metrics=MagicMock(
                accumulated_usage=Usage(inputTokens=5, outputTokens=7, totalTokens=12),
                accumulated_metrics=Metrics(latencyMs=execution_time),
            ),
        )
        results[node_history[-1]] = NodeResult(
            result=agent_result,
            status=Status.COMPLETED,
            execution_count=1,
            accumulated_usage=Usage(inputTokens=5, outputTokens=7, totalTokens=12),
            accumulated_metrics=Metrics(latencyMs=execution_time),
        )

    return SwarmResult(
        status=status,
        results=results,
        accumulated_usage=Usage(inputTokens=5, outputTokens=7, totalTokens=12),
        accumulated_metrics=Metrics(latencyMs=execution_time),
        execution_count=len(node_history),
        execution_time=execution_time,
        node_history=swarm_nodes,
    )


class TestSpecValidation:
    """Guards on the model-controlled `agents` payload."""

    @pytest.mark.asyncio
    async def test_rejects_non_list_agents(self):
        with pytest.raises(ValueError, match="agents must be a list"):
            await swarm(agents="not-a-list", initial_input="go", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_rejects_empty_agents(self):
        with pytest.raises(ValueError, match="agents list is empty"):
            await swarm(agents=[], initial_input="go", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_rejects_too_many_agents(self):
        specs = [_spec(name=f"a{i}") for i in range(6)]
        with pytest.raises(ValueError, match="too many agents"):
            await swarm(agents=specs, initial_input="go", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_respects_configurable_max_agents(self):
        tool = make_swarm(max_agents=2)
        specs = [_spec("a"), _spec("b"), _spec("c")]
        with pytest.raises(ValueError, match="too many agents"):
            await tool(agents=specs, initial_input="go", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_rejects_non_dict_spec(self):
        with pytest.raises(ValueError, match="must be an object"):
            await swarm(agents=["oops"], initial_input="go", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_rejects_missing_name(self):
        with pytest.raises(ValueError, match="name must be a non-empty string"):
            await swarm(agents=[{"system_prompt": "hi", "tools": []}], initial_input="go", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_rejects_empty_name(self):
        with pytest.raises(ValueError, match="name must be a non-empty string"):
            await swarm(
                agents=[{"name": "   ", "system_prompt": "sp", "tools": []}],
                initial_input="go",
                tool_context=_tool_context(),
            )

    @pytest.mark.asyncio
    async def test_rejects_name_exceeding_char_cap(self):
        # 65 chars — one past the cap. The shared dialect regex caps to 64.
        overlong = "a" * 65
        with pytest.raises(ValueError, match="exceeds 64-char cap"):
            await swarm(agents=[_spec(name=overlong)], initial_input="go", tool_context=_tool_context())

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_name",
        ["1leading-digit", "has space", "has-dash", "has.dot", "has$symbol"],
    )
    async def test_rejects_name_not_matching_regex(self, bad_name):
        with pytest.raises(ValueError, match="must match"):
            await swarm(agents=[_spec(name=bad_name)], initial_input="go", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_rejects_duplicate_names(self):
        specs = [_spec("a"), _spec("a")]
        with pytest.raises(ValueError, match="duplicate"):
            await swarm(agents=specs, initial_input="go", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_rejects_missing_system_prompt(self):
        # `system_prompt` is required per the shared dialect.
        specs = [{"name": "a", "tools": []}]
        with pytest.raises(ValueError, match="system_prompt is required"):
            await swarm(agents=specs, initial_input="go", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_rejects_non_string_system_prompt(self):
        specs = [{"name": "a", "system_prompt": 123, "tools": []}]
        with pytest.raises(ValueError, match="system_prompt is required and must be a string"):
            await swarm(agents=specs, initial_input="go", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_rejects_oversized_system_prompt(self):
        oversized = "x" * (8 * 1024 + 1)
        specs = [_spec("a", system_prompt=oversized)]
        with pytest.raises(ValueError, match="system_prompt exceeds size cap"):
            await swarm(agents=specs, initial_input="go", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_rejects_missing_tools(self):
        # `tools` is required per the shared dialect (may be empty list).
        specs = [{"name": "a", "system_prompt": "sp"}]
        with pytest.raises(ValueError, match="tools is required"):
            await swarm(agents=specs, initial_input="go", tool_context=_tool_context())

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tools_value",
        ["bash", ["bash", 42]],
        ids=["non_list", "non_string_entry"],
    )
    async def test_rejects_invalid_tools(self, tools_value):
        specs = [{"name": "a", "system_prompt": "sp", "tools": tools_value}]
        with pytest.raises(ValueError, match="tools is required and must be a list of strings"):
            await swarm(agents=specs, initial_input="go", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_rejects_tools_exceeding_cap(self):
        # Cap is 64 per the shared dialect. 65 entries should fail.
        specs = [_spec("a", tools=[f"t{i}" for i in range(65)])]
        with pytest.raises(ValueError, match="tools exceeds cap"):
            await swarm(agents=specs, initial_input="go", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_rejects_unknown_spec_fields(self):
        # Explicitly guard against the model inventing a `model` or `hooks` field.
        specs = [_spec("a", model="gpt-9000")]
        with pytest.raises(ValueError, match="unsupported fields"):
            await swarm(agents=specs, initial_input="go", tool_context=_tool_context())

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "initial_input",
        ["", 123],
        ids=["empty", "non_string"],
    )
    async def test_rejects_invalid_initial_input(self, initial_input):
        specs = [_spec("a")]
        with pytest.raises(ValueError, match="initial_input must be a non-empty string"):
            await swarm(agents=specs, initial_input=initial_input, tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_rejects_oversized_initial_input(self):
        oversized = "x" * (32 * 1024 + 1)
        with pytest.raises(ValueError, match="initial_input exceeds size cap"):
            await swarm(agents=[_spec("a")], initial_input=oversized, tool_context=_tool_context())


class TestToolAllowlist:
    """The tool allowlist is the primary security control."""

    @pytest.mark.asyncio
    async def test_rejects_unknown_tool(self):
        specs = [_spec("a", tools=["bash"])]
        with pytest.raises(ValueError, match="requested unknown tool 'bash'"):
            await swarm(agents=specs, initial_input="go", tool_context=_tool_context(parent_tools={}))

    @pytest.mark.asyncio
    async def test_rejects_tool_not_in_parent_registry(self):
        specs = [_spec("a", tools=["evil"])]
        parent_tools = {"safe": _FakeTool("safe")}
        with pytest.raises(ValueError, match="unknown tool 'evil'"):
            await swarm(
                agents=specs,
                initial_input="go",
                tool_context=_tool_context(parent_tools=parent_tools),
            )

    @pytest.mark.asyncio
    async def test_rejects_wildcard_style_names(self):
        # Wildcards are literal strings — they should be treated as unknown tool names,
        # not expanded.
        specs = [_spec("a", tools=["*"])]
        with pytest.raises(ValueError, match="unknown tool"):
            await swarm(
                agents=specs,
                initial_input="go",
                tool_context=_tool_context(parent_tools={"safe": _FakeTool("safe")}),
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "multiagent_tool",
        ["use_agent", "swarm", "graph", "a2a_client"],
    )
    async def test_rejects_multiagent_tool_names(self, multiagent_tool):
        # Defense-in-depth: even if the parent registered a multi-agent tool
        # under one of these reserved names, the child spec may not list it —
        # otherwise a compromised model could bypass the shared depth counter
        # by having a child re-invoke `swarm`/`graph`/etc.
        specs = [_spec("a", tools=[multiagent_tool])]
        parent_tools = {multiagent_tool: _FakeTool(multiagent_tool)}
        with pytest.raises(ValueError, match="multi-agent tool"):
            await swarm(
                agents=specs,
                initial_input="go",
                tool_context=_tool_context(parent_tools=parent_tools),
            )


class TestEntryAgent:
    @pytest.mark.asyncio
    async def test_rejects_entry_agent_not_in_list(self):
        specs = [_spec("a"), _spec("b")]
        with patch.object(_swarm_module, "Swarm"):
            with pytest.raises(ValueError, match="entry_agent 'c' not in agents list"):
                await swarm(
                    agents=specs,
                    initial_input="go",
                    tool_context=_tool_context(),
                    entry_agent="c",
                )


class TestRecursionDepth:
    """The shared multi-agent depth counter caps runaway delegation chains."""

    @pytest.mark.asyncio
    async def test_refuses_at_cap(self):
        ctx = _tool_context(invocation_state={"multiagent_depth": 3})
        with pytest.raises(MultiagentDepthExceeded, match="recursion depth cap"):
            await swarm(agents=[_spec("a")], initial_input="go", tool_context=ctx)

    @pytest.mark.asyncio
    async def test_forwards_incremented_depth_to_child_swarm(self):
        expected = _build_result(node_history=["a"], final_text="ok")
        fake_instance = MagicMock()
        fake_instance.invoke_async = AsyncMock(return_value=expected)
        fake_instance.add_hook = MagicMock()

        with patch.object(_swarm_module, "Swarm", return_value=fake_instance):
            await swarm(
                agents=[_spec("a")],
                initial_input="go",
                tool_context=_tool_context(invocation_state={"multiagent_depth": 1}),
            )

        fake_instance.invoke_async.assert_awaited_once_with("go", invocation_state={"multiagent_depth": 2})

    @pytest.mark.asyncio
    async def test_treats_garbage_depth_as_zero(self):
        expected = _build_result(node_history=["a"], final_text="ok")
        fake_instance = MagicMock()
        fake_instance.invoke_async = AsyncMock(return_value=expected)
        fake_instance.add_hook = MagicMock()

        with patch.object(_swarm_module, "Swarm", return_value=fake_instance):
            await swarm(
                agents=[_spec("a")],
                initial_input="go",
                tool_context=_tool_context(invocation_state={"multiagent_depth": "not-an-int"}),
            )

        fake_instance.invoke_async.assert_awaited_once_with("go", invocation_state={"multiagent_depth": 1})


class TestHappyPath:
    """End-to-end flow with the Swarm class patched to a controllable stub."""

    @pytest.mark.asyncio
    async def test_returns_normalized_result(self):
        expected = _build_result(
            node_history=["researcher", "writer"],
            final_text="Here is your report.",
            status=Status.COMPLETED,
            execution_time=250,
        )
        fake_instance = MagicMock()
        fake_instance.invoke_async = AsyncMock(return_value=expected)
        fake_instance.add_hook = MagicMock()

        with patch.object(_swarm_module, "Swarm", return_value=fake_instance) as fake_swarm_cls:
            result = await swarm(
                agents=[
                    _spec("researcher", system_prompt="You research."),
                    _spec("writer", system_prompt="You write."),
                ],
                initial_input="Write me a report on octopi.",
                tool_context=_tool_context(),
            )

        fake_swarm_cls.assert_called_once()
        _args, kwargs = fake_swarm_cls.call_args
        assert kwargs["max_iterations"] == 10
        assert kwargs["max_handoffs"] == 10
        assert kwargs["execution_timeout"] == 300.0
        assert kwargs["node_timeout"] == 120.0
        fake_instance.add_hook.assert_called_once()
        fake_instance.invoke_async.assert_awaited_once_with(
            "Write me a report on octopi.", invocation_state={"multiagent_depth": 1}
        )

        # Status translated from SDK's `completed` into the shared dialect's `success`.
        assert result == {
            "status": "success",
            "output": "Here is your report.",
            "node_history": ["researcher", "writer"],
            "execution_count": 2,
            "execution_time_ms": 250,
            "usage": {"inputTokens": 5, "outputTokens": 7, "totalTokens": 12},
        }

    @pytest.mark.asyncio
    async def test_maps_failed_result_with_empty_output(self):
        expected = _build_result(node_history=[], status=Status.FAILED)
        fake_instance = MagicMock()
        fake_instance.invoke_async = AsyncMock(return_value=expected)
        fake_instance.add_hook = MagicMock()

        with patch.object(_swarm_module, "Swarm", return_value=fake_instance):
            result = await swarm(
                agents=[_spec("loner")],
                initial_input="do a thing",
                tool_context=_tool_context(),
            )

        # SDK's `failed` maps to shared dialect's `error`.
        assert result["status"] == "error"
        assert result["output"] == ""
        assert result["node_history"] == []

    @pytest.mark.asyncio
    async def test_maps_interrupted_result_to_cancelled(self):
        # SDK's `interrupted` maps to the shared dialect's `cancelled`. This
        # covers a genuine user interrupt reaching the wrapper — parent
        # cancellation takes a different path (covered end-to-end in
        # TestCancellation.test_hook_fired_maps_result_to_cancelled).
        expected = _build_result(node_history=[], status=Status.INTERRUPTED)
        fake_instance = MagicMock()
        fake_instance.invoke_async = AsyncMock(return_value=expected)
        fake_instance.add_hook = MagicMock()

        with patch.object(_swarm_module, "Swarm", return_value=fake_instance):
            result = await swarm(
                agents=[_spec("a")],
                initial_input="go",
                tool_context=_tool_context(),
            )

        assert result["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_forwards_entry_agent(self):
        expected = _build_result(node_history=["b"], final_text="from b")
        fake_instance = MagicMock()
        fake_instance.invoke_async = AsyncMock(return_value=expected)
        fake_instance.add_hook = MagicMock()

        with patch.object(_swarm_module, "Swarm", return_value=fake_instance) as fake_swarm_cls:
            await swarm(
                agents=[_spec("a"), _spec("b")],
                initial_input="go",
                tool_context=_tool_context(),
                entry_agent="b",
            )
        _args, kwargs = fake_swarm_cls.call_args
        entry = kwargs["entry_point"]
        assert entry is not None
        assert entry.name == "b"


class TestCancellation:
    """Parent's cancel signal propagates to child swarm via BeforeNodeCallEvent."""

    def test_hook_cancels_when_parent_signal_set(self):
        from strands.hooks import BeforeNodeCallEvent
        from strands.vended_tools.swarm.swarm import _ParentCancelHook

        parent = SimpleNamespace(_cancel_signal=_FakeCancelSignal(True))
        hook = _ParentCancelHook(parent)

        event = MagicMock(spec=BeforeNodeCallEvent)
        event.cancel_node = False
        hook.on_before_node_call(event)
        assert event.cancel_node == "cancelled by parent agent"
        assert hook.fired is True

    def test_hook_no_op_when_parent_signal_clear(self):
        from strands.hooks import BeforeNodeCallEvent
        from strands.vended_tools.swarm.swarm import _ParentCancelHook

        parent = SimpleNamespace(_cancel_signal=_FakeCancelSignal(False))
        hook = _ParentCancelHook(parent)

        event = MagicMock(spec=BeforeNodeCallEvent)
        event.cancel_node = False
        hook.on_before_node_call(event)
        assert event.cancel_node is False
        assert hook.fired is False

    def test_hook_construction_fails_loudly_without_cancel_signal(self):
        from strands.vended_tools.swarm.swarm import _ParentCancelHook

        # Guard against the SDK renaming/removing the private attribute — a silent
        # fallback would mask cancellation being dropped on the floor.
        parent = SimpleNamespace()
        with pytest.raises(AttributeError, match="_cancel_signal"):
            _ParentCancelHook(parent)

    @pytest.mark.asyncio
    async def test_hook_fired_maps_result_to_cancelled(self):
        # End-to-end: when the parent-cancel hook fires, the SDK's Swarm sets
        # completion_status to FAILED (not INTERRUPTED). The wrapper detects the
        # hook fired and overrides the resulting status to `cancelled` so
        # downstream models can distinguish parent cancellation from a real
        # failure. This is the glue between _ParentCancelHook and _map_status
        # that the "interrupted -> cancelled" mapping test alone does not cover.
        from strands.hooks import BeforeNodeCallEvent

        # Fake Swarm that, on invoke_async, drives the registered hook and then
        # returns a FAILED result — the same shape the real Swarm produces when
        # a hook sets cancel_node.
        parent_agent = SimpleNamespace(
            model=_FakeModel(),
            tool_registry=_FakeRegistry(),
            _cancel_signal=_FakeCancelSignal(True),
        )
        registered_hooks: list[Any] = []

        class _FakeSwarm:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                pass

            def add_hook(self, callback: Any, _event_type: Any) -> None:
                registered_hooks.append(callback)

            async def invoke_async(self, _input: str, invocation_state: Any = None) -> Any:
                # Drive the hook with a fake event so `fired` flips to True,
                # then return the FAILED result the SDK would produce.
                event = MagicMock(spec=BeforeNodeCallEvent)
                event.cancel_node = False
                for cb in registered_hooks:
                    cb(event)
                return _build_result(node_history=[], status=Status.FAILED)

        ctx = ToolContext(
            tool_use={"name": "swarm", "toolUseId": "tid", "input": {}},
            agent=parent_agent,
            invocation_state={},
        )
        with patch.object(_swarm_module, "Swarm", _FakeSwarm):
            result = await swarm(agents=[_spec("a")], initial_input="go", tool_context=ctx)

        assert result["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_parent_invocation_state_is_preserved(self):
        # Tracing, telemetry, and per-run keys attached to the parent's
        # invocation_state must flow through to the child; only
        # multiagent_depth is overridden.
        expected = _build_result(node_history=["a"], final_text="ok")
        fake_instance = MagicMock()
        fake_instance.invoke_async = AsyncMock(return_value=expected)
        fake_instance.add_hook = MagicMock()

        parent_state = {"trace_id": "abc", "user_id": "u1", "multiagent_depth": 1}
        with patch.object(_swarm_module, "Swarm", return_value=fake_instance):
            await swarm(
                agents=[_spec("a")],
                initial_input="go",
                tool_context=_tool_context(invocation_state=parent_state),
            )

        fake_instance.invoke_async.assert_awaited_once_with(
            "go", invocation_state={"trace_id": "abc", "user_id": "u1", "multiagent_depth": 2}
        )
