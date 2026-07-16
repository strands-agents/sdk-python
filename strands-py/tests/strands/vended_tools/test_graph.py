"""Tests for the graph vended tool.

The tool is a thin shim over :class:`~strands.multiagent.graph.GraphBuilder`.
Security-focused tests check validation at the tool boundary (bad topology,
oversized inputs, unknown tools, wildcards, cycles); happy-path tests exercise
a small end-to-end graph using ``MockedModelProvider`` sub-agents.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from strands import Agent
from strands.types.tools import ToolContext
from strands.vended_tools.graph import graph
from strands.vended_tools.graph.graph import MultiagentDepthExceeded
from strands.vended_tools.graph.types import (
    MAX_INITIAL_INPUT_LENGTH,
    MAX_NODES,
    MAX_SYSTEM_PROMPT_LENGTH,
    MAX_TOOLS_PER_NODE,
)
from tests.fixtures.mocked_model_provider import MockedModelProvider


def _text_response(text: str) -> dict:
    return {"role": "assistant", "content": [{"text": text}]}


def _make_parent(responses_per_node: int = 1, extra_tools: list | None = None) -> Agent:
    """Build a parent agent with a mocked model.

    The parent isn't actually invoked; the graph tool reuses its ``model``
    attribute for the sub-agents built inside the graph. Each sub-agent
    consumes exactly one response from the shared mocked provider, so we
    provide enough entries for every node we expect to execute.
    """
    responses = [_text_response(f"reply {i}") for i in range(responses_per_node)]
    model = MockedModelProvider(agent_responses=responses)
    return Agent(model=model, tools=extra_tools or [])


def _tool_context(agent: Agent) -> ToolContext:
    return ToolContext(
        tool_use={"name": "graph", "toolUseId": "test-id", "input": {}},
        agent=agent,
        invocation_state={},
    )


@pytest.mark.asyncio
async def test_rejects_cycle():
    """A -> B -> A must be rejected before any sub-agent is built."""
    agent = _make_parent()
    with pytest.raises(ValueError, match="cycle"):
        await graph(
            nodes=[{"id": "a"}, {"id": "b"}],
            edges=[
                {"from_id": "a", "to_id": "b"},
                {"from_id": "b", "to_id": "a"},
            ],
            initial_input="hi",
            tool_context=_tool_context(agent),
        )


@pytest.mark.asyncio
async def test_rejects_self_loop():
    agent = _make_parent()
    with pytest.raises(ValueError, match="Self-loop"):
        await graph(
            nodes=[{"id": "a"}],
            edges=[{"from_id": "a", "to_id": "a"}],
            initial_input="hi",
            tool_context=_tool_context(agent),
        )


@pytest.mark.asyncio
async def test_rejects_unknown_tool_in_node_allowlist():
    """A node cannot request a tool the parent's registry does not expose."""
    agent = _make_parent()
    with pytest.raises(ValueError, match="not registered on the parent"):
        await graph(
            nodes=[{"id": "a", "tools": ["definitely_not_a_real_tool"]}],
            edges=[],
            initial_input="hi",
            tool_context=_tool_context(agent),
        )


@pytest.mark.asyncio
async def test_rejects_multiagent_tool_in_node_allowlist():
    """Spec: multi-agent tools (`use_agent`, `swarm`, `graph`, `a2a_client`)
    may not appear in a child node's allow-list. Defense-in-depth on top of
    the shared depth cap."""
    # Register a real graph tool on the parent so its name resolves in the
    # registry — otherwise the "unknown tool" rejection would fire first and
    # mask the multi-agent-name rejection this test is exercising.
    agent = _make_parent(extra_tools=[graph])

    with pytest.raises(ValueError, match="multi-agent tool"):
        await graph(
            nodes=[{"id": "a", "tools": ["graph"]}],
            edges=[],
            initial_input="hi",
            tool_context=_tool_context(agent),
        )


@pytest.mark.asyncio
async def test_rejects_wildcard_in_tool_allowlist():
    """The allowlist is names-only; wildcards are refused so 'all tools' can't leak in."""
    agent = _make_parent()
    with pytest.raises(ValueError, match="wildcard"):
        await graph(
            nodes=[{"id": "a", "tools": ["*"]}],
            edges=[],
            initial_input="hi",
            tool_context=_tool_context(agent),
        )


@pytest.mark.asyncio
async def test_rejects_oversized_tool_allowlist():
    """Spec: per-node ``tools`` list is capped so a malicious model cannot flood
    the resolver with a huge repeated list."""
    agent = _make_parent()
    over = ["not_a_tool"] * (MAX_TOOLS_PER_NODE + 1)
    with pytest.raises(ValueError, match="tools"):
        await graph(
            nodes=[{"id": "a", "tools": over}],
            edges=[],
            initial_input="hi",
            tool_context=_tool_context(agent),
        )


@pytest.mark.asyncio
async def test_rejects_too_many_nodes():
    agent = _make_parent()
    over = MAX_NODES + 1
    nodes = [{"id": f"n{i}"} for i in range(over)]
    with pytest.raises(ValueError, match="Too many nodes"):
        await graph(
            nodes=nodes,
            edges=[],
            initial_input="hi",
            tool_context=_tool_context(agent),
        )


@pytest.mark.asyncio
async def test_rejects_duplicate_node_ids():
    agent = _make_parent()
    with pytest.raises(ValueError, match="Duplicate node id"):
        await graph(
            nodes=[{"id": "a"}, {"id": "a"}],
            edges=[],
            initial_input="hi",
            tool_context=_tool_context(agent),
        )


@pytest.mark.asyncio
async def test_rejects_edge_referencing_unknown_node():
    agent = _make_parent()
    with pytest.raises(ValueError, match="unknown"):
        await graph(
            nodes=[{"id": "a"}],
            edges=[{"from_id": "a", "to_id": "ghost"}],
            initial_input="hi",
            tool_context=_tool_context(agent),
        )


@pytest.mark.asyncio
async def test_rejects_oversized_initial_input():
    agent = _make_parent()
    with pytest.raises(ValueError, match="initial_input"):
        await graph(
            nodes=[{"id": "a"}],
            edges=[],
            initial_input="x" * (MAX_INITIAL_INPUT_LENGTH + 1),
            tool_context=_tool_context(agent),
        )


@pytest.mark.asyncio
async def test_rejects_oversized_system_prompt():
    agent = _make_parent()
    with pytest.raises(ValueError, match="system_prompt"):
        await graph(
            nodes=[{"id": "a", "system_prompt": "x" * (MAX_SYSTEM_PROMPT_LENGTH + 1)}],
            edges=[],
            initial_input="hi",
            tool_context=_tool_context(agent),
        )


@pytest.mark.asyncio
async def test_rejects_bad_node_id_characters():
    agent = _make_parent()
    with pytest.raises(ValueError, match="characters outside"):
        await graph(
            nodes=[{"id": "bad id!"}],
            edges=[],
            initial_input="hi",
            tool_context=_tool_context(agent),
        )


@pytest.mark.asyncio
async def test_happy_path_linear_chain():
    """a -> b runs both nodes in order and returns per-node output."""
    # Two nodes, one response each.
    responses = [_text_response("first"), _text_response("second")]
    model = MockedModelProvider(agent_responses=responses)
    agent = Agent(model=model)

    result = await graph(
        nodes=[{"id": "a"}, {"id": "b"}],
        edges=[{"from_id": "a", "to_id": "b"}],
        initial_input="start",
        tool_context=_tool_context(agent),
    )

    assert result["status"] == "completed"
    assert set(result["results"].keys()) == {"a", "b"}
    assert result["results"]["a"]["status"] == "completed"
    assert result["results"]["b"]["status"] == "completed"
    assert result["execution_order"] == ["a", "b"]
    # ``output`` is the terminal node's text — ``b`` is the sole leaf here.
    assert result["output"] == result["results"]["b"]["output"]


@pytest.mark.asyncio
async def test_happy_path_fan_out_fan_in():
    """a fans out to b and c; both feed into d."""
    responses = [_text_response(f"node{i}") for i in range(4)]
    model = MockedModelProvider(agent_responses=responses)
    agent = Agent(model=model)

    result = await graph(
        nodes=[{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}],
        edges=[
            {"from_id": "a", "to_id": "b"},
            {"from_id": "a", "to_id": "c"},
            {"from_id": "b", "to_id": "d"},
            {"from_id": "c", "to_id": "d"},
        ],
        initial_input="start",
        tool_context=_tool_context(agent),
    )

    assert result["status"] == "completed"
    assert set(result["results"].keys()) == {"a", "b", "c", "d"}
    # 'a' must be first, 'd' must be last; b and c can be in either order.
    order = result["execution_order"]
    assert order[0] == "a"
    assert order[-1] == "d"


@pytest.mark.asyncio
async def test_returns_cancelled_when_parent_already_cancelled():
    """If the parent agent is already cancelled, the tool returns a cancelled
    result (not an exception). Spec: cancellation never raises past the tool
    boundary — a cancelled result is a signal to the loop, not an error."""
    agent = _make_parent()
    agent._cancel_signal = threading.Event()
    agent._cancel_signal.set()

    result = await graph(
        nodes=[{"id": "a"}],
        edges=[],
        initial_input="hi",
        tool_context=_tool_context(agent),
    )
    assert result["status"] == "cancelled"
    assert result["output"] == ""
    assert result["execution_order"] == []
    assert result["results"] == {}


@pytest.mark.asyncio
async def test_before_node_call_hook_cancels_mid_flight():
    """A parent cancel fired *during* execution aborts the next node.

    Exercises the ``BeforeNodeCallEvent`` hook path — not the pre-flight check.
    A hook is registered on the parent agent that sets ``_cancel_signal`` the
    first time the hook fires on the graph itself; the tool must then still
    return ``{status: cancelled}`` from the hook-driven ``RuntimeError`` re-raise.

    This test would fail without the round-3 fix that catches
    ``RuntimeError("...graph-tool-parent-cancel...")`` and maps it to a cancelled
    result — the previous code let the exception propagate to the caller.
    """
    responses = [_text_response("first"), _text_response("second")]
    model = MockedModelProvider(agent_responses=responses)
    agent = Agent(model=model)
    # Cancel signal starts un-set — we want the hook to trip mid-flight.
    agent._cancel_signal = threading.Event()

    # Monkeypatch `_ParentCancelHook.on_before_node_call` to arm the signal
    # *after* the first call has passed through the original (which observes
    # signal-not-set and does nothing). The second call then sees the armed
    # signal and sets `event.cancel_node`, which the SDK re-raises as
    # `RuntimeError` — the round-3 fix must catch that and return
    # `{status: 'cancelled'}` instead of propagating.
    from strands.vended_tools.graph.graph import _ParentCancelHook

    original_on_before = _ParentCancelHook.on_before_node_call

    call_counter = {"count": 0}

    def cancel_after_first_call(self, event):  # type: ignore[no-untyped-def]
        # Call the original first so it can observe the current state, then
        # arm the signal so the *next* node's invocation trips it.
        original_on_before(self, event)
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            agent._cancel_signal.set()

    _ParentCancelHook.on_before_node_call = cancel_after_first_call  # type: ignore[assignment]
    try:
        result = await graph(
            nodes=[{"id": "a"}, {"id": "b"}],
            edges=[{"from_id": "a", "to_id": "b"}],
            initial_input="start",
            tool_context=_tool_context(agent),
        )
    finally:
        _ParentCancelHook.on_before_node_call = original_on_before  # type: ignore[assignment]

    # The tool caught the hook-driven RuntimeError and returned cancelled.
    assert result["status"] == "cancelled"
    # The hook was invoked at least twice — once for node A (which ran) and
    # once for node B (where the cancel_node fired). If it were only 1, the
    # cancellation was happening pre-flight, not mid-flight.
    assert call_counter["count"] >= 2, (
        f"hook only fired {call_counter['count']}x — the test isn't exercising the mid-flight branch"
    )


@pytest.mark.asyncio
async def test_refuses_when_depth_cap_reached():
    """A parent already at the depth cap must not construct a new graph frame."""
    agent = _make_parent()
    ctx = ToolContext(
        tool_use={"name": "graph", "toolUseId": "test-id", "input": {}},
        agent=agent,
        invocation_state={"multiagent_depth": 3},
    )
    with pytest.raises(MultiagentDepthExceeded):
        await graph(
            nodes=[{"id": "a"}],
            edges=[],
            initial_input="hi",
            tool_context=ctx,
        )


@pytest.mark.asyncio
async def test_passes_incremented_depth_to_child_graph(monkeypatch):
    """Child ``invocation_state`` receives ``multiagent_depth = parent + 1``.

    The value threaded into ``Graph.invoke_async`` is what stops sibling
    multi-agent tools (`use_agent`, `swarm`, `a2a_client`) from resetting the
    counter at each hop.
    """
    responses = [_text_response("only-node")]
    model = MockedModelProvider(agent_responses=responses)
    agent = Agent(model=model)

    parent_ctx = ToolContext(
        tool_use={"name": "graph", "toolUseId": "test-id", "input": {}},
        agent=agent,
        invocation_state={"multiagent_depth": 1},
    )

    from strands.multiagent.graph import Graph

    captured: dict[str, Any] = {}
    original_invoke = Graph.invoke_async

    async def spy_invoke_async(self, task, invocation_state=None, **kwargs):
        captured["invocation_state"] = invocation_state
        return await original_invoke(self, task, invocation_state, **kwargs)

    monkeypatch.setattr(Graph, "invoke_async", spy_invoke_async)

    await graph(
        nodes=[{"id": "a"}],
        edges=[],
        initial_input="hi",
        tool_context=parent_ctx,
    )

    assert captured["invocation_state"] is not None
    assert captured["invocation_state"].get("multiagent_depth") == 2


def test_schema_excludes_context():
    props = graph.tool_spec["inputSchema"]["json"]["properties"]
    assert "nodes" in props
    assert "edges" in props
    assert "initial_input" in props
    assert "tool_context" not in props
