import asyncio
import importlib
from typing import Any, cast

import pytest

from strands import Agent, tool
from strands.experimental.tools import make_use_agent, use_agent
from strands.hooks import BeforeToolCallEvent, BeforeToolsEvent
from strands.interventions import Confirm, InterventionHandler, Proceed
from strands.types._events import ToolResultEvent
from strands.types.content import Message
from tests.fixtures.mocked_model_provider import MockedModelProvider


def _text_turn(text: str) -> Message:
    return {"role": "assistant", "content": [{"text": text}]}


def _tool_turn(name: str, tool_use_id: str, tool_input: dict[str, Any] | None = None) -> Message:
    return {
        "role": "assistant",
        "content": [{"toolUse": {"name": name, "toolUseId": tool_use_id, "input": tool_input or {}}}],
    }


async def _run_tool(
    agent: Agent,
    use_agent_tool: Any,
    tool_input: Any,
) -> ToolResultEvent:
    return await anext(
        use_agent_tool.stream(
            {"name": "use_agent", "toolUseId": "use-agent-call", "input": tool_input},
            {"agent": agent},
        )
    )


def _result_json(event: ToolResultEvent) -> dict[str, Any]:
    return cast(dict[str, Any], event.tool_result["content"][0].get("json"))


class _ConfirmDangerous(InterventionHandler):
    @property
    def name(self) -> str:
        return "confirm-dangerous"

    def before_tool_call(self, event: BeforeToolCallEvent, **kwargs: Any) -> Proceed | Confirm:
        if event.tool_use["name"].startswith("dangerous"):
            return Confirm(prompt="approve?")
        return Proceed()


def test_tool_spec_exposes_narrow_input_and_bounded_options():
    spec = make_use_agent().tool_spec
    schema = spec["inputSchema"]["json"]

    assert use_agent.tool_name == spec["name"] == "use_agent"
    assert set(schema["properties"]) == {"task", "instructions", "tools"}
    assert schema["required"] == ["task"]
    assert schema["additionalProperties"] is False
    with pytest.raises(TypeError, match="limits.turns must be an integer between 1 and 50"):
        make_use_agent(limits={"turns": 51})


@pytest.mark.asyncio
async def test_grants_only_named_parent_tools_and_preserves_inherited_pre_tool_policy():
    calls: list[str] = []

    @tool
    def extra() -> str:
        calls.append("extra")
        return "extra"

    @tool
    def blocked() -> str:
        calls.append("blocked")
        return "blocked"

    omitted_tool = make_use_agent()
    omitted_parent = Agent(
        model=MockedModelProvider([_tool_turn("extra", "extra-call"), _text_turn("handled")]),
        tools=[extra, omitted_tool],
        callback_handler=None,
    )
    omitted = await _run_tool(omitted_parent, omitted_tool, {"task": "work"})

    before_tools = 0
    policy_order: list[str] = []

    def record_tools(event: BeforeToolsEvent) -> None:
        nonlocal before_tools
        before_tools += 1
        assert event.agent.system_prompt == "Research carefully."

    def record_ancestor(event: BeforeToolCallEvent) -> None:
        policy_order.append("ancestor")

    ancestor = Agent(model=MockedModelProvider([_text_turn("unused")]), callback_handler=None)
    ancestor.add_hook(record_ancestor, BeforeToolCallEvent, order=100)
    governed_tool = make_use_agent()
    governed_parent = Agent(
        model=MockedModelProvider([_tool_turn("extra", "extra-call"), _text_turn("handled")]),
        tools=[extra, blocked, governed_tool],
        callback_handler=None,
    )
    governed_parent.hooks._inherit_callbacks_from(ancestor.hooks, [BeforeToolsEvent, BeforeToolCallEvent])
    governed_parent.add_hook(record_tools, BeforeToolsEvent)

    def replace_tool(event: BeforeToolCallEvent) -> None:
        policy_order.append("parent")
        event.selected_tool = blocked

    governed_parent.add_hook(replace_tool, BeforeToolCallEvent)
    governed = await _run_tool(
        governed_parent,
        governed_tool,
        {"task": "work", "instructions": "Research carefully.", "tools": ["extra"]},
    )

    assert calls == []
    assert before_tools == 1
    assert policy_order == ["parent", "ancestor"]
    assert _result_json(omitted)["output"] == "handled\n"
    assert governed.tool_result["status"] == "success"


@pytest.mark.asyncio
async def test_rejects_grants_or_models_that_bypass_the_governed_tool_registry(monkeypatch: pytest.MonkeyPatch):
    registered = make_use_agent()
    parent = Agent(
        model=MockedModelProvider([_text_turn("unused")]),
        tools=[registered],
        callback_handler=None,
    )

    missing = await _run_tool(parent, registered, {"task": "work", "tools": ["missing"]})
    different_tool = make_use_agent()
    different = await _run_tool(parent, different_tool, {"task": "work", "tools": ["use_agent"]})

    monkeypatch.setattr(parent.model, "get_config", lambda: {"params": {"tools": [{"type": "web_search"}]}})
    native = await _run_tool(parent, registered, {"task": "work"})

    recursive_tool = make_use_agent(limits={"depth": 1})
    recursive_parent = Agent(
        model=MockedModelProvider(
            [
                _tool_turn("use_agent", "nested-call", {"task": "nested"}),
                _text_turn("done"),
            ]
        ),
        tools=[recursive_tool],
        callback_handler=None,
    )
    recursive = await _run_tool(recursive_parent, recursive_tool, {"task": "work", "tools": ["use_agent"]})

    assert _result_json(missing)["error"] == "Tool 'missing' was not found on the parent agent"
    assert _result_json(different)["error"] == "A child can receive only the currently executing use_agent tool"
    assert "provider-native model tools" in _result_json(native)["error"]
    assert _result_json(recursive)["output"] == "done\n"


@pytest.mark.asyncio
async def test_scopes_repeated_interrupts_and_resumes_the_same_child_state():
    calls = 0

    @tool
    def dangerous() -> str:
        nonlocal calls
        calls += 1
        return "deleted"

    model = MockedModelProvider(
        [
            _tool_turn("use_agent", "use-agent-call", {"task": "delete", "tools": ["dangerous"]}),
            _tool_turn("dangerous", "dangerous-call"),
            _tool_turn("dangerous", "dangerous-call"),
            _text_turn("child done"),
            _text_turn("outer done"),
        ]
    )
    use_agent_tool = make_use_agent()
    parent = Agent(
        model=model,
        tools=[dangerous, use_agent_tool],
        interventions=[_ConfirmDangerous()],
        callback_handler=None,
    )

    first = await parent.invoke_async("go")
    assert first.interrupts
    first_interrupt = first.interrupts[0]

    restored_tool = make_use_agent()
    restored_parent = Agent(
        model=MockedModelProvider([_text_turn("unused")]),
        tools=[dangerous, restored_tool],
        interventions=[_ConfirmDangerous()],
        callback_handler=None,
    )
    restored_parent.load_snapshot(parent.take_snapshot(preset="session"))
    restored = await _run_tool(restored_parent, restored_tool, {"task": "delete", "tools": ["dangerous"]})

    second = await parent.invoke_async([{"interruptResponse": {"interruptId": first_interrupt.id, "response": "yes"}}])
    assert second.interrupts
    second_interrupt = second.interrupts[0]
    completed = await parent.invoke_async(
        [{"interruptResponse": {"interruptId": second_interrupt.id, "response": "yes"}}]
    )

    assert second_interrupt.id != first_interrupt.id
    assert str(completed).rstrip() == "outer done"
    assert calls == 2
    assert _result_json(restored)["error"] == (
        "use_agent cannot resume an interrupted child after the parent or tool instance was restored"
    )


@pytest.mark.asyncio
async def test_preserves_turn_and_token_budgets_across_an_interrupt():
    for limits, usages, stop_reason in [
        ({"turns": 1}, None, "limit_turns"),
        (
            {"total_tokens": 1},
            [
                {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                {"inputTokens": 5, "outputTokens": 5, "totalTokens": 10},
                {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
            ],
            "limit_total_tokens",
        ),
    ]:
        calls = 0

        @tool
        def dangerous() -> str:
            nonlocal calls
            calls += 1
            return "deleted"

        model = MockedModelProvider(
            [
                _tool_turn("use_agent", "use-agent-call", {"task": "delete", "tools": ["dangerous"]}),
                _tool_turn("dangerous", "dangerous-call"),
                _text_turn("budget exhausted"),
            ],
            usages=usages,
        )
        use_agent_tool = make_use_agent(limits=limits)
        parent = Agent(
            model=model,
            tools=[dangerous, use_agent_tool],
            interventions=[_ConfirmDangerous()],
            callback_handler=None,
        )

        interrupted = await parent.invoke_async("go")
        assert interrupted.interrupts
        await parent.invoke_async(
            [{"interruptResponse": {"interruptId": interrupted.interrupts[0].id, "response": "yes"}}]
        )
        tool_result = next(
            content["toolResult"]
            for message in parent.messages
            for content in message["content"]
            if "toolResult" in content and content["toolResult"]["toolUseId"] == "use-agent-call"
        )

        assert calls == 0
        result_json = cast(dict[str, Any], tool_result["content"][0].get("json"))
        assert result_json == {"status": "failed", "error": f"use_agent child stopped before completion: {stop_reason}"}


@pytest.mark.asyncio
async def test_returns_promptly_on_parent_cancellation_and_timeout(monkeypatch: pytest.MonkeyPatch):
    async def wait_forever(*args: Any, **kwargs: Any) -> Any:
        await asyncio.Event().wait()

    monkeypatch.setattr(Agent, "invoke_async", wait_forever)
    cancelled_parent = Agent(model=MockedModelProvider([_text_turn("unused")]), callback_handler=None)
    task = asyncio.create_task(_run_tool(cancelled_parent, make_use_agent(), {"task": "work"}))
    await asyncio.sleep(0)
    cancelled_parent.cancel()
    cancelled = await asyncio.wait_for(task, timeout=1)

    ticks = iter([0.0, 2.0])
    use_agent_module = importlib.import_module("strands.experimental.tools.use_agent.use_agent")
    monkeypatch.setattr(use_agent_module, "monotonic", lambda: next(ticks))
    timed_parent = Agent(model=MockedModelProvider([_text_turn("unused")]), callback_handler=None)
    timed = await _run_tool(timed_parent, make_use_agent(limits={"timeout_seconds": 1}), {"task": "work"})

    assert _result_json(cancelled)["status"] == "cancelled"
    assert _result_json(timed)["error"] == "use_agent child exceeded its execution timeout"
