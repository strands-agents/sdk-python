"""Tests for max_iterations cap on the Agent event loop.

Without a cap, a degenerate model that always emits the same tool_use can drive
the event loop into unbounded recursion: each cycle the model re-emits the same
tool call, the tool result is appended, and the loop continues forever. This
test suite verifies that Agent(max_iterations=N) causes the loop to terminate
after at most N tool-call cycles with a synthetic terminal message and a
warning log.
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import pytest

import strands
from strands import Agent
from strands.models import Model
from strands.types.content import Messages, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec


class LoopingStubModel(Model):
    """Model that always emits the same tool_use — never terminates on its own."""

    def __init__(self) -> None:
        self.call_count = 0

    def format_chunk(self, event: Any) -> StreamEvent:
        return event

    def format_request(self, messages, tool_specs=None, system_prompt=None):
        return None

    def get_config(self):
        return {}

    def update_config(self, **cfg):
        pass

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError("LoopingStubModel does not support structured output")
        yield  # pragma: no cover — tells the type checker this is a generator

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        invocation_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict, None]:
        self.call_count += 1
        yield {"messageStart": {"role": "assistant"}}
        yield {
            "contentBlockStart": {
                "start": {
                    "toolUse": {
                        "name": "get_weather",
                        "toolUseId": f"tu_{self.call_count}",
                    }
                }
            }
        }
        yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps({"city": "SF"})}}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "tool_use"}}


@strands.tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"sunny in {city}"


@pytest.mark.asyncio
async def test_agent_max_iterations_terminates_loop(caplog):
    """Agent configured with max_iterations=5 must halt a looping model at exactly 5 cycles.

    Asserts:
      (a) The loop terminates (does not hang).
      (b) The model was invoked exactly `max_iterations` times (pins off-by-one regressions).
      (c) A warning is emitted with the exact expected substring when the cap trips.
      (d) `result.stop_reason == "max_iterations"`.
      (e) The last message is a synthetic assistant message whose content mentions
          "max_iterations" and which has a `metadata` key (so downstream consumers
          that read metadata.usage / metadata.metrics don't KeyError).
    """
    caplog.set_level(logging.WARNING, logger="strands.event_loop.event_loop")

    model = LoopingStubModel()
    agent = Agent(model=model, tools=[get_weather], max_iterations=5)

    # Must terminate within a bounded wall-clock budget.
    result = await asyncio.wait_for(agent.invoke_async("weather in SF?"), timeout=30.0)

    # (b) cap enforced exactly — pins off-by-one regressions
    assert model.call_count == 5, (
        f"model invoked {model.call_count} times, expected exactly 5"
    )

    # (a) loop terminated with a result
    assert result is not None

    # (d) stop_reason reflects the cap
    assert result.stop_reason == "max_iterations"

    # (e) synthetic terminal assistant message with metadata populated
    last_message = agent.messages[-1]
    assert last_message["role"] == "assistant"
    flat_text = "".join(
        block.get("text", "") for block in last_message.get("content", [])
    )
    assert "max_iterations" in flat_text, (
        f"expected synthetic message to mention 'max_iterations', got: {flat_text!r}"
    )
    assert "metadata" in last_message, (
        "synthetic message missing 'metadata' key — consumers that read "
        "metadata.usage / metadata.metrics will KeyError"
    )
    # metadata must have the usage + metrics shape matching normal assistant messages
    assert "usage" in last_message["metadata"]
    assert "metrics" in last_message["metadata"]
    # metadata must carry the `synthetic: True` marker so downstream
    # token-budgeting / analytics can filter the halt message out of
    # per-call cost percentiles (it is NOT a real model call).
    assert last_message["metadata"].get("synthetic") is True, (
        "synthetic halt message missing metadata['synthetic']=True marker"
    )

    # (c) warning emitted with the exact substring we document
    warning_texts = [
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    ]
    assert any(
        "max_iterations cap reached" in msg for msg in warning_texts
    ), f"expected 'max_iterations cap reached' in warnings, got: {warning_texts}"


@pytest.mark.asyncio
async def test_agent_max_iterations_one_boundary():
    """max_iterations=1 must trip after exactly 1 cycle — no crash, synthetic message present."""
    model = LoopingStubModel()
    agent = Agent(model=model, tools=[get_weather], max_iterations=1)

    result = await asyncio.wait_for(agent.invoke_async("weather?"), timeout=10.0)

    assert model.call_count == 1
    assert result.stop_reason == "max_iterations"
    assert agent.messages[-1]["role"] == "assistant"
    assert "metadata" in agent.messages[-1]
    assert agent.messages[-1]["metadata"].get("synthetic") is True


@pytest.mark.parametrize(
    "bad_value",
    [
        0,
        -1,
        "5",
        2.5,
        True,  # bool is a subclass of int; must be rejected explicitly
        False,
    ],
)
def test_agent_max_iterations_validation_rejects(bad_value):
    """Agent constructor must reject non-positive-int max_iterations values."""
    with pytest.raises(ValueError):
        Agent(max_iterations=bad_value)


@pytest.mark.asyncio
async def test_agent_max_iterations_counter_resets_between_invocations():
    """Reusing the SAME invocation_state dict across calls must NOT leak the counter.

    Regression guard for: cycle counter stored on caller-supplied dict accumulating
    across independent agent invocations.
    """
    model = LoopingStubModel()
    agent = Agent(model=model, tools=[get_weather], max_iterations=3)

    shared_state: dict[str, Any] = {}

    # First call: cap should trip at 3.
    await asyncio.wait_for(
        agent.invoke_async("first", invocation_state=shared_state), timeout=10.0
    )
    first_count = model.call_count
    assert first_count == 3, f"first call: expected 3, got {first_count}"

    # Second call: if the counter leaked, we'd trip immediately (0 more calls).
    # If the counter resets correctly, we get another 3 cycles.
    await asyncio.wait_for(
        agent.invoke_async("second", invocation_state=shared_state), timeout=10.0
    )
    delta = model.call_count - first_count
    assert delta == 3, (
        f"second call: expected another 3 cycles (counter reset), got {delta}. "
        "Counter leaked across invocations via shared invocation_state dict."
    )


@pytest.mark.asyncio
async def test_agent_max_iterations_resets_on_context_overflow():
    """max_iterations should cap tool-call cycles, NOT model-retry cycles.

    When ContextWindowOverflowException fires and the conversation_manager
    successfully reduces context, the recursive retry should reset the
    cycle counter so the user's budget is restored (cap = max tool-call cycles,
    not total model invocations).
    """
    from strands.types.exceptions import ContextWindowOverflowException

    model = LoopingStubModel()
    agent = Agent(model=model, tools=[get_weather], max_iterations=4)

    real_event_loop_cycle = strands.event_loop.event_loop.event_loop_cycle
    overflow_fired = {"done": False}

    async def cycle_that_overflows_once(*args, **kwargs):
        """First invocation raises overflow AFTER bumping the counter, then
        subsequent invocations go through to the real cycle.
        """
        if not overflow_fired["done"]:
            overflow_fired["done"] = True
            inv_state = kwargs.get("invocation_state") or (args[1] if len(args) > 1 else {})
            # Simulate a cycle that got partway through: bump the counter (as
            # the real cycle does when it invokes the model) before raising.
            inv_state["event_loop_cycle_count"] = (
                inv_state.get("event_loop_cycle_count", 0) + 1
            )
            raise ContextWindowOverflowException("simulated overflow")
            yield  # unreachable; marks this as an async generator
        async for ev in real_event_loop_cycle(*args, **kwargs):
            yield ev

    # Stub reduce_context so we don't fight the sliding-window trimming logic
    # (which requires enough messages to trim from). The behavior under test
    # is the counter reset, not trim logic.
    agent.conversation_manager.reduce_context = lambda *a, **kw: None

    # Patch the symbol as imported into agent.py.
    import strands.agent.agent as agent_mod

    original = agent_mod.event_loop_cycle
    agent_mod.event_loop_cycle = cycle_that_overflows_once
    try:
        await asyncio.wait_for(agent.invoke_async("weather?"), timeout=10.0)
    finally:
        agent_mod.event_loop_cycle = original

    # If the counter were NOT reset on the overflow retry, the pre-overflow cycle
    # would consume 1 unit of budget and we'd only get 3 real cycles before the
    # cap trips. With the reset we get the full 4-cycle budget post-recovery.
    assert model.call_count == 4, (
        f"expected 4 post-overflow cycles (counter reset), got {model.call_count}"
    )


@pytest.mark.asyncio
async def test_agent_max_iterations_resets_on_hook_resume():
    """Each hook-driven `resume` leg must get a fresh `max_iterations` budget.

    `AfterInvocationEvent` handlers can set `event.resume = <input>` to drive
    another leg of the agent loop (see `test_agent_hooks.test_after_invocation_
    resume_triggers_new_invocation`). Each resume leg is logically a fresh
    invocation from the cap's perspective — otherwise the first leg's cycles
    would bleed into every subsequent leg's budget and trip the cap prematurely.

    Regression guard: without the per-iteration `pop` inside `_run_loop`'s
    `while current_messages is not None:` loop, leg 1 consumes N cycles and
    leg 2 starts at cycle N+1, leaving only `max_iterations - N` cycles.
    """
    from strands.hooks import AfterInvocationEvent

    model = LoopingStubModel()
    agent = Agent(model=model, tools=[get_weather], max_iterations=3)

    resume_count = 0

    async def resume_once(event: AfterInvocationEvent) -> None:
        nonlocal resume_count
        if resume_count == 0:
            resume_count += 1
            event.resume = "keep going"

    agent.hooks.add_callback(AfterInvocationEvent, resume_once)

    await asyncio.wait_for(agent.invoke_async("weather?"), timeout=15.0)

    # Leg 1 burns 3 cycles (cap trips), then resume fires. If the counter
    # didn't reset on the resume leg, leg 2 would start at count=3 (already
    # at/over the cap) and trip immediately without any model calls — giving
    # a total of 3 model calls. With the reset, leg 2 gets a fresh 3-cycle
    # budget and the total is 6.
    assert resume_count == 1, "resume hook must fire exactly once"
    assert model.call_count == 6, (
        f"expected 6 total model calls (3 per leg × 2 legs with reset), "
        f"got {model.call_count}. Counter leaked across resume legs."
    )


@pytest.mark.asyncio
async def test_agent_max_iterations_halt_does_not_emit_model_message_event():
    """The halt path must NOT emit `ModelMessageEvent` — no model invocation occurred.

    `ModelMessageEvent` is the SDK's signal "the model just produced this message."
    Consumers tracking 1:1 correspondence with model calls (for metrics, retry
    accounting, replay, etc.) would miscount if the synthetic halt message were
    announced as a model-produced message. The `MessageAddedEvent` hook + terminal
    `EventLoopStopEvent` are the correct signals on this path.
    """
    from strands.types import _events as events_mod

    # Count ModelMessageEvent constructions across the invocation. Patching the
    # class __init__ is the most direct way to observe event-type emission
    # without depending on the internal wiring between `_run_loop`,
    # `stream_async`, telemetry, and `as_dict()` shapes.
    model_message_event_count = 0
    original_init = events_mod.ModelMessageEvent.__init__

    def counting_init(self_, *args, **kwargs):
        nonlocal model_message_event_count
        model_message_event_count += 1
        original_init(self_, *args, **kwargs)

    events_mod.ModelMessageEvent.__init__ = counting_init
    try:
        model = LoopingStubModel()
        agent = Agent(model=model, tools=[get_weather], max_iterations=2)
        result = await asyncio.wait_for(agent.invoke_async("weather?"), timeout=10.0)
    finally:
        events_mod.ModelMessageEvent.__init__ = original_init

    # `LoopingStubModel` emits a tool_use on every cycle, producing one real
    # `ModelMessageEvent` per real model call. With max_iterations=2 we get 2
    # real model turns, then the cap trips. The synthetic halt message must
    # NOT contribute an additional `ModelMessageEvent` — no model invocation
    # occurred on the halt path.
    assert result.stop_reason == "max_iterations"
    assert model.call_count == 2, f"expected 2 real model calls, got {model.call_count}"
    assert model_message_event_count == 2, (
        f"expected 2 ModelMessageEvents (one per real model call), got "
        f"{model_message_event_count}. The synthetic halt message must NOT emit "
        f"a ModelMessageEvent — no model invocation occurred on the halt path."
    )
