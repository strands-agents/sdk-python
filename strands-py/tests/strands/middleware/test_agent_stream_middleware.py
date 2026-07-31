"""Integration tests for AgentStreamStage middleware (the outermost interception point)."""

from dataclasses import replace

import pytest

import strands
from strands import Agent
from strands._middleware.stages import AgentStreamContext, AgentStreamStage, MiddlewareInterruptResult
from strands.session import FileSessionManager
from strands.telemetry.metrics import EventLoopMetrics
from strands.types._events import EventLoopStopEvent, InitEventLoopEvent, ModelMessageEvent, TextStreamEvent
from tests.fixtures.mocked_model_provider import MockedModelProvider


@pytest.fixture
def model():
    return MockedModelProvider([{"role": "assistant", "content": [{"text": "Hello!"}]}])


@pytest.fixture
def agent(model):
    return Agent(model=model, callback_handler=None)


# --- wrap phase: wrapping the whole stream ---


def test_wrap_executes_around_full_stream(agent):
    """Middleware runs before and after the entire agent stream."""
    call_order: list[str] = []

    async def middleware(context, next_fn):
        call_order.append("before")
        async for event in next_fn(context):
            yield event
        call_order.append("after")

    agent._middleware_registry.add_middleware(AgentStreamStage, middleware)
    result = agent("Test prompt")

    assert call_order == ["before", "after"]
    assert result.stop_reason == "end_turn"
    assert result.message["content"][0]["text"] == "Hello!"


def test_wrap_receives_agent_stream_context(agent):
    """Middleware receives an AgentStreamContext with agent, messages, and invocation_state."""
    received: list[AgentStreamContext] = []

    async def capture(context, next_fn):
        received.append(context)
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, capture)
    agent("Test prompt")

    assert len(received) == 1
    ctx = received[0]
    assert ctx.agent is agent
    assert isinstance(ctx.messages, list)
    assert isinstance(ctx.invocation_state, dict)


def test_context_invocation_state_shared_by_reference(agent):
    """invocation_state is shared by reference: the context holds the caller's dict and mutations stick.

    Mirrors the TS copy-on-input suite, which pins that AgentStreamContext shares args/options by
    reference. Python's analog is invocation_state (see the README "shared by reference" callout).
    """
    caller_state = {"key": "value"}
    is_same_object = False

    async def capture(context, next_fn):
        nonlocal is_same_object
        is_same_object = context.invocation_state is caller_state
        context.invocation_state["added_by_middleware"] = True
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, capture)
    agent("Test prompt", invocation_state=caller_state)

    assert is_same_object
    # The middleware's mutation is visible on the caller's own dict (no defensive copy).
    assert caller_state["added_by_middleware"] is True


def test_wrap_short_circuits_entire_stream(agent):
    """Middleware can short-circuit the whole pass by yielding a stop event without calling next."""
    agent.model.stream = _fail_if_called

    async def short_circuit(context, next_fn):
        message = {"role": "assistant", "content": [{"text": "Short-circuited"}]}
        yield EventLoopStopEvent("end_turn", message, EventLoopMetrics(), {})

    agent._middleware_registry.add_middleware(AgentStreamStage, short_circuit)
    result = agent("Test prompt")

    assert result.stop_reason == "end_turn"
    assert result.message["content"][0]["text"] == "Short-circuited"


def test_wrap_multiple_middleware_execute_in_registration_order(agent):
    """Multiple AgentStreamStage middleware compose in registration order (outer wraps inner)."""
    call_order: list[str] = []

    async def outer(context, next_fn):
        call_order.append("outer-before")
        async for event in next_fn(context):
            yield event
        call_order.append("outer-after")

    async def inner(context, next_fn):
        call_order.append("inner-before")
        async for event in next_fn(context):
            yield event
        call_order.append("inner-after")

    agent._middleware_registry.add_middleware(AgentStreamStage, outer)
    agent._middleware_registry.add_middleware(AgentStreamStage, inner)
    agent("Test prompt")

    assert call_order == ["outer-before", "inner-before", "inner-after", "outer-after"]


def test_wrap_can_filter_events(agent):
    """Middleware can drop events from the stream without affecting the final result."""

    async def drop_init_events(context, next_fn):
        async for event in next_fn(context):
            if not isinstance(event, InitEventLoopEvent):
                yield event

    saw_init = False

    async def observer(context, next_fn):
        nonlocal saw_init
        async for event in next_fn(context):
            if isinstance(event, InitEventLoopEvent):
                saw_init = True
            yield event

    # observer (outer) sees events after drop_init_events (inner) removed them.
    agent._middleware_registry.add_middleware(AgentStreamStage, observer)
    agent._middleware_registry.add_middleware(AgentStreamStage, drop_init_events)
    result = agent("Test prompt")

    assert not saw_init
    assert result.stop_reason == "end_turn"


def test_wrap_buffers_content_across_a_multi_turn_pass():
    """Middleware wraps a whole multi-cycle pass and can suppress intermediate-turn content.

    A single AgentStreamStage pass spans every event-loop cycle of the invocation (tool-use turn
    then follow-up turn), so middleware can buffer text deltas per turn and emit only the final
    turn's — the TS "stream final turn only" use case. This exercises the stage across more than
    one turn, which the single-turn wrap tests do not.
    """
    tool_use_msg = {
        "role": "assistant",
        "content": [{"text": "Let me calculate.", "toolUse": {"toolUseId": "t1", "name": "calc", "input": {}}}],
    }
    final_msg = {"role": "assistant", "content": [{"text": "The answer is 42."}]}
    model = MockedModelProvider([tool_use_msg, final_msg])

    @strands.tool(name="calc")
    def calc() -> str:
        """Return a constant."""
        return "42"

    agent = Agent(model=model, tools=[calc], callback_handler=None)

    emitted_text: list[str] = []

    async def stream_final_turn_only(context, next_fn):
        buffered: list[TextStreamEvent] = []
        last_turn_text: list[TextStreamEvent] = []
        async for event in next_fn(context):
            if isinstance(event, TextStreamEvent):
                # Buffer this turn's text; drop it from the stream for now.
                buffered.append(event)
                continue
            if isinstance(event, ModelMessageEvent):
                # A turn completed: keep only its buffered text as the latest turn, discarding
                # any earlier intermediate turn. The last one standing is the final turn.
                last_turn_text = buffered
                buffered = []
            yield event
        # Flush the final turn's buffered text after the stream completes.
        for text_event in last_turn_text:
            emitted_text.append(text_event["data"])
            yield text_event

    agent._middleware_registry.add_middleware(AgentStreamStage, stream_final_turn_only)
    result = agent("go")

    assert result.stop_reason == "end_turn"
    # Only the final turn's text is emitted; the intermediate tool-use turn's text is suppressed.
    assert emitted_text == ["The answer is 42."]


def test_wrap_can_inject_trailing_event_after_stop(agent):
    """Middleware may yield events after the terminal stop event; the result stays authoritative.

    The result derives from the last EventLoopStopEvent, not the last event overall, so a
    trailing non-stop event injected after the stream does not displace the result.
    """

    async def inject_trailing(context, next_fn):
        async for event in next_fn(context):
            yield event
        yield InitEventLoopEvent()  # trailing non-stop event after the result

    agent._middleware_registry.add_middleware(AgentStreamStage, inject_trailing)
    result = agent("Test prompt")

    assert result.stop_reason == "end_turn"
    assert result.message["content"][0]["text"] == "Hello!"


def test_wrap_can_suppress_all_events_except_result(agent):
    """Middleware can drop every streamed event and still produce the result from the stop event."""

    async def suppress_non_results(context, next_fn):
        async for event in next_fn(context):
            if isinstance(event, EventLoopStopEvent):
                yield event

    events = []

    async def observer(context, next_fn):
        async for event in next_fn(context):
            events.append(event)
            yield event

    # observer (outer) sees only what suppress_non_results (inner) let through.
    agent._middleware_registry.add_middleware(AgentStreamStage, observer)
    agent._middleware_registry.add_middleware(AgentStreamStage, suppress_non_results)
    result = agent("Test prompt")

    assert all(isinstance(event, EventLoopStopEvent) for event in events)
    assert result.stop_reason == "end_turn"


def test_wrap_dropping_stop_event_raises_actionable_error(agent):
    """Dropping the terminal stop event produces an actionable RuntimeError, not an opaque KeyError."""

    async def drop_stop(context, next_fn):
        async for event in next_fn(context):
            if not isinstance(event, EventLoopStopEvent):
                yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, drop_stop)

    with pytest.raises(RuntimeError, match="no result event"):
        agent("Test prompt")


# --- input / output phases ---


def test_input_transforms_context_reaches_event_loop(agent):
    """A replace()-based context transform reaches the event loop, not just the next middleware.

    The terminal must run against the (possibly transformed) context it receives — a handler
    returning ``replace(context, invocation_state=...)`` must have that replacement drive the
    event loop cycle, or the transform is silently dropped.
    """
    from strands.hooks import BeforeModelCallEvent

    marker_seen_by_model = False

    def inject_marker(context):
        return replace(context, invocation_state={**context.invocation_state, "marker": "from_input"})

    def check_invocation_state(event: BeforeModelCallEvent):
        nonlocal marker_seen_by_model
        marker_seen_by_model = event.invocation_state.get("marker") == "from_input"

    agent.hooks.add_callback(BeforeModelCallEvent, check_invocation_state)
    agent._middleware_registry.add_middleware(AgentStreamStage.Input, inject_marker)
    agent("Test prompt")

    assert marker_seen_by_model


def test_messages_in_place_edit_reaches_history_but_replace_is_dropped(agent):
    """`messages` is shared by reference for in-place edits; ``replace(messages=...)`` is dropped.

    Input messages are appended to ``agent.messages`` before the chain runs, so mutating a message
    in place reaches the model, but swapping the list on the context does not (the terminal streams
    against ``agent.messages``, not ``ctx.messages``). This documents the deliberate asymmetry with
    ``invocation_state`` (which the terminal does read from the context).
    """

    async def edit_in_place(context, next_fn):
        context.messages[0]["content"] = [{"text": "mutated-in-place"}]
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, edit_in_place)
    agent("original")

    user_texts = [m["content"][0].get("text") for m in agent.messages if m["role"] == "user"]
    assert user_texts == ["mutated-in-place"]

    # A replace()-swapped list, by contrast, is not honored: history keeps the original input.
    other_model = MockedModelProvider([{"role": "assistant", "content": [{"text": "ok"}]}])
    other = Agent(model=other_model, callback_handler=None)

    async def swap_list(context, next_fn):
        modified = replace(context, messages=[{"role": "user", "content": [{"text": "replaced-list"}]}])
        async for event in next_fn(modified):
            yield event

    other._middleware_registry.add_middleware(AgentStreamStage, swap_list)
    other("original")

    other_user_texts = [m["content"][0].get("text") for m in other.messages if m["role"] == "user"]
    assert other_user_texts == ["original"]


def test_phase_ordering_at_agent_level(agent):
    """Input/Wrap/Output run in canonical order regardless of registration order."""
    order: list[str] = []

    def output_handler(result):
        order.append("output")
        return result

    async def wrap_handler(context, next_fn):
        order.append("wrap")
        async for event in next_fn(context):
            yield event

    def input_handler(context):
        order.append("input")
        return context

    agent._middleware_registry.add_middleware(AgentStreamStage.Output, output_handler)
    agent._middleware_registry.add_middleware(AgentStreamStage, wrap_handler)
    agent._middleware_registry.add_middleware(AgentStreamStage.Input, input_handler)
    agent("Test prompt")

    assert order == ["input", "wrap", "output"]


# --- no-middleware baseline ---


def test_no_agent_stream_middleware_works(agent):
    """With no AgentStreamStage middleware, the agent behaves normally (zero-overhead fast path)."""
    result = agent("hello")
    assert result.stop_reason == "end_turn"
    assert result.message["content"][0]["text"] == "Hello!"


def test_other_stage_middleware_does_not_affect_agent_stream():
    """Middleware on InvokeModelStage does not run as AgentStreamStage middleware."""
    from strands._middleware.stages import InvokeModelStage

    model = MockedModelProvider([{"role": "assistant", "content": [{"text": "ok"}]}])
    agent = Agent(model=model, callback_handler=None)

    agent_stream_ran = False

    async def invoke_model_mw(context, next_fn):
        async for event in next_fn(context):
            yield event

    async def agent_stream_mw(context, next_fn):
        nonlocal agent_stream_ran
        agent_stream_ran = True
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(InvokeModelStage, invoke_model_mw)
    agent("test")

    assert not agent_stream_ran


# --- interrupts ---


def test_middleware_interrupt_halts_agent(agent):
    """Calling context.interrupt() with no prior response halts the agent with stop_reason interrupt."""

    async def gate(context, next_fn):
        context.interrupt("confirm_stream", reason="Are you sure?")
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, gate)
    result = agent("Test")

    assert result.stop_reason == "interrupt"
    assert len(result.interrupts) == 1
    assert result.interrupts[0].name == "confirm_stream"
    assert result.interrupts[0].reason == "Are you sure?"


def test_middleware_interrupt_does_not_call_model(agent):
    """An AgentStreamStage interrupt stops before the model is invoked."""
    agent.model.stream = _fail_if_called

    async def gate(context, next_fn):
        context.interrupt("gate")
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, gate)
    result = agent("Test")

    assert result.stop_reason == "interrupt"


def test_middleware_interrupt_id_uses_agent_stream_namespace(agent):
    """The interrupt id is namespaced to the agent-stream stage and stable across resumes."""

    async def gate(context, next_fn):
        context.interrupt("my_gate")
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, gate)
    result = agent("Test")

    assert result.interrupts[0].id.startswith("v1:middleware_agent_stream:")


def test_middleware_interrupt_registered_in_state(agent):
    """When AgentStreamStage middleware interrupts, the interrupt is registered in agent state."""

    async def gate(context, next_fn):
        context.interrupt("gate", reason="confirm")
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, gate)
    result = agent("Test")

    assert result.stop_reason == "interrupt"
    assert len(agent._interrupt_state.interrupts) == 1


def test_middleware_gets_response_on_resume_and_continues():
    """After resuming, interrupt() returns the response (wrapped) and the stream continues to end_turn."""
    model = MockedModelProvider(
        [
            {"role": "assistant", "content": [{"text": "Hello!"}]},
            {"role": "assistant", "content": [{"text": "Hello!"}]},
        ]
    )
    agent = Agent(model=model, callback_handler=None)

    interrupt_result = None

    async def gate(context, next_fn):
        nonlocal interrupt_result
        interrupt_result = context.interrupt("gate", reason="Proceed?")
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, gate)

    result = agent("Test")
    assert result.stop_reason == "interrupt"

    result = agent([{"interruptResponse": {"interruptId": result.interrupts[0].id, "response": "go"}}])
    # interrupt() returns the response wrapped in a MiddlewareInterruptResult on resume.
    assert isinstance(interrupt_result, MiddlewareInterruptResult)
    assert interrupt_result.response == "go"
    assert result.stop_reason == "end_turn"
    assert result.message["content"][0]["text"] == "Hello!"


def test_middleware_interrupt_with_preemptive_response_skips_interrupt(agent):
    """Providing a preemptive response skips the interrupt entirely and the stream runs."""
    skipped = False

    async def gate(context, next_fn):
        nonlocal skipped
        interrupt_result = context.interrupt("gate", response="pre-approved")
        skipped = interrupt_result.response == "pre-approved"
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, gate)
    result = agent("Test")

    assert result.stop_reason == "end_turn"
    assert skipped


def test_resumed_interrupt_deactivates_state_after_completion():
    """After a resumed AgentStreamStage interrupt completes with end_turn, interrupt state clears.

    An agent-stream interrupt activates the interrupt state without a pending tool execution,
    so unless the run loop deactivates on non-interrupt completion, the state would leak and
    break the next fresh invocation. This guards that lifecycle.
    """
    model = MockedModelProvider(
        [
            {"role": "assistant", "content": [{"text": "First"}]},
            {"role": "assistant", "content": [{"text": "Second"}]},
        ]
    )
    agent = Agent(model=model, callback_handler=None)

    interrupted_once = False

    async def gate(context, next_fn):
        nonlocal interrupted_once
        if not interrupted_once:
            interrupted_once = True
            context.interrupt("gate")
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, gate)

    result = agent("Test")
    assert result.stop_reason == "interrupt"

    result = agent([{"interruptResponse": {"interruptId": result.interrupts[0].id, "response": "go"}}])
    assert result.stop_reason == "end_turn"
    # State is clean so a subsequent fresh invocation is not rejected.
    assert not agent._interrupt_state.activated
    assert agent._interrupt_state.interrupts == {}


def test_resumed_interrupt_can_proceed_into_a_tool_call():
    """A resumed AgentStreamStage interrupt whose pass then calls a tool completes cleanly.

    An agent-stream interrupt activates interrupt state with an empty context (no tool context).
    On resume the pass falls through to a normal model call that may return a tool_use, so the
    tool-execution path's context reads are gated on the presence of tool context rather than on
    `activated` alone — a resumed agent-stream interrupt that proceeds into a tool call runs and
    finishes without tripping over the empty context.
    """
    tool_use_msg = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "t1", "name": "calc", "input": {}}}],
    }
    final_msg = {"role": "assistant", "content": [{"text": "done"}]}
    model = MockedModelProvider([tool_use_msg, final_msg])

    @strands.tool(name="calc")
    def calc() -> str:
        """Return a constant."""
        return "42"

    agent = Agent(model=model, tools=[calc], callback_handler=None)

    interrupted_once = False

    async def gate(context, next_fn):
        nonlocal interrupted_once
        if not interrupted_once:
            interrupted_once = True
            context.interrupt("gate")
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, gate)

    result = agent("go")
    assert result.stop_reason == "interrupt"

    result = agent([{"interruptResponse": {"interruptId": result.interrupts[0].id, "response": "go"}}])
    assert result.stop_reason == "end_turn"
    assert result.message["content"][0]["text"] == "done"


def test_resumed_interrupt_reads_response_after_a_tool_cycle_runs():
    """A gate that re-reads its approval after next_fn drains still sees it when a tool ran.

    The middleware resolves its interrupt before next_fn and re-reads it after the chain drains.
    The inner pass runs a tool, whose successful cycle deactivates interrupt state in the event
    loop. The re-read must still return the resumed response (from a snapshot taken before the
    pass) rather than re-raising — otherwise the resume never converges.
    """
    tool_use_msg = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "t1", "name": "calc", "input": {}}}],
    }
    final_msg = {"role": "assistant", "content": [{"text": "done"}]}
    model = MockedModelProvider([tool_use_msg, final_msg])

    @strands.tool(name="calc")
    def calc() -> str:
        """Return a constant."""
        return "42"

    agent = Agent(model=model, tools=[calc], callback_handler=None)

    reads: list[tuple[str, object]] = []

    async def gate(context, next_fn):
        pre = context.interrupt("gate")
        reads.append(("pre", pre.response))
        async for event in next_fn(context):
            yield event
        post = context.interrupt("gate")
        reads.append(("post", post.response))

    agent._middleware_registry.add_middleware(AgentStreamStage, gate)

    result = agent("go")
    assert result.stop_reason == "interrupt"

    result = agent([{"interruptResponse": {"interruptId": result.interrupts[0].id, "response": "go"}}])
    assert result.stop_reason == "end_turn"
    assert result.message["content"][0]["text"] == "done"
    # Both the pre- and post-next_fn reads saw the resumed response (no re-raise mid-pass).
    assert reads == [("pre", "go"), ("post", "go")]


def test_resumed_agent_stream_interrupt_then_tool_interrupt():
    """A resumed agent-stream interrupt whose pass then raises a tool interrupt stops cleanly.

    Exercises the interaction between the two interrupt mechanisms: the agent-stream interrupt is
    resolved on resume (it carries a response, so it will not re-raise) while a fresh tool
    interrupt is raised in the same pass. The pass must stop with the tool interrupt, and resuming
    that one must drive the invocation to completion.
    """
    tool_use_msg = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "t1", "name": "confirm_tool", "input": {}}}],
    }
    final_msg = {"role": "assistant", "content": [{"text": "done"}]}
    model = MockedModelProvider([tool_use_msg, final_msg])

    @strands.tool(name="confirm_tool", context=True)
    def confirm_tool(tool_context) -> str:
        """Interrupt for approval, then return once resumed."""
        return tool_context.interrupt("tool_gate", reason="approve tool?")

    agent = Agent(model=model, tools=[confirm_tool], callback_handler=None)

    stream_interrupted_once = False

    async def gate(context, next_fn):
        nonlocal stream_interrupted_once
        if not stream_interrupted_once:
            stream_interrupted_once = True
            context.interrupt("stream_gate")
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, gate)

    # Pass 1: the agent-stream middleware interrupt halts the invocation.
    result = agent("go")
    assert result.stop_reason == "interrupt"
    assert result.interrupts[0].name == "stream_gate"

    # Pass 2: resume the agent-stream interrupt; the pass runs the model, calls the tool, and
    # the tool raises its own interrupt.
    result = agent([{"interruptResponse": {"interruptId": result.interrupts[0].id, "response": "go"}}])
    assert result.stop_reason == "interrupt"
    assert result.interrupts[0].name == "tool_gate"

    # Pass 3: resume the tool interrupt; the invocation completes.
    result = agent([{"interruptResponse": {"interruptId": result.interrupts[0].id, "response": "yes"}}])
    assert result.stop_reason == "end_turn"
    assert result.message["content"][0]["text"] == "done"
    # No interrupt state leaks past a clean completion.
    assert not agent._interrupt_state.activated
    assert agent._interrupt_state.interrupts == {}


def test_cancel_during_agent_stream_interrupt_resume_clears_state():
    """Cancelling a resumed AgentStreamStage interrupt clears the interrupt state.

    An agent-stream interrupt never stores tool context, so a cancelled resume ends the pass
    with stop_reason "cancelled" and the run loop deactivates the interrupt state — the agent is
    left clean and reusable. (This is the agent-stream counterpart to a tool interrupt, whose
    state the event loop deliberately preserves across a cancelled resume.)
    """
    model = MockedModelProvider(
        [
            {"role": "assistant", "content": [{"text": "First"}]},
            {"role": "assistant", "content": [{"text": "Second"}]},
        ]
    )
    agent = Agent(model=model, callback_handler=None)

    interrupted_once = False

    async def gate(context, next_fn):
        nonlocal interrupted_once
        if not interrupted_once:
            interrupted_once = True
            context.interrupt("gate")
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, gate)

    result = agent("Test")
    assert result.stop_reason == "interrupt"

    # Cancel the resume; the pass ends cancelled and the agent-stream interrupt state is cleared.
    agent.cancel()
    result = agent([{"interruptResponse": {"interruptId": result.interrupts[0].id, "response": "go"}}])

    assert result.stop_reason == "cancelled"
    assert not agent._interrupt_state.activated
    assert agent._interrupt_state.interrupts == {}


def test_sequential_agent_stream_interrupts_across_passes():
    """Two agent-stream interrupts in a row each halt and resume, cycling activate/deactivate.

    The chain is rebuilt per pass, so the interrupt state must fully reset between them: a first
    interrupt halts and resumes, a second (distinct) interrupt halts and resumes, and the final
    pass completes with clean state. This guards the activate -> deactivate -> activate ->
    deactivate lifecycle the run loop drives across passes.
    """
    model = MockedModelProvider(
        [
            {"role": "assistant", "content": [{"text": "a"}]},
            {"role": "assistant", "content": [{"text": "b"}]},
            {"role": "assistant", "content": [{"text": "c"}]},
        ]
    )
    agent = Agent(model=model, callback_handler=None)

    pass_count = 0

    async def gate(context, next_fn):
        nonlocal pass_count
        pass_count += 1
        if pass_count in (1, 2):
            context.interrupt(f"gate_{pass_count}")
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, gate)

    result = agent("go")
    assert result.stop_reason == "interrupt"
    assert result.interrupts[0].name == "gate_1"

    result = agent([{"interruptResponse": {"interruptId": result.interrupts[0].id, "response": "x"}}])
    assert result.stop_reason == "interrupt"
    assert result.interrupts[0].name == "gate_2"

    result = agent([{"interruptResponse": {"interruptId": result.interrupts[0].id, "response": "y"}}])
    assert result.stop_reason == "end_turn"
    assert not agent._interrupt_state.activated
    assert agent._interrupt_state.interrupts == {}


def test_interrupt_message_uses_last_message_when_messages_exist(agent):
    """The interrupt result message is the last message in history.

    Python appends the prompt to history before the AgentStreamStage chain runs, so at interrupt
    time ``self.messages[-1]`` is the current pass's user prompt (unlike TS, which appends inside
    ``next()`` and so hits the "Interrupted" fallback for a fresh agent). This pins the Python
    behavior — the result message is the last existing message.
    """

    async def gate(context, next_fn):
        context.interrupt("gate")
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, gate)
    result = agent("Test")

    assert result.stop_reason == "interrupt"
    assert result.message == agent.messages[-1]


# --- hooks fire outside the middleware chain ---


def test_invocation_hooks_fire_outside_agent_stream_middleware(agent):
    """Before/AfterInvocationEvent fire outside the AgentStreamStage chain (before/after it)."""
    from strands.hooks import AfterInvocationEvent, BeforeInvocationEvent

    order: list[str] = []

    class RecordingHooks:
        def register_hooks(self, registry, **kwargs):
            registry.add_callback(BeforeInvocationEvent, lambda event: order.append("before_invocation"))
            registry.add_callback(AfterInvocationEvent, lambda event: order.append("after_invocation"))

    agent.hooks.add_hook(RecordingHooks())

    async def middleware(context, next_fn):
        order.append("middleware-before")
        async for event in next_fn(context):
            yield event
        order.append("middleware-after")

    agent._middleware_registry.add_middleware(AgentStreamStage, middleware)
    agent("Test prompt")

    assert order == ["before_invocation", "middleware-before", "middleware-after", "after_invocation"]


def test_after_invocation_hook_fires_when_middleware_short_circuits(agent):
    """AfterInvocationEvent still fires (in the finally) when middleware short-circuits the pass."""
    from strands.hooks import AfterInvocationEvent

    after_fired = False

    class RecordingHooks:
        def register_hooks(self, registry, **kwargs):
            def _record(event):
                nonlocal after_fired
                after_fired = True

            registry.add_callback(AfterInvocationEvent, _record)

    agent.hooks.add_hook(RecordingHooks())

    async def short_circuit(context, next_fn):
        message = {"role": "assistant", "content": [{"text": "Short-circuited"}]}
        yield EventLoopStopEvent("end_turn", message, EventLoopMetrics(), {})

    agent._middleware_registry.add_middleware(AgentStreamStage, short_circuit)
    agent("Test prompt")

    assert after_fired


def test_user_message_added_hook_fires_before_the_chain(agent):
    """The input MessageAddedEvent fires before the AgentStreamStage chain, not inside it.

    Input messages are appended to history before the chain runs, so a wrap middleware sees the
    user turn already in ``agent.messages`` when it starts. This pins that ordering (a divergence
    from TS, which appends inside the terminal) so it can't regress silently.
    """
    from strands.hooks import MessageAddedEvent

    order: list[str] = []

    agent.hooks.add_callback(
        MessageAddedEvent, lambda event: order.append(f"msg_added:{event.message['role']}")
    )

    async def middleware(context, next_fn):
        order.append("middleware-before")
        async for event in next_fn(context):
            yield event
        order.append("middleware-after")

    agent._middleware_registry.add_middleware(AgentStreamStage, middleware)
    agent("Test prompt")

    # The user message is added before the chain starts; the assistant message during it.
    assert order == ["msg_added:user", "middleware-before", "msg_added:assistant", "middleware-after"]


def test_user_message_added_hook_fires_even_when_middleware_short_circuits(agent):
    """The input MessageAddedEvent fires even when a middleware short-circuits the pass.

    Because the input is appended before the chain, a short-circuit (which never runs the
    terminal) still records the user turn in history and still fires MessageAddedEvent — unlike
    TS, where the append lives in the terminal and is skipped on short-circuit.
    """
    from strands.hooks import MessageAddedEvent

    added_roles: list[str] = []

    agent.hooks.add_callback(MessageAddedEvent, lambda event: added_roles.append(event.message["role"]))

    async def short_circuit(context, next_fn):
        message = {"role": "assistant", "content": [{"text": "Short-circuited"}]}
        yield EventLoopStopEvent("end_turn", message, EventLoopMetrics(), {})

    agent._middleware_registry.add_middleware(AgentStreamStage, short_circuit)
    agent("Test prompt")

    assert added_roles == ["user"]
    assert agent.messages[-1]["role"] == "user"
    assert agent.messages[-1]["content"] == [{"text": "Test prompt"}]


def test_context_replace_preserves_interrupt(agent):
    """dataclasses.replace() on AgentStreamContext preserves interrupt functionality."""

    async def replace_then_interrupt(context, next_fn):
        modified = replace(context, invocation_state={**context.invocation_state, "tagged": True})
        modified.interrupt("gate")
        async for event in next_fn(modified):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, replace_then_interrupt)
    result = agent("Test")

    assert result.stop_reason == "interrupt"
    assert result.interrupts[0].id.startswith("v1:middleware_agent_stream:")


# --- tool interrupts still surface through the agent-stream chain ---


def test_tool_interrupt_surfaces_through_agent_stream_chain():
    """A tool-originated interrupt flows through AgentStreamStage middleware as a normal event."""

    @strands.tool(name="interrupting_tool", context=True)
    def interrupting_tool(tool_context) -> str:
        """Interrupts on first call."""
        return tool_context.interrupt("confirm", reason="approve?")

    tool_use_msg = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "t1", "name": "interrupting_tool", "input": {}}}],
    }
    final_msg = {"role": "assistant", "content": [{"text": "done"}]}
    model = MockedModelProvider([tool_use_msg, final_msg])
    agent = Agent(model=model, tools=[interrupting_tool], callback_handler=None)

    async def passthrough(context, next_fn):
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, passthrough)
    result = agent("go")

    assert result.stop_reason == "interrupt"
    assert result.interrupts[0].name == "confirm"


def test_agent_stream_interrupt_reports_all_unanswered_interrupts():
    """An agent-stream interrupt reports every still-unanswered interrupt, not just the one it raised.

    Mixing interrupt sources within one invocation: a tool raises two interrupts, the caller
    resumes only one, and on the resumed pass an agent-stream middleware raises a fresh one. The
    resulting stop must report both the still-unanswered tool interrupt and the new agent-stream
    one, so a caller treating ``result.interrupts`` as "everything I owe an answer to" is correct.
    """
    from strands.interrupt import Interrupt
    from strands.types._events import ToolInterruptEvent

    @strands.tool(name="multi_tool")
    def multi_tool() -> str:
        """Placeholder; real behavior is in the custom stream below."""
        return "unused"

    tool_interrupts = [
        Interrupt(id="v1:tool:a", name="gate_a", reason="a?"),
        Interrupt(id="v1:tool:b", name="gate_b", reason="b?"),
    ]

    async def multi_stream(tool_use, _invocation_state, **_kwargs):
        yield ToolInterruptEvent(tool_use, tool_interrupts)

    multi_tool.stream = multi_stream

    tool_use_msg = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "t1", "name": "multi_tool", "input": {}}}],
    }
    model = MockedModelProvider([tool_use_msg, {"role": "assistant", "content": [{"text": "done"}]}])
    agent = Agent(model=model, tools=[multi_tool], callback_handler=None)

    raise_stream_gate = False

    async def gate(context, next_fn):
        if raise_stream_gate:
            context.interrupt("stream_gate", reason="confirm stream?")
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, gate)

    # Pass 1: tool raises two interrupts.
    result = agent("go")
    assert result.stop_reason == "interrupt"
    assert {interrupt.name for interrupt in result.interrupts} == {"gate_a", "gate_b"}

    # Pass 2: answer only gate_a; the agent-stream middleware raises stream_gate on the resumed pass.
    raise_stream_gate = True
    result = agent([{"interruptResponse": {"interruptId": "v1:tool:a", "response": "yes"}}])

    assert result.stop_reason == "interrupt"
    # Both the still-unanswered tool interrupt and the fresh agent-stream one are reported.
    assert {interrupt.name for interrupt in result.interrupts} == {"gate_b", "stream_gate"}


def test_resumed_interrupt_is_not_re_asked_after_a_later_tool_interrupt():
    """An answered gate is asked once, even when a later pass of the same cycle interrupts again.

    The gate reads its approval before next_fn only. The resumed pass runs a successful tool cycle
    (which completes the tool resume) and then a second tool raises its own interrupt, so a further
    pass follows. The gate's answer must survive those passes: asking again would make the human
    re-approve the same action once per interrupt round trip.
    """
    model = MockedModelProvider(
        [
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "calc", "input": {}}}]},
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t2", "name": "confirm_tool", "input": {}}}]},
            {"role": "assistant", "content": [{"text": "done"}]},
        ]
    )

    @strands.tool(name="calc")
    def calc() -> str:
        """Return a constant."""
        return "42"

    @strands.tool(name="confirm_tool", context=True)
    def confirm_tool(tool_context) -> str:
        """Interrupt for approval, then return once resumed."""
        return tool_context.interrupt("tool_gate", reason="approve tool?")

    agent = Agent(model=model, tools=[calc, confirm_tool], callback_handler=None)

    async def gate(context, next_fn):
        context.interrupt("stream_gate", reason="approve the pass?")
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, gate)

    asked: list[str] = []
    result = agent("go")
    for _ in range(8):
        if result.stop_reason != "interrupt":
            break
        asked.extend(interrupt.name for interrupt in result.interrupts)
        result = agent(
            [
                {"interruptResponse": {"interruptId": interrupt.id, "response": "yes"}}
                for interrupt in result.interrupts
            ]
        )

    assert result.stop_reason == "end_turn"
    assert asked == ["stream_gate", "tool_gate"]


def test_answered_interrupt_is_not_reused_by_a_later_invocation():
    """A gate's answer dies with its interrupt cycle, so the next invocation asks again.

    The answered response is kept across the passes of one cycle (so the human is asked once),
    which must not turn into a standing approval. The resumed pass here runs a tool, so the tool
    cycle completes the tool resume and leaves the state deactivated with the answer still held —
    the cycle therefore ends without the run loop's interrupt-completion path firing, and the
    answer has to be dropped anyway.
    """
    model = MockedModelProvider(
        [
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "calc", "input": {}}}]},
            {"role": "assistant", "content": [{"text": "done"}]},
            {"role": "assistant", "content": [{"text": "unused"}]},
        ]
    )

    @strands.tool(name="calc")
    def calc() -> str:
        """Return a constant."""
        return "42"

    agent = Agent(model=model, tools=[calc], callback_handler=None)

    async def gate(context, next_fn):
        context.interrupt("stream_gate", reason="approve the pass?")
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, gate)

    first = agent("go")
    assert first.stop_reason == "interrupt"
    first = agent([{"interruptResponse": {"interruptId": first.interrupts[0].id, "response": "yes"}}])
    assert first.stop_reason == "end_turn"
    assert first.message["content"][0]["text"] == "done"
    assert not agent._interrupt_state.interrupts

    # A second, independent invocation must gate again rather than reuse the earlier approval.
    second = agent("go again")
    assert second.stop_reason == "interrupt"
    assert second.interrupts[0].name == "stream_gate"


def test_interrupt_after_the_pass_completes_raises():
    """Interrupting after the stream finished is refused instead of replaying the pass.

    Resuming such an interrupt has no tool-use message to replay, so the event loop would call the
    model a second time and append a duplicate assistant turn. The agent raises an actionable error
    instead. Interrupting before or while draining next_fn is unaffected.
    """
    model = MockedModelProvider([{"role": "assistant", "content": [{"text": "Hello!"}]}])
    agent = Agent(model=model, callback_handler=None)

    async def post_hoc_gate(context, next_fn):
        async for event in next_fn(context):
            yield event
        context.interrupt("approve_output", reason="approve the reply?")

    agent._middleware_registry.add_middleware(AgentStreamStage, post_hoc_gate)

    exp_message = r"interrupt_name=<approve_output> \| agent-stream middleware interrupted after the pass"
    with pytest.raises(RuntimeError, match=exp_message):
        agent("Test prompt")

    # The refused interrupt is not left registered, so the next invocation starts clean.
    assert not agent._interrupt_state.activated
    assert not agent._interrupt_state.interrupts


def test_after_invocation_result_is_the_stop_event_when_a_trailing_event_follows(agent):
    """AfterInvocationEvent.result comes from the stop event, not from a later trailing event."""
    from strands.hooks import AfterInvocationEvent

    results = []

    async def inject_trailing(context, next_fn):
        async for event in next_fn(context):
            yield event
        yield InitEventLoopEvent()

    agent._middleware_registry.add_middleware(AgentStreamStage, inject_trailing)
    agent.hooks.add_callback(AfterInvocationEvent, lambda event: results.append(event.result))

    agent("Test prompt")

    assert len(results) == 1
    assert results[0] is not None
    assert results[0].stop_reason == "end_turn"
    assert results[0].message["content"][0]["text"] == "Hello!"


def test_middleware_yielded_interrupt_stop_preserves_interrupt_state():
    """A pass that stops on an interrupt keeps its interrupt state, even with no tool context.

    Middleware can surface an interrupt stop itself (short-circuiting the pass). The run loop must
    not clear interrupt state after such a pass: the caller still owes a response.
    """
    model = MockedModelProvider(
        [
            {"role": "assistant", "content": [{"text": "first"}]},
            {"role": "assistant", "content": [{"text": "second"}]},
        ]
    )
    agent = Agent(model=model, callback_handler=None)

    short_circuit = False

    async def gate(context, next_fn):
        if short_circuit:
            message = {"role": "assistant", "content": [{"text": "Awaiting approval"}]}
            pending = list(agent._interrupt_state.interrupts.values())
            yield EventLoopStopEvent("interrupt", message, EventLoopMetrics(), {}, pending)
            return
        context.interrupt("stream_gate", reason="approve the pass?")
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, gate)

    result = agent("go")
    assert result.stop_reason == "interrupt"
    interrupt_id = result.interrupts[0].id

    # The resumed pass stops on a middleware-yielded interrupt stop event.
    short_circuit = True
    result = agent([{"interruptResponse": {"interruptId": interrupt_id, "response": "yes"}}])

    assert result.stop_reason == "interrupt"
    assert agent._interrupt_state.activated
    assert interrupt_id in agent._interrupt_state.interrupts


def _charging_agent(gate, charges, **kwargs):
    """An agent whose model calls a charging tool, gated by ``gate``."""
    tool_use = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "t1", "name": "charge_card", "input": {"amount": "$100"}}}],
    }
    done = {"role": "assistant", "content": [{"text": "done"}]}

    @strands.tool(name="charge_card")
    def charge_card(amount: str) -> str:
        """Charge the customer's card."""
        charges.append(amount)
        return f"charged {amount}"

    agent = Agent(
        model=MockedModelProvider([tool_use, done, tool_use, done]),
        tools=[charge_card],
        callback_handler=None,
        **kwargs,
    )
    agent._middleware_registry.add_middleware(AgentStreamStage, gate)
    return agent


def _approve(result):
    return [
        {"interruptResponse": {"interruptId": interrupt.id, "response": "yes"}} for interrupt in result.interrupts
    ]


def _charge_gate(gate_calls):
    async def gate(context, next_fn):
        gate_calls.append("call")
        context.interrupt("approve_charge", reason="approve this request?")
        async for event in next_fn(context):
            yield event

    return gate


def test_answered_interrupt_is_not_persisted_as_a_standing_approval(tmp_path):
    """A completed cycle leaves no answered interrupt in the session.

    The cycle is released before ``AfterInvocationEvent``, which is where the session manager
    syncs, so the persisted state carries no response. A fresh agent restored from that session —
    the ordinary one-agent-per-request server shape — must gate again rather than resolve against
    what the previous cycle answered.
    """
    gate_calls: list[str] = []
    charges: list[str] = []

    def build():
        return _charging_agent(
            _charge_gate(gate_calls),
            charges,
            session_manager=FileSessionManager(session_id="s1", storage_dir=str(tmp_path)),
        )

    first = build()("charge $100")
    assert first.stop_reason == "interrupt"

    resumed = build()(_approve(first))
    assert resumed.stop_reason == "end_turn"

    persisted = FileSessionManager(session_id="s1", storage_dir=str(tmp_path)).read_agent("s1", "default")
    assert persisted._internal_state["interrupt_state"]["interrupts"] == {}

    charges.clear()
    third = build()("charge $9999")
    assert third.stop_reason == "interrupt"
    assert charges == []


def test_answered_interrupt_is_released_when_a_pass_ends_with_an_error():
    """A raise from an AfterInvocationEvent hook does not strand an answered interrupt.

    The hook runs inside the run loop's ``finally`` and is a public extension point, so a failure
    there (a session write error, a summarizing conversation manager whose model call fails) must
    not leave a response behind for the next cycle to resolve against.
    """
    from strands.hooks import AfterInvocationEvent

    gate_calls: list[str] = []
    charges: list[str] = []
    agent = _charging_agent(_charge_gate(gate_calls), charges)

    hook_calls: list[object] = []

    def failing_hook(event):
        hook_calls.append(event)
        if len(hook_calls) == 2:  # the resumed pass
            raise RuntimeError("session write failed")

    agent.hooks.add_callback(AfterInvocationEvent, failing_hook)

    first = agent("charge $100")
    assert first.stop_reason == "interrupt"
    with pytest.raises(RuntimeError, match="session write failed"):
        agent(_approve(first))

    assert agent._interrupt_state.interrupts == {}

    # The next cycle raises its own unanswered interrupt rather than inheriting the answered one,
    # which would leave the caller told to respond with nothing to respond to.
    charges.clear()
    third = agent("charge $9999")
    assert third.stop_reason == "interrupt"
    assert [(interrupt.name, interrupt.response) for interrupt in third.interrupts] == [("approve_charge", None)]
    assert charges == []


@pytest.mark.asyncio
async def test_answered_interrupt_is_not_reused_after_the_caller_abandons_the_stream():
    """A caller that stops consuming the stream leaves no usable approval behind.

    The run loop's cleanup only runs if the generator is driven to completion, so a disconnected
    client must not leave a response that the next cycle's gate resolves against.
    """
    gate_calls: list[str] = []
    charges: list[str] = []
    agent = _charging_agent(_charge_gate(gate_calls), charges)

    first = await agent.invoke_async("charge $100")
    assert first.stop_reason == "interrupt"

    stream = agent.stream_async(_approve(first))
    async for event in stream:
        message = event.get("message")
        if message and message["role"] == "user" and "toolResult" in message["content"][0]:
            break
    await stream.aclose()

    charges.clear()
    second = await agent.invoke_async("charge $9999")
    assert second.stop_reason == "interrupt"
    assert charges == []


def test_interrupt_after_a_tool_interrupt_stop_is_allowed():
    """Gating on top of a tool interrupt works: the stored tool use is replayed on resume.

    Refusing a post-result interrupt is only correct when resuming would re-call the model. A pass
    that stopped for a tool interrupt has a tool-use message to replay, so the pass-level gate can
    still interrupt after the stream drains and the caller answers both in one round trip.
    """
    tool_use = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "t1", "name": "confirm_tool", "input": {}}}],
    }
    done = {"role": "assistant", "content": [{"text": "done"}]}

    @strands.tool(name="confirm_tool", context=True)
    def confirm_tool(tool_context) -> str:
        """Interrupt for approval, then return once resumed."""
        return f"ran: {tool_context.interrupt('tool_gate', reason='approve tool?')}"

    agent = Agent(model=MockedModelProvider([tool_use, done]), tools=[confirm_tool], callback_handler=None)

    async def batch_gate(context, next_fn):
        stopped_for_interrupt = False
        async for event in next_fn(context):
            if isinstance(event, EventLoopStopEvent) and event["stop"][0] == "interrupt":
                stopped_for_interrupt = True
            yield event
        if stopped_for_interrupt:
            context.interrupt("batch_approve", reason="approve the batch too?")

    agent._middleware_registry.add_middleware(AgentStreamStage, batch_gate)

    first = agent("go")
    assert first.stop_reason == "interrupt"
    assert {interrupt.name for interrupt in first.interrupts} == {"tool_gate", "batch_approve"}

    resumed = agent(_approve(first))
    assert resumed.stop_reason == "end_turn"
    assert [message["role"] for message in agent.messages] == ["user", "assistant", "user", "assistant"]


def test_interrupt_after_the_pass_completes_on_a_resumed_pass_leaves_no_state():
    """The refusal clears interrupt state, so the agent stays usable afterwards.

    A refusal on a resumed pass would otherwise leave the state activated with an interrupt the
    caller can no longer answer, wedging every later call.
    """
    model = MockedModelProvider(
        [
            {"role": "assistant", "content": [{"text": "first"}]},
            {"role": "assistant", "content": [{"text": "second"}]},
        ]
    )
    agent = Agent(model=model, callback_handler=None)

    # "before": gate ahead of next_fn; "after": gate once the stream drained; "off": pass through.
    mode = "before"

    async def gate(context, next_fn):
        if mode == "before":
            context.interrupt("stream_gate", reason="approve the pass?")
        async for event in next_fn(context):
            yield event
        if mode == "after":
            context.interrupt("post_gate", reason="approve the output?")

    agent._middleware_registry.add_middleware(AgentStreamStage, gate)

    first = agent("go")
    assert first.stop_reason == "interrupt"

    mode = "after"
    with pytest.raises(RuntimeError, match="interrupted after the pass produced its result"):
        agent(_approve(first))

    assert not agent._interrupt_state.activated
    assert not agent._interrupt_state.interrupts
    assert not agent._interrupt_state.context

    # The agent takes an ordinary prompt again rather than demanding interrupt responses.
    mode = "off"
    assert agent("plain prompt").stop_reason == "end_turn"


async def _drain_until_tool_result(agent, prompt):
    """Consume a stream up to the tool result, then abandon it like a disconnected client."""
    stream = agent.stream_async(prompt)
    async for event in stream:
        message = event.get("message")
        if message and message["role"] == "user" and "toolResult" in message["content"][0]:
            break
    await stream.aclose()


@pytest.mark.asyncio
async def test_answered_interrupt_is_not_persisted_when_a_stream_is_abandoned(tmp_path):
    """An abandoned stream leaves no answered interrupt in the session.

    Sessions sync on every message added, including the one right after a tool cycle retains an
    answered response, so keeping the retained response out of the session cannot rely on the
    end-of-cycle release running.
    """
    gate_calls: list[str] = []
    charges: list[str] = []

    def build():
        return _charging_agent(
            _charge_gate(gate_calls),
            charges,
            session_manager=FileSessionManager(session_id="s1", storage_dir=str(tmp_path)),
        )

    first = await build().invoke_async("charge $100")
    assert first.stop_reason == "interrupt"

    await _drain_until_tool_result(build(), _approve(first))

    persisted = FileSessionManager(session_id="s1", storage_dir=str(tmp_path)).read_agent("s1", "default")
    assert persisted._internal_state["interrupt_state"]["interrupts"] == {}


@pytest.mark.asyncio
async def test_restored_agent_reports_an_answerable_interrupt(tmp_path):
    """A restored agent never reports an interrupt stop the caller cannot answer.

    A stale answered entry would both satisfy the gate silently and be filtered out of the reported
    interrupts, leaving the caller told to respond with nothing to respond to.
    """
    gate_calls: list[str] = []
    charges: list[str] = []

    def build():
        return _charging_agent(
            _charge_gate(gate_calls),
            charges,
            session_manager=FileSessionManager(session_id="s1", storage_dir=str(tmp_path)),
        )

    first = await build().invoke_async("charge $100")
    await _drain_until_tool_result(build(), _approve(first))

    charges.clear()
    restored = await build().invoke_async("charge $9999 for something else")

    assert restored.stop_reason == "interrupt"
    assert [interrupt.name for interrupt in restored.interrupts] == ["approve_charge"]
    assert charges == []


def test_interrupt_after_a_middleware_yielded_result_is_allowed():
    """Gating after a short-circuited pass works: nothing was produced for a resume to replay.

    A middleware that yields the result itself never calls the model, so the resumed pass replays
    nothing and re-yields the same result — there is no duplicate assistant turn to prevent.
    """
    agent = Agent(
        model=MockedModelProvider([{"role": "assistant", "content": [{"text": "unused"}]}]),
        callback_handler=None,
    )
    agent.model.stream = _fail_if_called

    async def cached_then_confirm(context, next_fn):
        message = {"role": "assistant", "content": [{"text": "cached answer"}]}
        yield EventLoopStopEvent("end_turn", message, EventLoopMetrics(), {})
        context.interrupt("confirm_cached", reason="serve the cached answer?")

    agent._middleware_registry.add_middleware(AgentStreamStage, cached_then_confirm)

    first = agent("what is 2+2?")
    assert first.stop_reason == "interrupt"
    assert [interrupt.name for interrupt in first.interrupts] == ["confirm_cached"]

    resumed = agent(_approve(first))
    assert resumed.stop_reason == "end_turn"
    assert resumed.message["content"][0]["text"] == "cached answer"


def test_answered_interrupt_is_released_when_conversation_management_raises():
    """A failure in conversation management does not strand an answered interrupt.

    apply_management runs inside the run loop's finally, so the release has to happen before it for
    a raise there to leave nothing behind.
    """
    gate_calls: list[str] = []
    charges: list[str] = []
    agent = _charging_agent(_charge_gate(gate_calls), charges)

    calls: list[int] = []
    original_apply = agent.conversation_manager.apply_management

    def failing_apply(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:  # the resumed pass
            raise RuntimeError("summarization unavailable")
        return original_apply(*args, **kwargs)

    agent.conversation_manager.apply_management = failing_apply

    first = agent("charge $100")
    assert first.stop_reason == "interrupt"
    with pytest.raises(RuntimeError, match="summarization unavailable"):
        agent(_approve(first))

    assert agent._interrupt_state.interrupts == {}

async def _fail_if_called(*args, **kwargs):
    raise AssertionError("model.stream must not be called")
    yield  # noqa: B901  # make this an async generator
