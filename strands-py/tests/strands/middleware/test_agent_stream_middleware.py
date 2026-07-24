"""Integration tests for AgentStreamStage middleware (the outermost interception point)."""

from dataclasses import replace

import pytest

import strands
from strands import Agent
from strands._middleware.stages import AgentStreamContext, AgentStreamStage, MiddlewareInterruptResult
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


async def _fail_if_called(*args, **kwargs):
    raise AssertionError("model.stream must not be called")
    yield  # noqa: B901  # make this an async generator
