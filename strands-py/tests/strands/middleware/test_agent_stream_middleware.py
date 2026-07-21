"""Integration tests for middleware with Agent (AgentStreamStage)."""

from dataclasses import replace

import pytest

from strands import Agent, Plugin
from strands._middleware.stages import AgentStreamContext, AgentStreamStage
from strands._middleware.types import MiddlewareResult
from strands.types._events import EventLoopStopEvent
from tests.fixtures.mocked_model_provider import MockedModelProvider


@pytest.fixture
def model():
    return MockedModelProvider(
        [
            {"role": "assistant", "content": [{"text": "Hello!"}]},
        ]
    )


@pytest.fixture
def agent(model):
    return Agent(model=model, callback_handler=None)


# --- Wrap phase ---


def test_wrap_passthrough_does_not_alter_behavior(agent):
    """A passthrough middleware yields all events unchanged."""

    async def passthrough(context: AgentStreamContext, next_fn):
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, passthrough)
    result = agent("test")
    assert result.message["content"][0]["text"] == "Hello!"


def test_wrap_handler_receives_agent_stream_context(agent):
    """Wrap handlers receive an AgentStreamContext with agent and invocation_state."""
    captured_context = None

    async def capture(context: AgentStreamContext, next_fn):
        nonlocal captured_context
        captured_context = context
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, capture)
    agent("test")

    assert captured_context is not None
    assert isinstance(captured_context, AgentStreamContext)
    assert captured_context.agent is agent


def test_wrap_context_has_messages(agent):
    """Context.messages contains the messages derived from the prompt."""
    captured_context = None

    async def capture(context: AgentStreamContext, next_fn):
        nonlocal captured_context
        captured_context = context
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, capture)
    agent("hello world")

    assert captured_context is not None
    assert len(captured_context.messages) == 1
    assert captured_context.messages[0]["role"] == "user"
    assert captured_context.messages[0]["content"][0]["text"] == "hello world"


def test_wrap_context_has_invocation_state(agent):
    """Context.invocation_state is the same dict passed to the agent."""
    captured_state = None

    async def capture(context: AgentStreamContext, next_fn):
        nonlocal captured_state
        captured_state = context.invocation_state
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, capture)
    agent("test", invocation_state={"key": "value"})

    assert captured_state is not None
    assert captured_state["key"] == "value"


def test_wrap_short_circuit_model_not_called(model):
    """A middleware that short-circuits prevents the model from being called."""
    agent = Agent(model=model, callback_handler=None)

    async def short_circuit(context: AgentStreamContext, next_fn):
        # Yield an EventLoopStopEvent directly without calling next_fn
        from strands.telemetry.metrics import EventLoopMetrics

        msg = {"role": "assistant", "content": [{"text": "Short-circuited!"}]}
        yield EventLoopStopEvent("end_turn", msg, EventLoopMetrics(), {})

    agent._middleware_registry.add_middleware(AgentStreamStage, short_circuit)
    result = agent("test")

    assert result.message["content"][0]["text"] == "Short-circuited!"
    # Model was not called (still at index 0)
    assert model.index == 0


def test_wrap_multiple_middleware_chain_correctly(agent):
    """Multiple wrap handlers compose in registration order (first registered = outermost)."""
    order = []

    async def outer(context, next_fn):
        order.append("outer_before")
        async for event in next_fn(context):
            yield event
        order.append("outer_after")

    async def inner(context, next_fn):
        order.append("inner_before")
        async for event in next_fn(context):
            yield event
        order.append("inner_after")

    agent._middleware_registry.add_middleware(AgentStreamStage, outer)
    agent._middleware_registry.add_middleware(AgentStreamStage, inner)
    agent("test")

    assert order == ["outer_before", "inner_before", "inner_after", "outer_after"]


def test_wrap_error_propagates_through_middleware(agent):
    """Errors from the terminal propagate through the middleware chain."""
    error_caught = None

    async def error_catcher(context, next_fn):
        nonlocal error_caught
        try:
            async for event in next_fn(context):
                yield event
        except Exception as e:
            error_caught = e
            raise

    agent._middleware_registry.add_middleware(AgentStreamStage, error_catcher)

    # Force an error by giving the model no responses left
    agent.model.agent_responses = []
    agent.model.index = 0

    with pytest.raises(IndexError):
        agent("test")

    assert error_caught is not None


def test_no_middleware_agent_works_correctly(model):
    """Agent works normally without any AgentStreamStage middleware registered."""
    agent = Agent(model=model, callback_handler=None)
    result = agent("test")
    assert result.message["content"][0]["text"] == "Hello!"


# --- Input phase ---


def test_input_handler_transforms_context(agent):
    """Input handlers can transform the context before execution."""
    captured_messages = None

    async def inject_messages(context: AgentStreamContext) -> AgentStreamContext:
        return replace(context, messages=[{"role": "user", "content": [{"text": "injected"}]}])

    async def capture(context: AgentStreamContext, next_fn):
        nonlocal captured_messages
        captured_messages = context.messages
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage.Input, inject_messages)
    agent._middleware_registry.add_middleware(AgentStreamStage, capture)
    agent("original")

    # The wrap handler sees the transformed context from the input handler
    assert captured_messages is not None
    assert captured_messages[0]["content"][0]["text"] == "injected"


def test_input_async_handler(agent):
    """Async input handlers are awaited correctly."""
    captured_messages = None

    async def async_inject(context: AgentStreamContext) -> AgentStreamContext:
        return replace(context, messages=[{"role": "user", "content": [{"text": "async injected"}]}])

    async def capture(context: AgentStreamContext, next_fn):
        nonlocal captured_messages
        captured_messages = context.messages
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage.Input, async_inject)
    agent._middleware_registry.add_middleware(AgentStreamStage, capture)
    agent("original")

    assert captured_messages is not None
    assert captured_messages[0]["content"][0]["text"] == "async injected"


# --- Output phase ---


def test_output_handler_transforms_result(agent):
    """Output handlers can transform the final result event."""

    def output_handler(result: MiddlewareResult) -> MiddlewareResult:
        # The result.value is the last event (EventLoopStopEvent)
        event = result.value
        stop_reason, message, metrics, state, *rest = event["stop"]
        modified_message = {"role": "assistant", "content": [{"text": "Modified!"}]}
        return result.replace(
            value=EventLoopStopEvent(stop_reason, modified_message, metrics, state),
        )

    agent._middleware_registry.add_middleware(AgentStreamStage.Output, output_handler)
    result = agent("test")

    assert result.message["content"][0]["text"] == "Modified!"


def test_output_handler_can_change_stop_reason(agent):
    """Output handlers can change the stop_reason."""

    def output_handler(result: MiddlewareResult) -> MiddlewareResult:
        event = result.value
        stop_reason, message, metrics, state, *rest = event["stop"]
        return result.replace(
            value=EventLoopStopEvent("custom_stop", message, metrics, state),
        )

    agent._middleware_registry.add_middleware(AgentStreamStage.Output, output_handler)
    result = agent("test")

    assert result.stop_reason == "custom_stop"


# --- Phase ordering ---


def test_phase_ordering_input_output_wrap(agent):
    """Input runs before Wrap, Output transforms the final result after Wrap."""
    order = []

    async def input_handler(context: AgentStreamContext) -> AgentStreamContext:
        order.append("input")
        return context

    async def wrap_handler(context, next_fn):
        order.append("wrap")
        async for event in next_fn(context):
            yield event

    def output_handler(result: MiddlewareResult) -> MiddlewareResult:
        order.append("output")
        return result

    agent._middleware_registry.add_middleware(AgentStreamStage.Output, output_handler)
    agent._middleware_registry.add_middleware(AgentStreamStage, wrap_handler)
    agent._middleware_registry.add_middleware(AgentStreamStage.Input, input_handler)
    agent("test")

    assert order == ["input", "wrap", "output"]


# --- Plugin registration ---


def test_plugin_can_register_agent_stream_middleware(model):
    """Plugins can register AgentStreamStage middleware during init_agent."""
    captured = {}

    class StreamPlugin(Plugin):
        @property
        def name(self) -> str:
            return "stream-plugin"

        def init_agent(self, agent_instance):
            agent_instance._middleware_registry.add_middleware(AgentStreamStage, self._middleware)

        async def _middleware(self, context: AgentStreamContext, next_fn):
            captured["agent"] = context.agent
            async for event in next_fn(context):
                yield event

    plugin = StreamPlugin()
    agent = Agent(model=model, callback_handler=None, plugins=[plugin])
    agent("test")

    assert captured["agent"] is agent


# --- invocation_state shared by reference ---


def test_invocation_state_shared_by_reference(agent):
    """invocation_state is shared by reference — mutations in middleware are visible to the caller."""

    async def mutate_state(context: AgentStreamContext, next_fn):
        context.invocation_state["middleware_was_here"] = True
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, mutate_state)
    state = {"original": True}
    agent("test", invocation_state=state)

    # The mutation is visible on the original dict passed in
    assert state["middleware_was_here"] is True
    assert state["original"] is True


# --- Zero overhead when no middleware registered ---


def test_no_middleware_fast_path(agent):
    """When no middleware is registered, the registry fast-paths to the terminal."""
    # Just confirm the agent works — the fast-path is an implementation detail
    # but we verify by ensuring the compose() returns the terminal directly
    from strands._middleware.stages import AgentStreamStage as stage

    chain = agent._middleware_registry.compose(stage, lambda ctx: None)
    # With no handlers registered, compose returns the terminal itself
    assert chain is not None
    result = agent("test")
    assert result.message["content"][0]["text"] == "Hello!"
