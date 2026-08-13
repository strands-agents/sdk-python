"""Tests for ContextFitStrategy and route-before-compress coordination."""

import copy
import math

import pytest

from strands import Agent
from strands.agent.conversation_manager.conversation_manager import ConversationManager
from strands.models._defaults import DEFAULT_CONTEXT_WINDOW_LIMIT
from strands.models.routing import (
    ContextFitStrategy,
    FallbackStrategy,
    ModelRouter,
    RoutingAttempt,
    RoutingContext,
)
from tests.fixtures.mocked_model_provider import MockedModelProvider


class _FitModel(MockedModelProvider):
    def __init__(self, tokens, limit, text="ok", error=None):
        super().__init__([{"role": "assistant", "content": [{"text": text}]}])
        self.tokens = tokens
        self.limit = limit
        self.error = error
        self.count_calls = []
        self.stream_messages = []

    def get_config(self):
        return {} if self.limit is None else {"context_window_limit": self.limit}

    async def count_tokens(self, messages, tool_specs=None, system_prompt=None, system_prompt_content=None):
        self.count_calls.append((copy.deepcopy(messages), tool_specs, system_prompt, system_prompt_content))
        if self.error is not None:
            raise self.error
        return self.tokens

    async def stream(self, messages, *args, **kwargs):
        self.stream_messages.append(copy.deepcopy(messages))
        async for event in super().stream(messages, *args, **kwargs):
            yield event


class _FailingFitModel(_FitModel):
    async def stream(self, *args, **kwargs):
        raise RuntimeError("model failed")
        yield  # pragma: no cover


class _Pick:
    def __init__(self, index):
        self.index = index
        self.calls = 0

    async def select(self, context, **kwargs):
        self.calls += 1
        return context.candidates[self.index]


class _RecordingManager(ConversationManager):
    def __init__(self):
        super().__init__(proactive_compression=True)
        self.reductions = 0

    def apply_management(self, agent, **kwargs):
        pass

    def reduce_context(self, agent, e=None, **kwargs):
        self.reductions += 1
        agent.messages[:] = agent.messages[-1:]


def _context(router, *, attempts=()):
    return RoutingContext(
        messages=[{"role": "user", "content": [{"text": "request"}]}],
        system_prompt="system",
        tool_specs=[{"name": "tool", "description": "", "inputSchema": {"json": {}}}],
        candidates=router.candidates,
        invocation_state={},
        attempts=attempts,
    )


@pytest.mark.asyncio
async def test_selects_smallest_context_window_that_fits():
    large = _FitModel(50, 1_000)
    small = _FitModel(50, 100)
    router = ModelRouter([large, small])

    selected = await ContextFitStrategy().select(_context(router))

    assert selected is router.candidates[1]


@pytest.mark.asyncio
async def test_uses_candidate_specific_token_counts_and_request_fields():
    too_full = _FitModel(80, 100)
    fits = _FitModel(60, 100)
    router = ModelRouter([too_full, fits])

    selected = await ContextFitStrategy().select(_context(router))

    assert selected is router.candidates[1]
    assert too_full.count_calls == fits.count_calls
    _, tool_specs, system_prompt, system_prompt_content = fits.count_calls[0]
    assert tool_specs == [{"name": "tool", "description": "", "inputSchema": {"json": {}}}]
    assert system_prompt == "system"
    assert system_prompt_content == [{"text": "system"}]


@pytest.mark.asyncio
async def test_exact_threshold_boundary_fits():
    model = _FitModel(70, 100)
    router = ModelRouter([model])

    assert await ContextFitStrategy().select(_context(router)) is router.candidates[0]


@pytest.mark.asyncio
async def test_selects_largest_window_when_every_measurement_is_known_and_none_fits():
    small = _FitModel(100, 50)
    large = _FitModel(100, 80)
    router = ModelRouter([small, large])

    assert await ContextFitStrategy().select(_context(router)) is router.candidates[1]


@pytest.mark.asyncio
async def test_missing_window_uses_shared_default():
    defaulted = _FitModel(DEFAULT_CONTEXT_WINDOW_LIMIT * 0.7, None)
    larger = _FitModel(DEFAULT_CONTEXT_WINDOW_LIMIT * 0.7, 300_000)
    router = ModelRouter([defaulted, larger])

    assert await ContextFitStrategy().select(_context(router)) is router.candidates[0]


@pytest.mark.asyncio
async def test_equal_fitting_windows_preserve_declaration_order():
    first = _FitModel(10, 100)
    second = _FitModel(10, 100)
    router = ModelRouter([first, second])

    assert await ContextFitStrategy().select(_context(router)) is router.candidates[0]


@pytest.mark.asyncio
async def test_zero_context_window_is_measured_instead_of_replaced_by_default():
    zero = _FitModel(1, 0)
    positive = _FitModel(1, 2)
    router = ModelRouter([zero, positive])

    assert await ContextFitStrategy(threshold=1).select(_context(router)) is router.candidates[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("tokens", [-1, math.nan, math.inf, "invalid"])
async def test_invalid_token_count_is_unknown(tokens):
    unknown = _FitModel(tokens, 100)
    no_fit = _FitModel(100, 100)
    router = ModelRouter([unknown, no_fit])

    assert await ContextFitStrategy().select(_context(router)) is router.candidates[0]


@pytest.mark.asyncio
async def test_count_failure_is_unknown_but_known_fit_wins():
    unknown = _FitModel(0, 100, error=RuntimeError("count failed"))
    fitting = _FitModel(10, 100)
    router = ModelRouter([unknown, fitting])

    assert await ContextFitStrategy().select(_context(router)) is router.candidates[1]


@pytest.mark.asyncio
async def test_nested_router_is_unknown_and_is_not_resolved_by_strategy():
    class _RaisesIfAsked:
        async def select(self, context, **kwargs):
            raise AssertionError("nested router must not be resolved")

    nested = ModelRouter([_FitModel(1, 100)], strategy=_RaisesIfAsked())
    no_fit = _FitModel(100, 100)
    outer = ModelRouter([nested, no_fit])

    assert await ContextFitStrategy().select(_context(outer)) is outer.candidates[0]


@pytest.mark.asyncio
async def test_returns_none_after_failure_without_explicit_fallback():
    router = ModelRouter([_FitModel(1, 100), _FitModel(1, 200)])
    attempts = (RoutingAttempt(router.candidates[0], RuntimeError("failed")),)

    assert await ContextFitStrategy().select(_context(router, attempts=attempts)) is None


@pytest.mark.asyncio
async def test_delegates_failure_selection_to_explicit_fallback():
    router = ModelRouter([_FitModel(1, 100), _FitModel(1, 200)])
    attempts = (RoutingAttempt(router.candidates[0], RuntimeError("failed")),)

    selected = await ContextFitStrategy(fallback=FallbackStrategy()).select(_context(router, attempts=attempts))

    assert selected is router.candidates[1]


@pytest.mark.parametrize("threshold", [0, -0.1, 1.1, math.nan, math.inf])
def test_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold must be between"):
        ContextFitStrategy(threshold=threshold)


def test_is_re_exported_from_models_packages():
    import strands.models as models

    assert models.ContextFitStrategy is ContextFitStrategy
    assert models.routing.ContextFitStrategy is ContextFitStrategy
    assert "ContextFitStrategy" in models.__all__
    assert "ContextFitStrategy" in models.routing.__all__


def test_context_fit_routes_to_larger_model_before_compression():
    small = _FitModel(80, 100, text="small")
    large = _FitModel(80, 1_000, text="large")
    manager = _RecordingManager()
    agent = Agent(
        model=ModelRouter([small, large], strategy=ContextFitStrategy()),
        conversation_manager=manager,
        callback_handler=None,
    )

    result = agent("request")

    assert result.message["content"][0]["text"] == "large"
    assert manager.reductions == 0
    assert not small.stream_messages
    assert large.stream_messages


def test_compression_uses_selected_model_and_refreshes_terminal_messages():
    selected = _FitModel(80, 100, text="selected")
    other = _FitModel(1, 1_000, text="other")
    strategy = _Pick(0)
    manager = _RecordingManager()
    agent = Agent(
        model=ModelRouter([selected, other], strategy=strategy),
        messages=[
            {"role": "user", "content": [{"text": "old"}]},
            {"role": "assistant", "content": [{"text": "old response"}]},
        ],
        conversation_manager=manager,
        callback_handler=None,
    )

    agent("current")

    assert strategy.calls == 1
    assert manager.reductions == 1
    assert selected.stream_messages == [[{"role": "user", "content": [{"text": "current"}]}]]


def test_nested_selection_compresses_against_final_leaf_model():
    small = _FitModel(1, 100, text="small")
    large = _FitModel(800, 1_000, text="large")
    nested_strategy = _Pick(1)
    nested = ModelRouter([small, large], strategy=nested_strategy)
    outer_strategy = _Pick(0)
    manager = _RecordingManager()
    agent = Agent(
        model=ModelRouter([nested], strategy=outer_strategy),
        conversation_manager=manager,
        callback_handler=None,
    )

    result = agent("request")

    assert result.message["content"][0]["text"] == "large"
    assert outer_strategy.calls == nested_strategy.calls == 1
    assert manager.reductions == 1


def test_fallback_retry_compresses_against_newly_selected_model():
    failing = _FailingFitModel(50, 100)
    replacement = _FitModel(800, 1_000, text="replacement")
    manager = _RecordingManager()
    agent = Agent(
        model=ModelRouter(
            [failing, replacement],
            strategy=ContextFitStrategy(fallback=FallbackStrategy()),
        ),
        conversation_manager=manager,
        callback_handler=None,
        retry_strategy=None,
    )

    result = agent("request")

    assert result.message["content"][0]["text"] == "replacement"
    assert manager.reductions == 1
    assert replacement.stream_messages
