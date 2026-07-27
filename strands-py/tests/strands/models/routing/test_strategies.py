"""Tests for bundled routing strategies: FallbackStrategy (select-first) and ContextFitStrategy."""

import pytest

from strands.models.routing import ContextFitStrategy, FallbackStrategy, ModelRouter, RoutingCandidate, RoutingContext
from tests.fixtures.mocked_model_provider import MockedModelProvider


def _model():
    return MockedModelProvider([{"role": "assistant", "content": [{"text": "hi"}]}])


class _WindowModel(MockedModelProvider):
    """A model with a fixed context window and a fixed token-count estimate."""

    def __init__(self, window, tokens):
        super().__init__([{"role": "assistant", "content": [{"text": "hi"}]}])
        self._window = window
        self._tokens = tokens

    @property
    def context_window_limit(self):
        return self._window

    async def count_tokens(self, *args, **kwargs):
        return self._tokens


def _context(candidates):
    return RoutingContext(
        messages=[], system_prompt=None, tool_specs=[], candidates=candidates, invocation_state={}
    )


# --- FallbackStrategy ---


@pytest.mark.asyncio
async def test_fallback_strategy_selects_first_candidate():
    first = RoutingCandidate(_model(), name="first")
    second = RoutingCandidate(_model(), name="second")

    assert await FallbackStrategy().select(_context([first, second])) is first


# --- ContextFitStrategy ---


@pytest.mark.asyncio
async def test_context_fit_picks_smallest_window_that_fits():
    small = RoutingCandidate(_WindowModel(1_000, 5_000))
    medium = RoutingCandidate(_WindowModel(10_000, 5_000))
    large = RoutingCandidate(_WindowModel(100_000, 5_000))

    chosen = await ContextFitStrategy().select(_context([small, medium, large]))

    assert chosen is medium  # 5000 exceeds small's usable window; medium is the smallest that fits


@pytest.mark.asyncio
async def test_context_fit_falls_back_to_largest_when_none_fit():
    small = RoutingCandidate(_WindowModel(1_000, 5_000))
    medium = RoutingCandidate(_WindowModel(2_000, 5_000))

    chosen = await ContextFitStrategy().select(_context([small, medium]))

    assert chosen is medium  # none fit, so the largest window wins


@pytest.mark.asyncio
async def test_context_fit_treats_undeclared_window_as_shared_default():
    from strands.models.routing.strategies import DEFAULT_CONTEXT_WINDOW_LIMIT

    bounded = RoutingCandidate(_WindowModel(1_000, 5_000))
    undeclared = RoutingCandidate(_WindowModel(None, 5_000))  # -> DEFAULT_CONTEXT_WINDOW_LIMIT

    chosen = await ContextFitStrategy().select(_context([bounded, undeclared]))

    assert chosen is undeclared  # 5000 fits the 200k default window but not the 1000 one
    assert DEFAULT_CONTEXT_WINDOW_LIMIT == 200_000


@pytest.mark.asyncio
async def test_context_fit_threshold_reserves_room():
    tight = RoutingCandidate(_WindowModel(10_000, 8_500))
    big = RoutingCandidate(_WindowModel(100_000, 8_500))

    # At 0.7, tight allows 7000 tokens < 8500, so it does not fit.
    assert await ContextFitStrategy(threshold=0.7).select(_context([tight, big])) is big
    # At 0.9, tight allows 9000 >= 8500, so the smaller window wins.
    assert await ContextFitStrategy(threshold=0.9).select(_context([tight, big])) is tight


@pytest.mark.asyncio
async def test_context_fit_counts_tokens_per_candidate():
    lean = RoutingCandidate(_WindowModel(10_000, 5_000))
    verbose = RoutingCandidate(_WindowModel(10_000, 9_000))  # same window, tokenizes larger

    chosen = await ContextFitStrategy().select(_context([lean, verbose]))

    assert chosen is lean  # only the candidate whose own tokenizer fits is selected


@pytest.mark.asyncio
async def test_context_fit_treats_failed_count_as_not_fitting():
    class _RaisingModel(_WindowModel):
        async def count_tokens(self, *args, **kwargs):
            raise RuntimeError("count exploded")

    raising = RoutingCandidate(_RaisingModel(1_000_000, 0))  # big window, but counting fails
    healthy = RoutingCandidate(_WindowModel(10_000, 100))

    chosen = await ContextFitStrategy().select(_context([raising, healthy]))

    assert chosen is healthy  # the candidate whose count failed is not treated as a fit


@pytest.mark.asyncio
async def test_context_fit_uses_nested_router_default_window():
    small = RoutingCandidate(_WindowModel(1_000, 900))
    inner = ModelRouter(models=[_WindowModel(2_000, 900)])
    nested = RoutingCandidate(inner)

    chosen = await ContextFitStrategy().select(_context([small, nested]))

    assert chosen is nested  # small allows 700 < 900; the nested router's default window (2000) fits


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5])
def test_context_fit_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        ContextFitStrategy(threshold=threshold)


# --- ContextFitStrategy robustness (best-effort counting) ---


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_count", [None, "not-a-number", float("nan")])
async def test_context_fit_treats_bad_count_as_not_fitting(bad_count):
    class _BadCountModel(_WindowModel):
        async def count_tokens(self, *args, **kwargs):
            return bad_count

    bad = RoutingCandidate(_BadCountModel(1_000, 0))
    good = RoutingCandidate(_WindowModel(1_000, 100))

    assert await ContextFitStrategy().select(_context([bad, good])) is good


@pytest.mark.asyncio
async def test_context_fit_survives_a_window_read_error():
    class _BadWindowModel(_WindowModel):
        @property
        def context_window_limit(self):
            raise RuntimeError("config not initialised")

    only = RoutingCandidate(_BadWindowModel(None, 100))

    # Must not crash: the unreadable window falls back to the shared default.
    assert await ContextFitStrategy().select(_context([only])) is only


@pytest.mark.asyncio
async def test_context_fit_respects_an_explicit_zero_window():
    zero = RoutingCandidate(_WindowModel(0, 100))
    big = RoutingCandidate(_WindowModel(1_000, 100))

    # An explicit 0 window must not be silently replaced by the default; nothing fits 0, so big wins.
    assert await ContextFitStrategy(threshold=1.0).select(_context([zero, big])) is big
