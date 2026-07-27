"""Tests for ModelRouter core: candidate validation, strategy selection, guards."""

import types

import pytest

from strands import Agent, Plugin
from strands.event_loop._retry import ModelRetryStrategy
from strands.models import BedrockModel
from strands.models.routing import ModelRouter, RoutingCandidate, RoutingContext, RoutingStrategy
from strands.models.routing.router import _ROUTING_KEY
from strands.types.exceptions import ModelThrottledException
from tests.fixtures.mocked_model_provider import MockedModelProvider


class StatefulModel(MockedModelProvider):
    @property
    def stateful(self):
        return True


def _model(text="hi"):
    return MockedModelProvider([{"role": "assistant", "content": [{"text": text}]}])


class _PickByName:
    """Strategy that selects the candidate with a given name and counts its calls."""

    def __init__(self, name):
        self.name = name
        self.calls = 0

    async def select(self, context):
        self.calls += 1
        return next(candidate for candidate in context.candidates if candidate.name == self.name)


def _routing_context(candidates, invocation_state=None):
    return RoutingContext(
        messages=[],
        system_prompt=None,
        tool_specs=[],
        candidates=candidates,
        invocation_state=invocation_state if invocation_state is not None else {},
    )


def _invoke_context(invocation_state, model):
    return types.SimpleNamespace(
        messages=[], system_prompt=None, tool_specs=[], invocation_state=invocation_state, model=model
    )


# --- plugin identity ---


def test_router_is_a_plugin_with_stable_name():
    router = ModelRouter(models=[_model()])

    assert isinstance(router, Plugin)
    assert router.name == "strands:model-router"


# --- candidates + metadata ---


def test_routing_candidate_metadata_is_preserved():
    m = _model()
    router = ModelRouter(models=[RoutingCandidate(model=m, name="routine", description="simple tasks")])

    candidate = router.candidates[0]
    assert (candidate.model, candidate.name, candidate.description) == (m, "routine", "simple tasks")


def test_repeated_model_object_is_allowed():
    m = _model()
    router = ModelRouter(models=[m, m])

    assert router.default_model is m


def test_bedrock_model_object_is_a_valid_candidate():
    haiku = BedrockModel(model_id="haiku")
    router = ModelRouter(models=[haiku, BedrockModel(model_id="opus")])

    assert router.default_model is haiku


# --- default resolution (first candidate) ---


def test_default_model_is_first_candidate():
    m0, m1 = _model("0"), _model("1")
    router = ModelRouter(models=[m0, m1])

    assert router.default_model is m0


def test_nested_router_default_resolves_recursively():
    inner_model = _model()
    inner = ModelRouter(models=[inner_model])
    outer = ModelRouter(models=[inner, _model("x")])

    assert outer.default_model is inner_model


# --- strategy selection ---


@pytest.mark.asyncio
async def test_select_model_defaults_to_first_candidate():
    m0, m1 = _model(), _model()
    router = ModelRouter(models=[m0, m1])

    assert await router._select_model(_routing_context(router.candidates)) is m0


@pytest.mark.asyncio
async def test_custom_strategy_selects_named_candidate():
    fast, smart = _model(), _model()
    router = ModelRouter(
        models=[RoutingCandidate(fast, name="fast"), RoutingCandidate(smart, name="smart")],
        strategy=_PickByName("smart"),
    )

    assert await router._select_model(_routing_context(router.candidates)) is smart


@pytest.mark.asyncio
async def test_selection_recurses_into_nested_router_strategy():
    inner_fast, inner_smart = _model(), _model()
    inner = ModelRouter(
        models=[RoutingCandidate(inner_fast, name="if"), RoutingCandidate(inner_smart, name="is")],
        strategy=_PickByName("is"),
    )
    outer = ModelRouter(
        models=[_model(), RoutingCandidate(inner, name="inner")],
        strategy=_PickByName("inner"),
    )

    assert await outer._select_model(_routing_context(outer.candidates)) is inner_smart


def test_non_strategy_raises():
    with pytest.raises(TypeError, match="RoutingStrategy"):
        ModelRouter(models=[_model()], strategy=object())


def test_routing_strategy_protocol_is_runtime_checkable():
    assert isinstance(_PickByName("x"), RoutingStrategy)
    assert not isinstance(object(), RoutingStrategy)


@pytest.mark.asyncio
async def test_select_result_must_be_a_candidate():
    class _Rogue:
        async def select(self, context):
            return RoutingCandidate(_model())  # a fresh candidate, not one of context.candidates

    router = ModelRouter(models=[_model()], strategy=_Rogue())
    with pytest.raises(ValueError, match="one of the candidates"):
        await router._select_model(_routing_context(router.candidates))


# --- selection middleware ---


@pytest.mark.asyncio
async def test_selection_middleware_sets_model_and_caches_per_invocation():
    fast, smart = _model(), _model()
    strategy = _PickByName("smart")
    router = ModelRouter(
        models=[RoutingCandidate(fast, name="fast"), RoutingCandidate(smart, name="smart")], strategy=strategy
    )
    middleware = router._selection_middleware()
    state: dict = {}

    first = await middleware(_invoke_context(state, model=fast))
    assert first.model is smart
    assert strategy.calls == 1

    second = await middleware(_invoke_context(state, model=fast))
    assert second.model is smart
    assert strategy.calls == 1  # reused from invocation_state, not re-selected


@pytest.mark.asyncio
async def test_selection_middleware_reselects_for_new_invocation_state():
    fast, smart = _model(), _model()
    strategy = _PickByName("smart")
    router = ModelRouter(
        models=[RoutingCandidate(fast, name="fast"), RoutingCandidate(smart, name="smart")], strategy=strategy
    )
    middleware = router._selection_middleware()

    await middleware(_invoke_context({}, model=fast))
    await middleware(_invoke_context({}, model=fast))

    assert strategy.calls == 2


# --- guards ---


def test_empty_models_raises():
    with pytest.raises(ValueError, match="at least one"):
        ModelRouter(models=[])


def test_stateful_candidate_raises():
    with pytest.raises(ValueError, match=r"StatefulModel.*stateful"):
        ModelRouter(models=[StatefulModel([])])


def test_stateful_candidate_error_uses_name_when_present():
    with pytest.raises(ValueError, match=r"vip.*stateful"):
        ModelRouter(models=[RoutingCandidate(StatefulModel([]), name="vip")])


def test_duplicate_candidate_names_raise():
    with pytest.raises(ValueError, match="duplicate"):
        ModelRouter(models=[RoutingCandidate(_model(), name="a"), RoutingCandidate(_model(), name="a")])


def test_mapping_models_raises():
    with pytest.raises(TypeError, match="sequence of candidates"):
        ModelRouter(models={"cheap": _model()})


def test_bare_string_models_raises():
    with pytest.raises(TypeError, match="sequence of candidates"):
        ModelRouter(models="my-model-id")


def test_string_candidate_is_rejected():
    with pytest.raises(TypeError, match="candidate must be"):
        ModelRouter(models=["my-model-id"])


def test_invalid_candidate_raises():
    with pytest.raises(TypeError, match="candidate must be"):
        ModelRouter(models=[object()])


# --- agent integration ---


def test_agent_accepts_model_router_and_exposes_default():
    m = _model("routed")
    router = ModelRouter(models=[m])
    agent = Agent(model=router, callback_handler=None)

    assert agent.model is m
    assert agent._model_router is router


def test_agent_registers_router_as_plugin():
    router = ModelRouter(models=[_model()])
    agent = Agent(model=router, callback_handler=None)

    assert router.name in agent._plugin_registry._plugins


def test_router_via_plugins_is_rejected():
    router = ModelRouter(models=[_model()])
    with pytest.raises(ValueError, match=r"model=.*not plugins"):
        Agent(plugins=[router], callback_handler=None)


def test_agent_runs_with_default_first_candidate():
    router = ModelRouter(models=[_model("routed")])
    agent = Agent(model=router, callback_handler=None)

    result = agent("hello")

    assert result.message["content"][0]["text"] == "routed"


def test_agent_routes_to_strategy_selected_candidate():
    fast = _model("fast-says")
    smart = _model("smart-says")
    router = ModelRouter(
        models=[RoutingCandidate(fast, name="fast"), RoutingCandidate(smart, name="smart")],
        strategy=_PickByName("smart"),
    )
    agent = Agent(model=router, callback_handler=None)

    assert agent.model is fast  # default is still the first candidate

    result = agent("hello")

    assert result.message["content"][0]["text"] == "smart-says"  # strategy overrode the per-call model


class _ModelProbe(Plugin):
    """Plugin that records the ``context.model`` seen by a downstream Input middleware."""

    name = "test:model-probe"

    def __init__(self):
        super().__init__()
        self.seen = None

    def init_agent(self, agent):
        from strands._middleware.stages import InvokeModelStage

        def record(context):
            self.seen = context.model
            return context

        agent._middleware_registry.add_middleware(InvokeModelStage.Input, record)


def test_routing_runs_before_other_input_middleware():
    fast = _model("f")
    smart = _model("s")
    router = ModelRouter(
        models=[RoutingCandidate(fast, name="fast"), RoutingCandidate(smart, name="smart")],
        strategy=_PickByName("smart"),
    )
    probe = _ModelProbe()
    agent = Agent(model=router, plugins=[probe], callback_handler=None)

    agent("hello")

    assert probe.seen is smart  # routing set the per-call model before the probe middleware ran


# --- ordered fallback ---


class _FailingModel(MockedModelProvider):
    """A model whose stream always raises the given exception."""

    def __init__(self, exception):
        super().__init__([{"role": "assistant", "content": [{"text": "unused"}]}])
        self._exception = exception

    async def stream(self, *args, **kwargs):
        raise self._exception
        yield  # pragma: no cover - marks this an async generator


class _FlakyModel(MockedModelProvider):
    """A model that raises a throttling exception a set number of times, then streams normally."""

    def __init__(self, failures, text):
        super().__init__([{"role": "assistant", "content": [{"text": text}]}])
        self._remaining_failures = failures

    async def stream(self, *args, **kwargs):
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise ModelThrottledException("flaky")
        async for event in super().stream(*args, **kwargs):
            yield event


class _RaisingStrategy:
    """A strategy whose select always raises, to exercise fallback-resolution error containment."""

    async def select(self, context, **kwargs):
        raise RuntimeError("strategy boom")


def _agent_stub():
    """A minimal stand-in for the fields the fallback hook reads off the agent."""
    return types.SimpleNamespace(
        messages=[], system_prompt=None, tool_registry=types.SimpleNamespace(get_all_tool_specs=lambda: [])
    )


def test_fallback_advances_to_next_candidate_on_throttling():
    failing = _FailingModel(ModelThrottledException("throttled"))
    good = _model("recovered")
    router = ModelRouter(models=[failing, good])
    agent = Agent(model=router, retry_strategy=None, callback_handler=None)

    result = agent("hello")

    assert result.message["content"][0]["text"] == "recovered"


def test_fallback_advances_on_non_retryable_error():
    failing = _FailingModel(ValueError("boom"))
    good = _model("recovered")
    router = ModelRouter(models=[failing, good])
    agent = Agent(model=router, retry_strategy=None, callback_handler=None)

    result = agent("hello")

    assert result.message["content"][0]["text"] == "recovered"


def test_fallback_exhausts_all_candidates_then_raises():
    router = ModelRouter(
        models=[_FailingModel(ModelThrottledException("a")), _FailingModel(ModelThrottledException("b"))]
    )
    agent = Agent(model=router, retry_strategy=None, callback_handler=None)

    with pytest.raises(ModelThrottledException):
        agent("hello")


@pytest.mark.asyncio
async def test_advance_is_noop_without_routing_state():
    router = ModelRouter(models=[_model(), _model()])
    event = types.SimpleNamespace(
        retry=False, stop_response=None, exception=ValueError("x"), invocation_state={}, agent=None
    )

    await router._on_model_result(event)

    assert event.retry is False  # no selection was cached, so there is nothing to advance


def test_fallback_resets_retry_budget_so_next_candidate_gets_fresh_retries():
    # First candidate always fails; second needs two retries before it succeeds. This only passes if
    # advancing resets the retry budget so the second candidate gets its own attempts.
    first = _FailingModel(ModelThrottledException("down"))
    second = _FlakyModel(failures=2, text="recovered")
    router = ModelRouter(models=[first, second])
    retry_strategy = ModelRetryStrategy(max_attempts=3, initial_delay=0, max_delay=0)
    agent = Agent(model=router, retry_strategy=retry_strategy, callback_handler=None)

    result = agent("hello")

    assert result.message["content"][0]["text"] == "recovered"


# --- state scoping / lifecycle ---


@pytest.mark.asyncio
async def test_selection_middleware_reselects_when_state_belongs_to_another_router():
    m_a, m_b = _model(), _model()
    router_a = ModelRouter(models=[m_a])
    router_b = ModelRouter(models=[m_b])
    shared: dict = {}

    await router_a._selection_middleware()(_invoke_context(shared, model=None))
    context_b = _invoke_context(shared, model=None)
    await router_b._selection_middleware()(context_b)

    assert context_b.model is m_b  # router_b does not reuse router_a's cached selection


@pytest.mark.asyncio
async def test_clear_state_removes_only_own_routing_state():
    router = ModelRouter(models=[_model()])
    other = ModelRouter(models=[_model()])

    own = {"router": router, "index": 0, "model": _model(), "tried": {0}}
    invocation_state = {_ROUTING_KEY: own}
    await router._clear_state(types.SimpleNamespace(invocation_state=invocation_state))
    assert _ROUTING_KEY not in invocation_state

    foreign = {"router": other, "index": 0, "model": _model(), "tried": {0}}
    invocation_state = {_ROUTING_KEY: foreign}
    await router._clear_state(types.SimpleNamespace(invocation_state=invocation_state))
    assert _ROUTING_KEY in invocation_state  # another router's state is left untouched


@pytest.mark.asyncio
async def test_successful_call_rearms_the_fallback_chain():
    router = ModelRouter(models=[_model(), _model(), _model()])
    state = {"router": router, "index": 1, "model": _model(), "tried": {0, 1}}
    event = types.SimpleNamespace(
        retry=False, stop_response=object(), exception=None, invocation_state={_ROUTING_KEY: state}, agent=None
    )

    await router._on_model_result(event)

    assert state["tried"] == {1}  # re-armed to the current selection so a later failure can fall over


@pytest.mark.asyncio
async def test_fallback_resolution_error_is_contained():
    router = ModelRouter(
        models=[_model(), RoutingCandidate(ModelRouter([_model()], strategy=_RaisingStrategy()))]
    )
    state = {"router": router, "index": 0, "model": _model(), "tried": {0}}
    event = types.SimpleNamespace(
        retry=False,
        stop_response=None,
        exception=ValueError("original model error"),
        invocation_state={_ROUTING_KEY: state},
        agent=_agent_stub(),
    )

    await router._on_model_result(event)

    assert event.retry is False  # a failed advance degrades to "no fallback" instead of crashing


def test_nested_router_is_one_atomic_fallback_slot():
    inner = ModelRouter(models=[_FailingModel(ValueError("inner down")), _model("inner-second")])
    router = ModelRouter(models=[inner, _model("outer-other")])
    agent = Agent(model=router, retry_strategy=None, callback_handler=None)

    result = agent("hello")

    # The nested router's first pick fails; the outer router falls over to its own next candidate
    # rather than trying the nested router's second model.
    assert result.message["content"][0]["text"] == "outer-other"
