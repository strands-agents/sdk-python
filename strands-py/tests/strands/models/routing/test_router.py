"""Tests for ModelRouter core: candidate validation, strategy selection, guards."""

import asyncio
import contextlib
import types

import pytest

from strands import Agent, Plugin
from strands.event_loop._retry import ModelRetryStrategy
from strands.models import BedrockModel
from strands.models.routing import (
    FallbackStrategy,
    ModelRouter,
    RoutingAttempt,
    RoutingCandidate,
    RoutingContext,
    RoutingStrategy,
)
from strands.models.routing.router import _RoutingState
from strands.multiagent import GraphBuilder
from strands.types.exceptions import ModelThrottledException
from tests.fixtures.mocked_model_provider import MockedModelProvider


class StatefulModel(MockedModelProvider):
    @property
    def stateful(self):
        return True


def _model(text="hi"):
    return MockedModelProvider([{"role": "assistant", "content": [{"text": text}]}])


class _PreferByName:
    """Strategy that puts named candidates first and counts its calls."""

    def __init__(self, *names):
        self.names = names
        self.calls = 0

    async def select(self, context):
        self.calls += 1
        by_name = {candidate.name: candidate for candidate in context.candidates}
        ordered = [by_name[name] for name in self.names]
        ordered += [c for c in context.candidates if c.name not in self.names]
        tried = {id(attempt.candidate) for attempt in context.attempts}
        return next((c for c in ordered if id(c) not in tried), None)


def _routing_context(candidates, invocation_state=None, attempts=()):
    return RoutingContext(
        messages=[],
        system_prompt=None,
        tool_specs=[],
        candidates=candidates,
        invocation_state=invocation_state if invocation_state is not None else {},
        attempts=attempts,
    )


_TEST_AGENT = object()


def _invoke_context(invocation_state, model, agent=None):
    return types.SimpleNamespace(
        agent=agent if agent is not None else _TEST_AGENT,
        messages=[],
        system_prompt=None,
        tool_specs=[],
        invocation_state=invocation_state,
        model=model,
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


@pytest.mark.parametrize(
    "build",
    [
        lambda m: ([m, _model("other")], m),
        lambda m: ([m, m], m),  # the same model object twice is allowed
        lambda m: ([BedrockModel(model_id="haiku"), m], None),
        lambda m: ([ModelRouter(models=[m]), _model("other")], m),  # nested resolves recursively
    ],
    ids=["first-candidate", "repeated-model", "bedrock-model", "nested-router"],
)
def test_default_model_is_the_first_declared_candidate(build):
    expected_first = _model("first")
    models, expected = build(expected_first)

    default = ModelRouter(models=models).default_model

    assert default is (expected if expected is not None else models[0])


# --- strategy selection ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("count", "history", "expected"),
    [
        (2, lambda c, e: (), 0),
        (2, lambda c, e: (RoutingAttempt(c[0], e),), 1),
        (2, lambda c, e: (RoutingAttempt(c[0], e), RoutingAttempt(c[1], e)), None),
        # A success clears the failures before it, so the earlier candidate is eligible again.
        (2, lambda c, e: (RoutingAttempt(c[0], e), RoutingAttempt(c[1]), RoutingAttempt(c[1], e)), 0),
        # With an untouched candidate available, the one carrying a failure is demoted below it.
        (3, lambda c, e: (RoutingAttempt(c[0], e), RoutingAttempt(c[1]), RoutingAttempt(c[1], e)), 2),
    ],
    ids=["opening-choice", "advances", "exhausted", "rearms-after-success", "prefers-least-failed"],
)
async def test_fallback_strategy_decides_from_the_attempt_log(count, history, expected):
    router = ModelRouter(models=[_model() for _ in range(count)])
    attempts = history(router.candidates, ValueError("down"))

    chosen = await FallbackStrategy().select(_routing_context(router.candidates, attempts=attempts))

    assert chosen is (None if expected is None else router.candidates[expected])


@pytest.mark.asyncio
async def test_custom_strategy_prefers_named_candidate():
    fast, smart = _model(), _model()
    router = ModelRouter(
        models=[RoutingCandidate(fast, name="fast"), RoutingCandidate(smart, name="smart")],
        strategy=_PreferByName("smart"),
    )

    assert await router._select_model(_routing_context(router.candidates)) is smart


@pytest.mark.asyncio
async def test_selection_recurses_into_nested_router_strategy():
    inner_fast, inner_smart = _model(), _model()
    inner = ModelRouter(
        models=[RoutingCandidate(inner_fast, name="if"), RoutingCandidate(inner_smart, name="is")],
        strategy=_PreferByName("is"),
    )
    outer = ModelRouter(
        models=[_model(), RoutingCandidate(inner, name="inner")],
        strategy=_PreferByName("inner"),
    )

    assert await outer._select_model(_routing_context(outer.candidates)) is inner_smart


class _SyncSelect:
    def select(self, context):  # not a coroutine function
        return None


class _ExtraMembers:
    """Members the protocol does not declare must not disqualify a strategy."""

    version = 2

    async def select(self, context):
        return context.candidates[0]


@pytest.mark.parametrize(
    "strategy",
    [object(), _SyncSelect(), types.SimpleNamespace(select="not a method")],
    ids=["no-select", "sync-select", "select-not-callable"],
)
def test_construction_rejects_a_strategy_without_an_async_select(strategy):
    with pytest.raises(TypeError, match="async select"):
        ModelRouter(models=[_model()], strategy=strategy)


def test_construction_requires_only_select_so_extra_members_are_allowed():
    assert ModelRouter(models=[_model()], strategy=_ExtraMembers())
    assert isinstance(_ExtraMembers(), RoutingStrategy)


@pytest.mark.asyncio
async def test_contract_violation_surfaces_instead_of_degrading():
    # A wrong return type is a bug in the strategy, not an outage to route around.
    class _Wrong:
        async def select(self, context):
            return "cheap"

    router = ModelRouter(models=[_model(), _model()], strategy=_Wrong())

    with pytest.raises(TypeError, match="RoutingCandidate or None"):
        await router._selection_middleware()(_invoke_context({}, model=None))


@pytest.mark.asyncio
async def test_nested_strategy_is_asked_without_the_outer_routers_attempts():
    seen = []

    class _RecordsAttempts:
        async def select(self, context):
            seen.append(tuple(context.attempts))
            return context.candidates[0]

    inner = ModelRouter(models=[_model("inner")], strategy=_RecordsAttempts())
    router = ModelRouter(models=[_FailingModel(ValueError("outer down")), RoutingCandidate(inner, name="inner")])
    agent = Agent(model=router, retry_strategy=None, callback_handler=None)

    result = agent("hello")

    assert result.message["content"][0]["text"] == "inner"
    # The outer router had one failed attempt when it resolved the nested router.
    assert seen == [()]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("returns", "exc", "match"),
    [
        (lambda c: list(c), TypeError, "RoutingCandidate or None; got list"),
        # A judge naming a candidate by string is the likeliest mistake.
        (lambda c: "cheap", TypeError, "RoutingCandidate or None; got str"),
        (lambda c: RoutingCandidate(_model()), ValueError, "from context.candidates"),
    ],
    ids=["list", "string", "foreign-candidate"],
)
async def test_strategy_selection_rejects_unusable_results(returns, exc, match):
    class _InvalidSelection:
        async def select(self, context):
            return returns(context.candidates)

    router = ModelRouter(models=[_model(), _model()], strategy=_InvalidSelection())

    with pytest.raises(exc, match=match):
        await router._select_model(_routing_context(router.candidates))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "strategy_body",
    [lambda context: None, lambda context: (_ for _ in ()).throw(RuntimeError("judge timed out"))],
    ids=["declines", "raises"],
)
async def test_opening_choice_degrades_to_the_default_candidate(strategy_body):
    # An expensive strategy can time out or answer unparseably; the request is still servable.
    default, other = _model("default"), _model("other")

    class _Unhelpful:
        async def select(self, context):
            return strategy_body(context)

    router = ModelRouter(models=[default, other], strategy=_Unhelpful())
    agent = Agent(model=router, callback_handler=None)

    assert agent("hello").message["content"][0]["text"] == "default"


def test_strategy_that_declines_to_reconsider_gets_no_fallback():
    # Returning None after the first choice keeps the invocation on one model: the error surfaces
    # even though a healthy candidate is configured.
    chosen = _FailingModel(ValueError("chosen model down"))
    healthy = _model("should-not-be-used")

    class _PickChosen:
        async def select(self, context):
            if context.attempts:
                return None
            return next(c for c in context.candidates if c.name == "chosen")

    router = ModelRouter(
        models=[RoutingCandidate(chosen, name="chosen"), RoutingCandidate(healthy, name="healthy")],
        strategy=_PickChosen(),
    )
    agent = Agent(model=router, retry_strategy=None, callback_handler=None)

    with pytest.raises(ValueError, match="chosen model down"):
        agent("hello")


# --- selection middleware ---


@pytest.mark.asyncio
async def test_selection_middleware_sets_model_and_caches_per_invocation_state():
    fast, smart = _model(), _model()
    strategy = _PreferByName("smart")
    router = ModelRouter(
        models=[RoutingCandidate(fast, name="fast"), RoutingCandidate(smart, name="smart")], strategy=strategy
    )
    middleware = router._selection_middleware()

    # Same invocation_state: selects once, then reuses the cached model.
    state: dict = {}
    assert (await middleware(_invoke_context(state, model=fast))).model is smart
    assert (await middleware(_invoke_context(state, model=fast))).model is smart
    assert strategy.calls == 1

    # A fresh invocation_state selects again.
    assert (await middleware(_invoke_context({}, model=fast))).model is smart
    assert strategy.calls == 2


@pytest.mark.asyncio
async def test_strategy_cannot_mutate_the_request_it_is_asked_about():
    class _Mutating:
        async def select(self, context, **kwargs):
            context.messages.append({"role": "user", "content": [{"text": "injected"}]})
            context.messages[0]["content"][0]["text"] = "rewritten"
            return context.candidates[0]

    m = _model()
    router = ModelRouter(models=[m], strategy=_Mutating())
    context = _invoke_context({}, model=m)
    context.messages = [{"role": "user", "content": [{"text": "original"}]}]

    await router._selection_middleware()(context)

    assert context.messages == [{"role": "user", "content": [{"text": "original"}]}]


# --- construction guards ---


@pytest.mark.parametrize(
    ("make_models", "exc", "match"),
    [
        (lambda: [], ValueError, "at least one"),
        (lambda: "my-model-id", TypeError, "sequence of candidates"),
        (lambda: {"cheap": _model()}, TypeError, "sequence of candidates"),
        (lambda: ["my-model-id"], TypeError, "candidate must be"),
        (lambda: [object()], TypeError, "candidate must be"),
        (lambda: [StatefulModel([])], ValueError, r"StatefulModel.*stateful"),
        (lambda: [RoutingCandidate(StatefulModel([]), name="vip")], ValueError, r"vip.*stateful"),
        (
            lambda: [RoutingCandidate(_model(), name="a"), RoutingCandidate(_model(), name="a")],
            ValueError,
            "duplicate candidate name",
        ),
        (lambda: [RoutingCandidate(_model())] * 2, ValueError, "duplicate RoutingCandidate instance"),
    ],
    ids=[
        "empty",
        "bare-string",
        "mapping",
        "string-candidate",
        "invalid-object",
        "stateful",
        "stateful-uses-name-in-error",
        "duplicate-name",
        "duplicate-instance",
    ],
)
def test_construction_rejects_invalid_input(make_models, exc, match):
    with pytest.raises(exc, match=match):
        ModelRouter(models=make_models())


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


def test_agent_routes_to_strategy_selected_candidate():
    fast = _model("fast-says")
    smart = _model("smart-says")
    router = ModelRouter(
        models=[RoutingCandidate(fast, name="fast"), RoutingCandidate(smart, name="smart")],
        strategy=_PreferByName("smart"),
    )
    agent = Agent(model=router, callback_handler=None)

    assert agent.model is fast  # default is still the first candidate

    result = agent("hello")

    assert result.message["content"][0]["text"] == "smart-says"  # strategy overrode the per-call model


def test_one_agent_routes_each_invocation_on_its_own_request():
    # The right model depends on the request, so a later hard turn must not inherit the model an
    # earlier easy turn selected.
    cheap, frontier = _model("cheap-said"), _model("frontier-said")

    class _ByLatestTurn:
        def __init__(self):
            self.seen = []

        async def select(self, context):
            latest = str(context.messages[-1]) if context.messages else ""
            wanted = "frontier" if "research" in latest else "cheap"
            self.seen.append(wanted)
            return next(c for c in context.candidates if c.name == wanted)

    strategy = _ByLatestTurn()
    router = ModelRouter(
        models=[RoutingCandidate(cheap, name="cheap"), RoutingCandidate(frontier, name="frontier")],
        strategy=strategy,
    )
    agent = Agent(model=router, callback_handler=None)

    easy = agent("what is 1+1")
    hard = agent("do a deep research report on this")

    assert easy.message["content"][0]["text"] == "cheap-said"
    assert hard.message["content"][0]["text"] == "frontier-said"
    assert strategy.seen == ["cheap", "frontier"]


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
        strategy=_PreferByName("smart"),
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
    """A strategy whose select always raises, to exercise resolution error containment."""

    async def select(self, context, **kwargs):
        raise RuntimeError("strategy boom")


def _agent_stub():
    """A minimal stand-in for the fields the fallback hook reads off the agent."""
    return types.SimpleNamespace(
        messages=[],
        system_prompt=None,
        _system_prompt_content=None,
        tool_registry=types.SimpleNamespace(get_all_tool_specs=lambda: []),
        _retry_strategy=ModelRetryStrategy(max_attempts=1),
    )


def _hook_scaffold(router, *, candidate_index=0, model=None):
    """An agent stub plus seeded routing state, as the fallback hook expects to find them."""
    agent = _agent_stub()
    candidate = router.candidates[candidate_index]
    state = _RoutingState(candidate=candidate, model=model if model is not None else candidate.model)
    return agent, state, {router._state_key(agent): state}


def _model_result(invocation_state, agent, *, error=None):
    """An AfterModelCallEvent stand-in; ``error=None`` means the call succeeded."""
    return types.SimpleNamespace(
        retry=False,
        stop_response=None if error else object(),
        exception=error,
        invocation_state=invocation_state,
        agent=agent,
    )


@pytest.mark.parametrize(
    "exception",
    [ModelThrottledException("throttled"), ValueError("boom")],
    ids=["retryable", "non-retryable"],
)
def test_fallback_advances_past_a_failing_candidate(exception):
    good = _model("recovered")
    router = ModelRouter(models=[_FailingModel(exception), good])
    agent = Agent(model=router, retry_strategy=None, callback_handler=None)

    result = agent("hello")

    assert result.message["content"][0]["text"] == "recovered"


def test_strategy_controls_both_initial_choice_and_fallback_order():
    declared_first = _model("declared-first")
    prioritized = _FailingModel(ValueError("priority failed"))
    fallback = _model("strategy-fallback")
    router = ModelRouter(
        models=[
            RoutingCandidate(declared_first, name="declared-first"),
            RoutingCandidate(prioritized, name="priority"),
            RoutingCandidate(fallback, name="fallback"),
        ],
        strategy=_PreferByName("priority", "fallback"),
    )
    agent = Agent(model=router, retry_strategy=None, callback_handler=None)

    result = agent("hello")

    assert result.message["content"][0]["text"] == "strategy-fallback"


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
async def test_two_routers_on_one_agent_keep_separate_state():
    m_a, m_b = _model(), _model()
    router_a = ModelRouter(models=[m_a])
    router_b = ModelRouter(models=[m_b])
    shared: dict = {}

    context_a = _invoke_context(shared, model=None)
    await router_a._selection_middleware()(context_a)
    context_b = _invoke_context(shared, model=None)
    await router_b._selection_middleware()(context_b)

    assert (context_a.model, context_b.model) == (m_a, m_b)
    # Each router owns its own slot, so router_b's selection did not evict router_a's.
    assert (await router_a._selection_middleware()(_invoke_context(shared, model=None))).model is m_a


@pytest.mark.asyncio
async def test_two_agents_sharing_one_router_and_state_dict_route_independently():
    # Graph hands one invocation_state to every node, so a shared router must not let one agent
    # run on another agent's cached selection.
    fast, smart = _model(), _model()
    router = ModelRouter(
        models=[RoutingCandidate(fast, name="fast"), RoutingCandidate(smart, name="smart")],
        strategy=_PreferByName("smart"),
    )
    shared: dict = {}
    agent_one, agent_two = object(), object()

    context_one = _invoke_context(shared, model=None, agent=agent_one)
    await router._selection_middleware()(context_one)
    context_two = _invoke_context(shared, model=None, agent=agent_two)
    await router._selection_middleware()(context_two)

    assert (context_one.model, context_two.model) == (smart, smart)
    # Two distinct slots, so neither agent can switch or clear the other's model.
    assert len([key for key in shared if key.startswith("strands:model_routing")]) == 2

    await router._clear_state(types.SimpleNamespace(invocation_state=shared, agent=agent_one))
    assert [key for key in shared if key.startswith("strands:model_routing")] == [router._state_key(agent_two)]


@pytest.mark.asyncio
async def test_clear_state_removes_only_this_agents_routing_state():
    router = ModelRouter(models=[_model()])
    mine, theirs = object(), object()
    invocation_state = {
        router._state_key(mine): _RoutingState(candidate=router.candidates[0], model=router.default_model),
        router._state_key(theirs): _RoutingState(candidate=router.candidates[0], model=router.default_model),
    }

    await router._clear_state(types.SimpleNamespace(invocation_state=invocation_state, agent=mine))

    assert list(invocation_state) == [router._state_key(theirs)]

    # An agent that never selected has nothing to clear.
    await router._clear_state(types.SimpleNamespace(invocation_state=invocation_state, agent=object()))
    assert list(invocation_state) == [router._state_key(theirs)]


def test_router_does_not_clobber_caller_invocation_state():
    router = ModelRouter(models=[_model("ok")])
    agent = Agent(model=router, callback_handler=None)
    state = {"model_routing": "caller-owned", "keep": 1}

    agent("hi", invocation_state=state)

    assert state["model_routing"] == "caller-owned"
    assert state["keep"] == 1
    # Routing state is cleared at the end of the invocation.
    assert [key for key in state if key.startswith("strands:model_routing")] == []


@pytest.mark.asyncio
async def test_router_records_each_outcome_for_the_strategy():
    # The router keeps the attempt log the strategy reads; it does not interpret it.
    router = ModelRouter(models=[_model("first"), _model("second")])
    agent, state, invocation_state = _hook_scaffold(router)

    failure = _model_result(invocation_state, agent, error=ValueError("down"))
    await router._on_model_result(failure)

    assert failure.retry is True
    assert state.candidate is router.candidates[1]  # FallbackStrategy moved on
    assert [(a.candidate, type(a.error)) for a in state.attempts] == [(router.candidates[0], ValueError)]

    await router._on_model_result(_model_result(invocation_state, agent))

    assert state.attempts[-1] == RoutingAttempt(router.candidates[1], None)


@pytest.mark.asyncio
async def test_fallback_cycles_back_to_a_failed_candidate_in_the_next_round():
    router = ModelRouter(models=[_model("first"), _model("second")])
    agent, state, invocation_state = _hook_scaffold(router)

    await router._on_model_result(_model_result(invocation_state, agent, error=ValueError("down")))
    assert state.candidate is router.candidates[1]

    await router._on_model_result(_model_result(invocation_state, agent))

    # The success re-arms the first candidate, so a later failure returns to it.
    event = _model_result(invocation_state, agent, error=ValueError("down"))
    await router._on_model_result(event)

    assert state.candidate is router.candidates[0]
    assert state.model is router.candidates[0].model
    assert event.retry is True


@pytest.mark.asyncio
async def test_repeatedly_failing_candidate_is_demoted_below_healthy_ones():
    # A candidate that keeps failing sinks below healthy ones, bounding what re-arming costs in a
    # long tool loop.
    dead, live, spare = _model("dead"), _model("live"), _model("spare")
    router = ModelRouter(
        models=[
            RoutingCandidate(dead, name="dead"),
            RoutingCandidate(live, name="live"),
            RoutingCandidate(spare, name="spare"),
        ]
    )
    agent, state, invocation_state = _hook_scaffold(router, model=dead)

    await router._on_model_result(_model_result(invocation_state, agent, error=ValueError("down")))
    assert state.model is live
    await router._on_model_result(_model_result(invocation_state, agent))

    # live blips: declaration order would return to dead, but dead has the most failures.
    await router._on_model_result(_model_result(invocation_state, agent, error=ValueError("down")))

    assert state.model is spare


@pytest.mark.asyncio
async def test_a_strategy_failing_mid_invocation_leaves_the_model_error_to_surface():
    # Degrading again here would hide the model failure behind a routing failure.
    router = ModelRouter(models=[_model(), _model()], strategy=_RaisingStrategy())
    agent, state, _ = _hook_scaffold(router, model=router.default_model)
    event = _model_result({router._state_key(agent): state}, agent, error=ValueError("model down"))

    await router._on_model_result(event)

    assert event.retry is False


def test_a_strategy_re_offering_the_failed_candidate_terminates():
    # Invariant: every invocation ends. Re-offering the failed candidate must not reset the retry
    # budget, or "always prefer my primary" loops forever with the default max_switches=None.
    primary = _FailingModel(ValueError("primary down"))
    backup = _model("backup")

    class _AlwaysFirst:
        def __init__(self):
            self.asks = 0

        async def select(self, context):
            self.asks += 1
            return context.candidates[0]

    strategy = _AlwaysFirst()
    router = ModelRouter(
        models=[RoutingCandidate(primary, name="primary"), RoutingCandidate(backup, name="backup")],
        strategy=strategy,
    )
    agent = Agent(
        model=router,
        retry_strategy=ModelRetryStrategy(max_attempts=2, initial_delay=0, max_delay=0),
        callback_handler=None,
    )

    with pytest.raises(ValueError, match="primary down"):
        agent("hello")

    assert strategy.asks < 10  # bounded, not spinning


@pytest.mark.parametrize(("cap", "expected"), [(0, "first down"), (1, "second down")], ids=["cap-0", "cap-1"])
def test_max_switches_allows_exactly_that_many_switches(cap, expected):
    router = ModelRouter(
        models=[
            _FailingModel(ValueError("first down")),
            _FailingModel(ValueError("second down")),
            _model("healthy"),
        ],
        max_switches=cap,
    )
    agent = Agent(model=router, retry_strategy=None, callback_handler=None)

    with pytest.raises(ValueError, match=expected):
        agent("hello")


def test_routing_state_does_not_span_invocations_in_a_reused_dict():
    # Invariant: per-invocation state never outlives its invocation, even if teardown is skipped.
    cheap, premium = _model("cheap"), _model("premium")

    class _ByTurnCount:
        async def select(self, context):
            wanted = "premium" if len(context.messages) > 1 else "cheap"
            return next(c for c in context.candidates if c.name == wanted)

    router = ModelRouter(
        models=[RoutingCandidate(cheap, name="cheap"), RoutingCandidate(premium, name="premium")],
        strategy=_ByTurnCount(),
    )
    agent = Agent(model=router, callback_handler=None)
    shared: dict = {}

    first = agent("one", invocation_state=shared)
    # Simulate a skipped teardown by re-seeding the slot the router would have cleared.
    shared[router._state_key(agent)] = _RoutingState(candidate=router.candidates[0], model=cheap)
    second = agent("two", invocation_state=shared)

    assert first.message["content"][0]["text"] == "cheap"
    assert second.message["content"][0]["text"] == "premium"


def test_nested_router_contributes_one_candidate_without_internal_failover():
    inner_down, inner_spare = _FailingModel(ValueError("inner down")), _model("inner-spare")
    inner = ModelRouter(models=[inner_down, inner_spare])
    outer_backup = _model("outer-backup")
    agent = Agent(
        model=ModelRouter(models=[RoutingCandidate(inner, name="inner"), outer_backup]),
        retry_strategy=None,
        callback_handler=None,
    )

    result = agent("hello")

    # The outer router leaves the whole nested candidate rather than advancing to inner_spare.
    assert result.message["content"][0]["text"] == "outer-backup"


def test_a_sync_select_is_rejected_at_construction():
    class _Sync:
        def select(self, context):
            return None

    with pytest.raises(TypeError, match="async select"):
        ModelRouter(models=[_model()], strategy=_Sync())


def test_negative_max_switches_is_rejected():
    with pytest.raises(ValueError, match="max_switches must be zero or greater"):
        ModelRouter(models=[_model()], max_switches=-1)


@pytest.mark.asyncio
async def test_advance_stops_when_every_candidate_named_is_unresolvable():
    # A strategy that keeps naming a candidate the router cannot resolve must not spin.
    broken = RoutingCandidate(ModelRouter([_model()], strategy=_RaisingStrategy()), name="broken")

    class _Insists:
        async def select(self, context):
            return next(c for c in context.candidates if c.name == "broken")

    router = ModelRouter(models=[RoutingCandidate(_model(), name="healthy"), broken], strategy=_Insists())
    agent, state, _ = _hook_scaffold(router, model=router.default_model)
    event = _model_result({router._state_key(agent): state}, agent, error=ValueError("original model error"))

    await router._on_model_result(event)

    assert event.retry is False


@pytest.mark.asyncio
async def test_fallback_resolution_error_is_contained():
    router = ModelRouter(models=[_model(), RoutingCandidate(ModelRouter([_model()], strategy=_RaisingStrategy()))])
    agent, state, _ = _hook_scaffold(router, model=router.default_model)
    event = _model_result({router._state_key(agent): state}, agent, error=ValueError("original model error"))

    await router._on_model_result(event)

    assert event.retry is False  # an unresolvable candidate degrades to "no fallback", not a crash


def test_fallback_skips_an_unresolvable_candidate_and_reaches_a_healthy_one():
    # [0] fails on call, [1] is a nested router whose strategy raises during resolution, [2] is
    # healthy. A resolution failure on [1] must skip to [2], not abandon the chain.
    unresolvable = RoutingCandidate(ModelRouter([_model()], strategy=_RaisingStrategy()))
    router = ModelRouter(models=[_FailingModel(ValueError("primary down")), unresolvable, _model("healthy")])
    agent = Agent(model=router, retry_strategy=None, callback_handler=None)

    result = agent("hello")

    assert result.message["content"][0]["text"] == "healthy"


@pytest.mark.parametrize("declining_first", [True, False], ids=["first-slot", "second-slot"])
def test_an_unresolvable_nested_router_is_skipped_wherever_it_is_declared(declining_first):
    class _Declines:
        async def select(self, context):
            return None

    inner = ModelRouter(models=[_model("inner")], strategy=_Declines())
    slots = [RoutingCandidate(inner, name="inner"), RoutingCandidate(_model("healthy"), name="healthy")]
    if not declining_first:
        slots.reverse()
    agent = Agent(model=ModelRouter(models=slots), retry_strategy=None, callback_handler=None)

    assert agent("hello").message["content"][0]["text"] == "healthy"


def test_opening_choice_stops_when_a_strategy_insists_on_an_unresolvable_candidate():
    broken = RoutingCandidate(ModelRouter([_model()], strategy=_RaisingStrategy()), name="broken")

    class _Insists:
        async def select(self, context):
            return next(c for c in context.candidates if c.name == "broken")

    router = ModelRouter(models=[broken, RoutingCandidate(_model("healthy"), name="healthy")], strategy=_Insists())
    agent = Agent(model=router, retry_strategy=None, callback_handler=None)

    with pytest.raises(RuntimeError, match="strategy boom"):
        agent("hello")


def test_opening_choice_raises_when_no_candidate_can_be_resolved():
    class _Declines:
        async def select(self, context):
            return None

    inner = ModelRouter(models=[_model("inner")], strategy=_Declines())
    router = ModelRouter(models=[RoutingCandidate(inner, name="inner")], strategy=_Declines())
    agent = Agent(model=router, retry_strategy=None, callback_handler=None)

    with pytest.raises(ValueError, match="nested strategy chose no candidate"):
        agent("hello")


class _RendezvousModel(MockedModelProvider):
    """Waits until every node has reached its model call, so the invocations genuinely overlap.

    Without this, Graph nodes finish one at a time and ``_clear_state`` removes the first node's
    state before the second selects, hiding cross-node state bleed.
    """

    def __init__(self, text, rendezvous, participants):
        super().__init__([{"role": "assistant", "content": [{"text": text}]} for _ in range(4)])
        self._rendezvous = rendezvous
        self._participants = participants
        self.calls = 0

    async def stream(self, *args, **kwargs):
        self.calls += 1
        self._rendezvous["arrived"] += 1
        if self._rendezvous["arrived"] >= self._participants:
            self._rendezvous["gate"].set()
        with contextlib.suppress(asyncio.TimeoutError):
            # A leak means one model is never reached, so the gate must not block forever.
            await asyncio.wait_for(self._rendezvous["gate"].wait(), timeout=2)
        async for event in super().stream(*args, **kwargs):
            yield event


@pytest.mark.asyncio
async def test_parallel_graph_nodes_sharing_one_router_route_independently():
    # Graph hands one invocation_state to every node, so a router shared by two agents keeps each
    # node's selection separate and consults the strategy once per node.
    rendezvous = {"arrived": 0, "gate": asyncio.Event()}
    fast = _RendezvousModel("fast-done", rendezvous, participants=2)
    smart = _RendezvousModel("smart-done", rendezvous, participants=2)

    class _BySystemPrompt:
        """Routes on the agent's own system prompt, never on invocation_state["agent"]."""

        def __init__(self):
            self.calls = 0

        async def select(self, context):
            self.calls += 1
            prompt = context.system_prompt or ""
            text = prompt if isinstance(prompt, str) else " ".join(b.get("text", "") for b in prompt)
            wanted = "smart" if "smart" in text else "fast"
            return next(c for c in context.candidates if c.name == wanted)

    strategy = _BySystemPrompt()
    router = ModelRouter(
        models=[RoutingCandidate(fast, name="fast"), RoutingCandidate(smart, name="smart")],
        strategy=strategy,
    )
    builder = GraphBuilder()
    builder.add_node(Agent(model=router, system_prompt="be fast", callback_handler=None), "fast_node")
    builder.add_node(Agent(model=router, system_prompt="be smart", callback_handler=None), "smart_node")

    result = await builder.build().invoke_async("go")

    assert strategy.calls == 2  # both nodes consulted the strategy
    assert (fast.calls, smart.calls) == (1, 1)  # neither node ran on the other's model
    texts = {node_id: node.result.message["content"][0]["text"] for node_id, node in result.results.items()}
    assert texts == {"fast_node": "fast-done", "smart_node": "smart-done"}
