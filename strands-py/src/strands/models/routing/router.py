"""ModelRouter: a reusable, immutable set of candidate models with a routing strategy.

A router is a ``Plugin``, so an agent accepts it through ``model=``. Its ``RoutingStrategy`` makes
every routing decision: the router asks for a candidate before the first model call, and again after
a call fails without a hook claiming the retry, passing the attempts so far. The strategy answers
with a candidate, or ``None`` to stop and let the error surface.

The router orchestrates only. It resolves a candidate to a concrete model, applies it to the call,
gives each new candidate a fresh retry budget, and holds per-invocation state. It has no failover
policy, so a strategy can change routing behavior without changing the router.

The default ``FallbackStrategy`` follows declaration order, making ``ModelRouter(models=[a, b])``
ordered failover, with two refinements: a successful call clears the failures recorded before it, so
a candidate that failed earlier becomes eligible again, and candidates with more recorded failures
are tried after healthier ones. A strategy that fails or declines the opening choice
degrades to the first declared candidate; ``max_switches`` caps switches per invocation. A strategy
that re-offers the candidate that just failed is taken to mean "stay here", so the model's error
surfaces rather than the router resetting the retry budget again.

A nested ``ModelRouter`` contributes **one** candidate: it is asked with its own candidates and no
attempts, so its strategy picks from a clean slate every time and the group performs no internal
failover. When a nested pick fails, the outer router moves off the whole nested candidate rather than
advancing within it.

Known limitation: a model that fails after streaming part of a response has already emitted those
events, so a streaming consumer sees that partial output followed by the replacement's full response.
``AfterModelCallEvent`` documents this for any hook-requested retry; routing reaches it more often
because it advances on failures retry declines, not only throttling.
"""

from __future__ import annotations

import copy
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Union

from ..._middleware.stages import InvokeModelStage
from ...hooks.events import AfterInvocationEvent, AfterModelCallEvent, BeforeInvocationEvent
from ...hooks.registry import HookOrder
from ...plugins.plugin import Plugin
from ..model import Model
from .strategy import RoutingAttempt, RoutingContext, RoutingStrategy

if TYPE_CHECKING:
    from ..._middleware.stages import InvokeModelContext
    from ...agent.agent import Agent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutingCandidate:
    """A routing candidate: a model with an optional name and description.

    ``model`` may be a nested ``ModelRouter``, which contributes one candidate: its strategy picks
    from its own candidates, and the group performs no internal failover.
    """

    model: Model | ModelRouter
    name: str | None = None
    description: str | None = None


CandidateInput = Union[Model, "ModelRouter", RoutingCandidate]

_ROUTER_PLUGIN_NAME = "strands:model-router"
_ROUTING_KEY_PREFIX = "strands:model_routing"


@dataclass
class _RoutingState:
    """Per-invocation routing state for one agent/router pair.

    ``attempts`` is the record the strategy reads to make its next decision; the router appends to it
    but never interprets it. ``switches`` counts model changes so ``max_switches`` can cap them.
    """

    candidate: RoutingCandidate
    model: Model
    attempts: list[RoutingAttempt] = field(default_factory=list)
    switches: int = 0


class FallbackStrategy:
    """Works down the candidates in declaration order, preferring the ones failing least.

    A success clears the failures before it, so a later failure may return to an earlier candidate.
    Candidates that keep failing sink below healthy ones.
    """

    async def select(self, context: RoutingContext, **kwargs: Any) -> RoutingCandidate | None:
        """Return the least-failed candidate not yet tried since the last success, else ``None``."""
        failures: dict[int, int] = {}
        for attempt in context.attempts:
            if attempt.error is None:
                failures.pop(id(attempt.candidate), None)
            else:
                failures[id(attempt.candidate)] = failures.get(id(attempt.candidate), 0) + 1

        tried_now = {id(attempt.candidate) for attempt in _attempts_since_success(context.attempts)}
        available = [candidate for candidate in context.candidates if id(candidate) not in tried_now]
        if not available:
            return None
        return min(available, key=lambda candidate: failures.get(id(candidate), 0))


class ModelRouter(Plugin):
    """A reusable set of candidate models routed in strategy-defined preference order."""

    def __init__(
        self,
        models: Sequence[CandidateInput],
        *,
        strategy: RoutingStrategy | None = None,
        max_switches: int | None = None,
    ) -> None:
        """Initialize the router.

        Args:
            models: Candidates as a sequence. Each is a ``Model``, a nested ``ModelRouter``, or a
                ``RoutingCandidate`` carrying an optional name/description. The first candidate is
                the router's concrete default, used when a strategy cannot produce a choice.
            strategy: Chooses the candidate for each model call, and is asked again after a failed
                call. Defaults to ``FallbackStrategy``: declaration order, except that a successful
                call clears the failures before it, so a candidate that already failed is eligible
                again, and candidates that keep failing are tried after healthier ones.
            max_switches: Cap on model switches within one invocation, after which the router stops
                asking and lets the error surface. Defaults to ``None`` (the strategy decides when to
                stop) -- set it when a strategy could keep escalating across a long tool loop.

        Raises:
            TypeError: If ``models`` is not a sequence, a candidate is not a ``Model`` or
                ``ModelRouter``, or ``strategy`` does not implement ``RoutingStrategy``.
            ValueError: If ``models`` is empty, candidate names collide, any candidate is a stateful
                model, or ``max_switches`` is negative.
        """
        super().__init__()
        # Only the member the router calls, so optional protocol members stay optional. A sync
        # select would otherwise construct fine and then be swallowed as a routing failure.
        if strategy is not None and not inspect.iscoroutinefunction(getattr(strategy, "select", None)):
            raise TypeError("strategy must implement RoutingStrategy: an async select(context) method")
        if max_switches is not None and max_switches < 0:
            raise ValueError("max_switches must be zero or greater")
        candidates = _normalize(models)
        if not candidates:
            raise ValueError("ModelRouter requires at least one candidate model")
        _reject_stateful(candidates)
        _reject_duplicates(candidates)
        self._candidates = candidates
        self._strategy: RoutingStrategy = strategy or FallbackStrategy()
        self._max_switches = max_switches

    @property
    def name(self) -> str:
        """Stable plugin identifier."""
        return _ROUTER_PLUGIN_NAME

    @property
    def candidates(self) -> tuple[RoutingCandidate, ...]:
        """The normalized candidates, in declaration order."""
        return self._candidates

    @property
    def default_model(self) -> Model:
        """The first declared candidate resolved to a concrete model."""
        model = self._candidates[0].model
        if isinstance(model, ModelRouter):
            return model.default_model
        return model

    def init_agent(self, agent: Agent) -> None:
        """Register routing middleware and hooks; reject attachment through ``plugins=[...]``.

        Args:
            agent: The agent the router is attached to.

        Raises:
            ValueError: If the router was not attached through ``Agent(model=...)``.
        """
        if agent._model_router is not self:
            raise ValueError("ModelRouter must be passed through Agent(model=...), not plugins=[...]")

        agent._middleware_registry.add_middleware(InvokeModelStage.Input, self._selection_middleware())
        # Fallback must see whether ModelRetryStrategy (DEFAULT) already claimed the retry.
        agent.hooks.add_callback(AfterModelCallEvent, self._on_model_result, order=HookOrder.MODEL_ROUTING)
        # Cleared at both ends: teardown last so other end-of-invocation callbacks still observe the
        # selection, and again on entry so a skipped teardown cannot pin a reused invocation_state.
        agent.hooks.add_callback(AfterInvocationEvent, self._clear_state, order=HookOrder.SDK_LAST)
        agent.hooks.add_callback(BeforeInvocationEvent, self._clear_state, order=HookOrder.SDK_FIRST)

    @property
    def _strategy_name(self) -> str:
        """Strategy class name, for logs that explain a routing decision."""
        return type(self._strategy).__name__

    async def _ask(self, context: RoutingContext) -> RoutingCandidate | None:
        """Ask the strategy for a candidate and validate the answer."""
        return self._validated(await self._strategy.select(context), context)

    def _validated(self, candidate: object, context: RoutingContext) -> RoutingCandidate | None:
        """Return the candidate, raising if the strategy broke its contract."""
        if candidate is None:
            return None
        if not isinstance(candidate, RoutingCandidate):
            raise TypeError(f"strategy.select must return a RoutingCandidate or None; got {type(candidate).__name__}")
        if not any(candidate is configured for configured in context.candidates):
            raise ValueError("strategy.select must return a candidate from context.candidates")
        return candidate

    async def _open(self, context: RoutingContext) -> tuple[RoutingCandidate, Model]:
        """Choose and resolve the candidate to start on, skipping any that cannot be resolved.

        A candidate that fails to resolve is skipped wherever it sits in the list, so a nested router
        that declines does not fail the invocation just because it was declared first.
        """
        attempts: list[RoutingAttempt] = []
        errors: list[Exception] = []
        # One ask per candidate, plus a final ask once every candidate has failed to resolve.
        for _ in range(len(self._candidates) + 1):
            ask_context = replace(context, attempts=tuple(attempts))
            candidate = await self._select_initial(ask_context, attempts)
            if candidate is None:
                break
            try:
                return candidate, await self._resolve(candidate, ask_context)
            except Exception as error:
                logger.warning(
                    "candidate=<%s>, error=<%s> | candidate could not be resolved",
                    _candidate_label(candidate),
                    error,
                )
                attempts.append(RoutingAttempt(candidate=candidate, error=error))
                errors.append(error)

        raise errors[-1]

    async def _select_initial(
        self, context: RoutingContext, attempts: Sequence[RoutingAttempt]
    ) -> RoutingCandidate | None:
        """Choose the candidate to start with, degrading to a default if the strategy fails."""
        try:
            answer = await self._strategy.select(context)
        except Exception as error:
            logger.warning(
                "strategy=<%s>, error=<%s> | routing failed, using the default candidate",
                self._strategy_name,
                error,
            )
            return self._default_candidate(attempts)

        candidate = self._validated(answer, context)
        if candidate is None:
            logger.warning(
                "strategy=<%s> | strategy chose no candidate, using the default candidate", self._strategy_name
            )
            return self._default_candidate(attempts)

        logger.info(
            "strategy=<%s>, candidate=<%s> | candidate selected",
            self._strategy_name,
            _candidate_label(candidate),
        )
        return candidate

    def _default_candidate(self, attempts: Sequence[RoutingAttempt]) -> RoutingCandidate | None:
        """First declared candidate that has not already failed, or ``None`` when all have."""
        failed = {id(attempt.candidate) for attempt in attempts}
        return next((candidate for candidate in self._candidates if id(candidate) not in failed), None)

    async def _resolve(self, candidate: RoutingCandidate, context: RoutingContext) -> Model:
        """Resolve a candidate to a concrete model, recursing into a nested router's selection.

        A nested router is asked with its own candidates and no attempts, since the outer log records
        candidates that are not its own. It therefore contributes one candidate and never advances
        internally.
        """
        model = candidate.model
        if isinstance(model, ModelRouter):
            return await model._select_model(replace(context, candidates=model.candidates, attempts=()))
        return model

    async def _select_model(self, context: RoutingContext) -> Model:
        """Resolve this router's chosen candidate; raising marks the whole router unusable."""
        candidate = await self._ask(context)
        if candidate is None:
            raise ValueError("nested strategy chose no candidate")
        return await self._resolve(candidate, context)

    def _selection_middleware(self) -> Callable[[InvokeModelContext], Awaitable[InvokeModelContext]]:
        """Build an ``InvokeModelStage.Input`` handler that applies the per-invocation selection."""

        async def middleware(context: InvokeModelContext) -> InvokeModelContext:
            key = self._state_key(context.agent)
            state = _routing_state(context.invocation_state, key)
            if state is None:
                # Snapshot: a strategy must not be able to mutate the request this call runs on.
                routing_context = self._routing_context(
                    copy.deepcopy(context.messages),
                    copy.deepcopy(context.system_prompt),
                    context.tool_specs,
                    context.invocation_state,
                )
                candidate, model = await self._open(routing_context)
                state = _RoutingState(candidate=candidate, model=model)
                context.invocation_state[key] = state
            context.model = state.model
            return context

        return middleware

    async def _on_model_result(self, event: AfterModelCallEvent) -> None:
        """Record the outcome and, after an unclaimed failure, apply the strategy's next choice."""
        state = _routing_state(event.invocation_state, self._state_key(event.agent))
        if state is None:
            return
        if event.stop_response is not None:
            state.attempts.append(RoutingAttempt(candidate=state.candidate))
            return
        if event.retry or event.exception is None:
            return

        state.attempts.append(RoutingAttempt(candidate=state.candidate, error=event.exception))
        if self._max_switches is not None and state.switches >= self._max_switches:
            logger.warning("max_switches=<%d> | switch cap reached, leaving the error to surface", self._max_switches)
            return
        previous = state.candidate

        # Bounded by the candidate count so a strategy repeating one unresolvable choice cannot spin.
        for _ in range(len(self._candidates)):
            routing_context = self._agent_routing_context(event.agent, event.invocation_state, state.attempts)
            try:
                answer = await self._strategy.select(routing_context)
            except Exception as error:
                logger.warning(
                    "strategy=<%s>, error=<%s> | routing failed, leaving the error to surface",
                    self._strategy_name,
                    error,
                )
                return
            candidate = self._validated(answer, routing_context)
            if candidate is None:
                return
            try:
                model = await self._resolve(candidate, routing_context)
            except Exception as error:
                logger.warning(
                    "candidate=<%s>, error=<%s> | candidate could not be resolved",
                    _candidate_label(candidate),
                    error,
                )
                state.attempts.append(RoutingAttempt(candidate=candidate, error=error))
                continue

            # Re-offering the candidate that just failed means "stay here". Treating it as a switch
            # would reset the retry budget every round and the invocation would never end.
            if candidate is previous:
                logger.info(
                    "candidate=<%s> | strategy re-offered the failed candidate, leaving the error to surface",
                    _candidate_label(candidate),
                )
                return

            logger.info(
                "from_candidate=<%s>, to_candidate=<%s>, error=<%s> | model call failed, switching candidate",
                _candidate_label(previous),
                _candidate_label(candidate),
                type(event.exception).__name__,
            )
            state.candidate = candidate
            state.model = model
            state.switches += 1
            # Reaches into ModelRetryStrategy; no public seam for a budget reset exists yet.
            event.agent._retry_strategy._reset_retry_state()
            event.retry = True
            return

    async def _clear_state(self, event: AfterInvocationEvent | BeforeInvocationEvent) -> None:
        """Drop this agent's routing state so it never spans invocations."""
        key = self._state_key(event.agent)
        if _routing_state(event.invocation_state, key) is not None:
            del event.invocation_state[key]

    def _state_key(self, agent: object) -> str:
        """Scope routing state to one agent/router pair.

        One ``invocation_state`` can serve several agents, and one router several agents, so neither
        identity alone is a sufficient key.
        """
        return f"{_ROUTING_KEY_PREFIX}:{id(agent):x}:{id(self):x}"

    def _agent_routing_context(
        self, agent: Any, invocation_state: Mapping[str, Any], attempts: Sequence[RoutingAttempt] = ()
    ) -> RoutingContext:
        """Build a ``RoutingContext`` from the agent, matching the shapes middleware passes."""
        system_prompt = (
            agent._system_prompt_content if agent._system_prompt_content is not None else agent.system_prompt
        )
        return self._routing_context(
            copy.deepcopy(agent.messages),
            copy.deepcopy(system_prompt),
            agent.tool_registry.get_all_tool_specs(),
            invocation_state,
            attempts,
        )

    def _routing_context(
        self,
        messages: Any,
        system_prompt: Any,
        tool_specs: Any,
        invocation_state: Mapping[str, Any],
        attempts: Sequence[RoutingAttempt] = (),
    ) -> RoutingContext:
        """Build a ``RoutingContext`` over this router's candidates."""
        return RoutingContext(
            messages=messages,
            system_prompt=system_prompt,
            tool_specs=tool_specs,
            candidates=self._candidates,
            invocation_state=invocation_state,
            attempts=tuple(attempts),
        )


def _attempts_since_success(attempts: Sequence[RoutingAttempt]) -> Sequence[RoutingAttempt]:
    """Return the trailing attempts that all failed, dropping everything up to the last success."""
    for index in range(len(attempts) - 1, -1, -1):
        if attempts[index].error is None:
            return attempts[index + 1 :]
    return attempts


def _candidate_label(candidate: RoutingCandidate) -> str:
    """Return a stable human-readable label for logs."""
    return candidate.name or type(candidate.model).__name__


def _routing_state(invocation_state: Mapping[str, Any], key: str) -> _RoutingState | None:
    """Return the routing state stored under ``key``, ignoring any foreign value."""
    value = invocation_state.get(key)
    return value if isinstance(value, _RoutingState) else None


def _normalize(models: object) -> tuple[RoutingCandidate, ...]:
    """Coerce the input sequence into ``RoutingCandidate`` objects, validating candidate types."""
    if isinstance(models, (str, bytes, Mapping)) or not isinstance(models, Sequence):
        raise TypeError("models must be a sequence of candidates")
    return tuple(_as_candidate(item) for item in models)


def _as_candidate(item: CandidateInput) -> RoutingCandidate:
    """Wrap a candidate input in a ``RoutingCandidate``, validating its model type."""
    candidate = item if isinstance(item, RoutingCandidate) else RoutingCandidate(model=item)
    if not isinstance(candidate.model, (Model, ModelRouter)):
        raise TypeError(f"candidate must be a Model or ModelRouter; got {type(candidate.model).__name__}")
    return candidate


def _reject_stateful(candidates: tuple[RoutingCandidate, ...]) -> None:
    """Reject any stateful candidate model."""
    for candidate in candidates:
        if isinstance(candidate.model, Model) and candidate.model.stateful:
            raise ValueError(
                f"candidate=<{_candidate_label(candidate)}> is stateful; routing among stateful models is not supported"
            )


def _reject_duplicates(candidates: tuple[RoutingCandidate, ...]) -> None:
    """Reject repeated candidate instances or colliding names; repeated models are allowed."""
    seen_candidates: set[int] = set()
    seen_names: set[str] = set()
    for candidate in candidates:
        identity = id(candidate)
        if identity in seen_candidates:
            raise ValueError("duplicate RoutingCandidate instance")
        seen_candidates.add(identity)

        if candidate.name is None:
            continue
        if candidate.name in seen_names:
            raise ValueError(f"duplicate candidate name=<{candidate.name}>")
        seen_names.add(candidate.name)
