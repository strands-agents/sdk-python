"""ModelRouter: a reusable, immutable set of candidate models with a routing strategy.

A router holds an ordered sequence of candidates and is a ``Plugin`` so an agent can accept it
through ``model=``. Its ``RoutingStrategy`` selects a candidate once per agent invocation; the
choice is cached and reused for every model call in that invocation, including tool-loop turns.
When the selected model fails and no hook has claimed the retry, the router advances to the next
untried candidate in declaration order (ordered fallback), re-arming the chain after any successful
call. A nested ``ModelRouter`` candidate is a single atomic fallback slot: its own strategy chooses
its model, but the outer router does not fall back among the nested candidates.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Union

from ...hooks.events import AfterInvocationEvent, AfterModelCallEvent
from ...hooks.registry import HookOrder
from ...plugins.plugin import Plugin
from ..model import Model
from .strategy import RoutingContext, RoutingStrategy

if TYPE_CHECKING:
    from ..._middleware.stages import InvokeModelContext
    from ...agent.agent import Agent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutingCandidate:
    """A routing candidate: a model with an optional name and description.

    ``model`` may be a nested ``ModelRouter``; in a fallback chain it is one atomic slot (its own
    strategy picks its model, and the outer router does not fall back among the nested candidates).
    """

    model: Model | ModelRouter
    name: str | None = None
    description: str | None = None


CandidateInput = Union[Model, "ModelRouter", RoutingCandidate]

_ROUTER_PLUGIN_NAME = "strands:model-router"
_ROUTING_KEY = "__strands_model_routing__"


class FallbackStrategy:
    """Selects the first candidate; the router advances through the rest on failure."""

    async def select(self, context: RoutingContext, **kwargs: Any) -> RoutingCandidate:
        """Return the first candidate."""
        return context.candidates[0]


class ModelRouter(Plugin):
    """A reusable, ordered set of candidate models with a routing strategy and ordered fallback."""

    def __init__(self, models: Sequence[CandidateInput], *, strategy: RoutingStrategy | None = None) -> None:
        """Initialize the router.

        Args:
            models: Candidates as a sequence. Each is a ``Model``, a nested ``ModelRouter``, or a
                ``RoutingCandidate`` carrying an optional name/description. The first candidate is
                the router's default.
            strategy: Selects a candidate per invocation. Defaults to selecting the first candidate.

        Raises:
            TypeError: If ``models`` is not a sequence, a candidate is not a ``Model`` or
                ``ModelRouter``, or ``strategy`` does not implement ``RoutingStrategy``.
            ValueError: If ``models`` is empty, candidate names collide, or any candidate is a
                stateful model.
        """
        super().__init__()
        if strategy is not None and not isinstance(strategy, RoutingStrategy):
            raise TypeError("strategy must implement RoutingStrategy (a select(context) method)")
        candidates = _normalize(models)
        if not candidates:
            raise ValueError("ModelRouter requires at least one candidate model")
        _reject_stateful(candidates)
        _reject_duplicate_names(candidates)
        self._candidates = candidates
        self._strategy: RoutingStrategy = strategy or FallbackStrategy()

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
        """The first candidate resolved to a concrete model, recursing nested routers."""
        model = self._candidates[0].model
        if isinstance(model, ModelRouter):
            return model.default_model
        return model

    def init_agent(self, agent: Agent) -> None:
        """Register the fallback and state-cleanup hooks; reject attachment through ``plugins=[...]``.

        Args:
            agent: The agent the router is attached to.

        Raises:
            ValueError: If the router was not attached through ``Agent(model=...)``.
        """
        if agent._model_router is not self:
            raise ValueError("ModelRouter must be passed through Agent(model=...), not plugins=[...]")
        # SDK_LAST so fallback runs after ModelRetryStrategy has decided whether to retry.
        agent.hooks.add_callback(AfterModelCallEvent, self._on_model_result, order=HookOrder.SDK_LAST)
        agent.hooks.add_callback(AfterInvocationEvent, self._clear_state, order=HookOrder.SDK_LAST)

    async def _pick(self, context: RoutingContext) -> RoutingCandidate:
        """Select a candidate via the strategy, requiring the result to be one of the candidates."""
        candidate = await self._strategy.select(context)
        if not any(candidate is existing for existing in context.candidates):
            raise ValueError("strategy.select must return one of the candidates")
        return candidate

    async def _resolve(self, candidate: RoutingCandidate, context: RoutingContext) -> Model:
        """Resolve a candidate to a concrete model, recursing into a nested router's strategy."""
        model = candidate.model
        if isinstance(model, ModelRouter):
            return await model._select_model(replace(context, candidates=model.candidates))
        return model

    async def _select_model(self, context: RoutingContext) -> Model:
        """Select and resolve a candidate to a concrete model."""
        return await self._resolve(await self._pick(context), context)

    def _selection_middleware(self) -> Callable[[InvokeModelContext], Awaitable[InvokeModelContext]]:
        """Build an ``InvokeModelStage.Input`` handler that selects the per-invocation model."""

        async def middleware(context: InvokeModelContext) -> InvokeModelContext:
            state = context.invocation_state.get(_ROUTING_KEY)
            if state is None or state.get("router") is not self:
                routing_context = self._routing_context(
                    context.messages, context.system_prompt, context.tool_specs, context.invocation_state
                )
                candidate = await self._pick(routing_context)
                index = self._index_of(candidate)
                state = {
                    "router": self,
                    "index": index,
                    "model": await self._resolve(candidate, routing_context),
                    "tried": {index},
                }
                context.invocation_state[_ROUTING_KEY] = state
            context.model = state["model"]
            return context

        return middleware

    async def _on_model_result(self, event: AfterModelCallEvent) -> None:
        """Re-arm the fallback chain on success; on an unretried failure, advance to the next candidate."""
        state = event.invocation_state.get(_ROUTING_KEY)
        if state is None or state.get("router") is not self:
            return
        if event.stop_response is not None:
            state["tried"] = {state["index"]}  # a successful call re-arms the rest of the chain
            return
        if event.retry or event.exception is None:
            return

        tried: set[int] = state["tried"]
        next_index = next((index for index in range(len(self._candidates)) if index not in tried), None)
        if next_index is None:
            return

        routing_context = self._routing_context(
            event.agent.messages,
            event.agent.system_prompt,
            event.agent.tool_registry.get_all_tool_specs(),
            event.invocation_state,
        )
        try:
            model = await self._resolve(self._candidates[next_index], routing_context)
        except Exception as error:
            # A failed advance must not replace the original model error or suppress ForceStopEvent.
            logger.warning("candidate_index=<%d>, error=<%s> | fallback resolution failed", next_index, error)
            return

        logger.info(
            "from_index=<%d>, to_index=<%d>, error=<%s> | model call failed, advancing to next candidate",
            state["index"],
            next_index,
            type(event.exception).__name__,
        )
        state["model"] = model
        state["index"] = next_index
        tried.add(next_index)
        event.agent._retry_strategy.reset_retry_state()
        event.retry = True

    async def _clear_state(self, event: AfterInvocationEvent) -> None:
        """Drop this router's routing state at the end of the invocation so it does not leak."""
        state = event.invocation_state.get(_ROUTING_KEY)
        if state is not None and state.get("router") is self:
            del event.invocation_state[_ROUTING_KEY]

    def _index_of(self, candidate: RoutingCandidate) -> int:
        """Return the declaration-order index of a candidate by identity."""
        return next(index for index, existing in enumerate(self._candidates) if existing is candidate)

    def _routing_context(
        self, messages: Any, system_prompt: Any, tool_specs: Any, invocation_state: Mapping[str, Any]
    ) -> RoutingContext:
        """Build a ``RoutingContext`` over this router's candidates."""
        return RoutingContext(
            messages=messages,
            system_prompt=system_prompt,
            tool_specs=tool_specs,
            candidates=self._candidates,
            invocation_state=invocation_state,
        )


def _normalize(models: object) -> tuple[RoutingCandidate, ...]:
    """Coerce the input sequence into ``RoutingCandidate`` objects, validating candidate types."""
    if isinstance(models, (str, bytes, Mapping)) or not isinstance(models, Sequence):
        raise TypeError("models must be a sequence of candidates")
    return tuple(_as_candidate(item) for item in models)


def _as_candidate(item: CandidateInput) -> RoutingCandidate:
    """Wrap a candidate input in a ``RoutingCandidate``, validating its model type."""
    if isinstance(item, RoutingCandidate):
        _validate_candidate_model(item.model)
        return item
    return RoutingCandidate(model=_validate_candidate_model(item))


def _validate_candidate_model(model: object) -> Model | ModelRouter:
    """Return the model if it is a ``Model`` or ``ModelRouter``; reject other types."""
    if isinstance(model, (Model, ModelRouter)):
        return model
    raise TypeError(f"candidate must be a Model or ModelRouter; got {type(model).__name__}")


def _reject_stateful(candidates: tuple[RoutingCandidate, ...]) -> None:
    """Reject any stateful candidate model."""
    for candidate in candidates:
        if isinstance(candidate.model, Model) and candidate.model.stateful:
            label = candidate.name or type(candidate.model).__name__
            raise ValueError(f"candidate=<{label}> is stateful; routing among stateful models is not supported")


def _reject_duplicate_names(candidates: tuple[RoutingCandidate, ...]) -> None:
    """Reject colliding candidate names; unnamed candidates and repeated models are allowed."""
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.name is None:
            continue
        if candidate.name in seen:
            raise ValueError(f"duplicate candidate name=<{candidate.name}>")
        seen.add(candidate.name)
