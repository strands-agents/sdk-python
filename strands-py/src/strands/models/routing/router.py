"""ModelRouter: a reusable, immutable set of candidate models with a concrete default.

A router holds an ordered sequence of candidate models and is a ``Plugin`` so an agent can
accept it through ``model=``. It exposes the first candidate, resolved to a concrete model, as
its default and rejects stateful candidates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Union

from ...plugins.plugin import Plugin
from ..model import Model

if TYPE_CHECKING:
    from ...agent.agent import Agent


CandidateInput = Union[Model, "ModelRouter"]

_ROUTER_PLUGIN_NAME = "strands:model-router"


class ModelRouter(Plugin):
    """A reusable, ordered set of candidate models with a concrete default."""

    def __init__(self, models: Sequence[CandidateInput]) -> None:
        """Initialize the router.

        Args:
            models: Candidates as a sequence of ``Model`` objects and nested ``ModelRouter``
                instances. The first candidate is the router's default.

        Raises:
            TypeError: If ``models`` is not a sequence, or a candidate is not a ``Model`` or
                ``ModelRouter``.
            ValueError: If ``models`` is empty or any candidate is a stateful model.
        """
        super().__init__()
        candidates = _normalize(models)
        if not candidates:
            raise ValueError("ModelRouter requires at least one candidate model")
        _reject_stateful(candidates)
        self._candidates = candidates

    @property
    def name(self) -> str:
        """Stable plugin identifier."""
        return _ROUTER_PLUGIN_NAME

    @property
    def default_model(self) -> Model:
        """The first candidate resolved to a concrete model, recursing nested routers."""
        candidate = self._candidates[0]
        if isinstance(candidate, ModelRouter):
            return candidate.default_model
        return candidate

    def init_agent(self, agent: Agent) -> None:
        """Reject a router attached through ``plugins=[...]`` instead of ``model=``.

        Raises:
            ValueError: If the router was not attached through ``Agent(model=...)``.
        """
        if agent._model_router is not self:
            raise ValueError("ModelRouter must be passed through Agent(model=...), not plugins=[...]")


def _normalize(models: object) -> tuple[CandidateInput, ...]:
    """Validate the input is a sequence of candidates and return them as a tuple."""
    if isinstance(models, (str, bytes, Mapping)) or not isinstance(models, Sequence):
        raise TypeError("models must be a sequence of candidates")
    return tuple(_validate_candidate(item) for item in models)


def _validate_candidate(candidate: object) -> CandidateInput:
    """Return the candidate if it is a ``Model`` or ``ModelRouter``; reject other types."""
    if isinstance(candidate, (Model, ModelRouter)):
        return candidate
    raise TypeError(f"candidate must be a Model or ModelRouter; got {type(candidate).__name__}")


def _reject_stateful(candidates: tuple[CandidateInput, ...]) -> None:
    """Reject any stateful candidate model."""
    for candidate in candidates:
        if isinstance(candidate, Model) and candidate.stateful:
            raise ValueError(
                f"candidate=<{type(candidate).__name__}> is stateful; routing among stateful models is not supported"
            )
