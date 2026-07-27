"""Routing strategy protocol and the context passed to it.

A ``RoutingStrategy`` picks one candidate per invocation from the router's candidates, given the
call's messages, system prompt, tool specs, and invocation state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ...types.content import Messages, SystemPrompt
    from ...types.tools import ToolSpec
    from .router import RoutingCandidate


@dataclass(frozen=True)
class RoutingContext:
    """Read-only inputs a strategy sees when selecting a candidate.

    The collections are shared by reference and must not be mutated; a strategy reads them to make
    its decision and returns one of ``candidates``.
    """

    messages: Messages
    system_prompt: SystemPrompt | None
    tool_specs: Sequence[ToolSpec]
    candidates: Sequence[RoutingCandidate]
    invocation_state: Mapping[str, Any]


@runtime_checkable
class RoutingStrategy(Protocol):
    """Selects one candidate for an invocation."""

    async def select(self, context: RoutingContext, **kwargs: Any) -> RoutingCandidate:
        """Return the candidate to use, which must be one of ``context.candidates``."""
        ...
