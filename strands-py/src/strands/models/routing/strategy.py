"""Routing strategy protocol, the attempt record it sees, and the context passed to it.

A ``RoutingStrategy`` decides which candidate an invocation uses. ``ModelRouter`` asks it for a
candidate before the first model call and again after a failed call, passing the attempts made so
far, so every routing decision -- initial choice, whether to fail over, and to what -- belongs to the
strategy. The router only orchestrates: it resolves the candidate, applies it to the call, manages
retry budgets and invocation state, and stops when the strategy has no further candidate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ...types.content import Messages, SystemPrompt
    from ...types.tools import ToolSpec
    from .router import RoutingCandidate


@dataclass(frozen=True)
class RoutingAttempt:
    """A candidate this invocation already used, and how the call ended.

    ``candidate`` is the instance from ``RoutingContext.candidates``, not a copy. ``exception`` is
    ``None`` when the call succeeded, and is otherwise the same object ``AfterModelCallEvent`` reports.
    Attempts are in chronological order, so a strategy can tell a first failure from a repeated one
    and treat a candidate that recovered as healthy.
    """

    candidate: RoutingCandidate
    exception: Exception | None = None


@dataclass(frozen=True)
class RoutingContext:
    """Read-only inputs a strategy sees when choosing a candidate.

    ``messages``, ``system_prompt``, and ``tool_specs`` are fresh copies per ask, so do not mutate
    them and do not rely on their object identity across asks. ``candidates`` and the ``candidate`` on
    each ``RoutingAttempt`` are the router's own instances and are stable for the router's lifetime,
    so a strategy may correlate attempts with candidates by identity. In a multi-agent run, one
    ``invocation_state`` may be shared across nodes, so its ``"agent"`` value may identify a sibling.

    A strategy is asked on every invocation because the right model usually depends on the request:
    reusing an earlier decision would answer a hard request with a model chosen for an easy one. A
    strategy that is expensive to evaluate should narrow what it looks at -- typically the latest
    turn rather than the whole transcript -- instead of caching a verdict across invocations.
    """

    messages: Messages
    system_prompt: SystemPrompt | None
    tool_specs: Sequence[ToolSpec]
    candidates: Sequence[RoutingCandidate]
    invocation_state: Mapping[str, Any]
    attempts: Sequence[RoutingAttempt] = field(default_factory=tuple)


@runtime_checkable
class RoutingStrategy(Protocol):
    """Chooses the candidate an invocation uses, including after a failure.

    ``ModelRouter`` requires only ``select``, so members added here later stay optional for
    strategies already written against this protocol.
    """

    async def select(self, context: RoutingContext, **kwargs: Any) -> RoutingCandidate | None:
        """Return the candidate to use, from ``context.candidates``, or ``None`` to stop routing.

        Called with no attempts for the initial choice, then again after each failed call with that
        failure appended. Returning ``None`` ends routing and lets the model's error surface, so a
        strategy that declines to reconsider keeps the invocation on the model it first chose.

        The return value must be one of the ``context.candidates`` instances; the router matches by
        identity, so an equal-looking ``RoutingCandidate`` built by the strategy is rejected.
        """
        ...
