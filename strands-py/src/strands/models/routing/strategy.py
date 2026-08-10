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
    ``None`` when the call succeeded. Otherwise it is either the model's error as
    ``AfterModelCallEvent`` reports it, or -- when the candidate could not be resolved to a model at
    all -- the error raised while resolving it, so a strategy that treats the two differently should
    inspect the exception rather than assume a model was reached. Attempts are in chronological order,
    so a strategy can tell a first failure from a repeated one and treat a candidate that recovered as
    healthy.
    """

    candidate: RoutingCandidate
    exception: Exception | None = None


@dataclass(frozen=True)
class RoutingContext:
    """Read-only inputs a strategy sees when choosing a candidate.

    ``messages``, ``system_prompt``, and ``tool_specs`` are fresh copies per ask, so do not mutate
    them and do not rely on their object identity across asks. ``candidates`` and the ``candidate`` on
    each ``RoutingAttempt`` are the router's own instances and are stable for the router's lifetime,
    so a strategy may correlate attempts with candidates by identity.

    ``invocation_state`` is the live dict, not a copy: reading it is the point, but writing to it
    reaches the agent's own state and the router's, which keeps its per-invocation state there under a
    ``strands:model_routing`` key. In a multi-agent run it may be shared across nodes, so its
    ``"agent"`` value may identify a sibling.

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
        """Return the candidate to use, from ``context.candidates``, or ``None`` to decline.

        Asked before the first model call, and again after each failed call with that failure appended
        to ``context.attempts``. ``attempts`` is usually empty on the opening ask but not always: a
        candidate that cannot be resolved to a model is recorded and the opening ask repeats, so
        ``attempts`` does not reliably distinguish the opening ask from a later one.

        ``None`` declines. On the opening ask the router still serves the request on the first declared
        candidate, so a servable request does not fail on a routing decision; on a later ask it ends
        routing and lets the model's error surface.

        The return value must be one of the ``context.candidates`` instances -- the router matches by
        identity, so an equal-looking ``RoutingCandidate`` built here is rejected.

        One router rule constrains the answer, and it is predictable from ``context.attempts``: a
        failure round switches to each candidate at most once, where a round is the run of failures
        since the last success. Naming a candidate the round already switched to is not an error and
        costs no model call -- the router simply asks again -- but it cannot be used to re-run that
        candidate, so a strategy that judges a failure transient should offer a different candidate and
        wait for the success that clears the round. When nothing is left to switch to, the model's
        error surfaces.

        Failover is this method's job: the router applies what is returned and never substitutes a
        candidate of its own, so a strategy that ignores ``context.attempts`` gets no failover. Wrap or
        delegate to ``FallbackStrategy`` to get it.
        """
        ...


def _attempts_since_success(attempts: Sequence[RoutingAttempt]) -> Sequence[RoutingAttempt]:
    """Return the trailing attempts that all failed, dropping everything up to the last success."""
    for index in range(len(attempts) - 1, -1, -1):
        if attempts[index].exception is None:
            return attempts[index + 1 :]
    return attempts
