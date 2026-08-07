"""The default routing strategy: ordered failover that prefers the candidates failing least."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .strategy import RoutingAttempt, RoutingContext

if TYPE_CHECKING:
    from .router import RoutingCandidate


class FallbackStrategy:
    """Works down the candidates in declaration order, preferring the ones failing least.

    A success clears the failures before it, so a later failure may return to an earlier candidate.
    Candidates that keep failing sink below healthy ones.
    """

    async def select(self, context: RoutingContext, **kwargs: Any) -> RoutingCandidate | None:
        """Return the least-failed candidate not yet tried since the last success, else ``None``."""
        failures: dict[int, int] = {}
        for attempt in context.attempts:
            if attempt.exception is None:
                failures.pop(id(attempt.candidate), None)
            else:
                failures[id(attempt.candidate)] = failures.get(id(attempt.candidate), 0) + 1

        tried_now = {id(attempt.candidate) for attempt in _attempts_since_success(context.attempts)}
        available = [candidate for candidate in context.candidates if id(candidate) not in tried_now]
        if not available:
            return None
        return min(available, key=lambda candidate: failures.get(id(candidate), 0))


def _attempts_since_success(attempts: Sequence[RoutingAttempt]) -> Sequence[RoutingAttempt]:
    """Return the trailing attempts that all failed, dropping everything up to the last success."""
    for index in range(len(attempts) - 1, -1, -1):
        if attempts[index].exception is None:
            return attempts[index + 1 :]
    return attempts
