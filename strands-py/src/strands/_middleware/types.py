"""Middleware type system."""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

TContext = TypeVar("TContext")
TResult = TypeVar("TResult")
TEvent = TypeVar("TEvent")


@dataclass
class MiddlewareResult(Generic[TResult]):
    """Wrapper passed to and returned from Output phase handlers.

    Wrapping the value (rather than handing back the raw result event) gives Output
    handlers a stable surface to evolve — e.g. we may add aggregated metadata fields here
    later without changing the handler signature.

    Attributes:
        value: The stage's result — the last event from the chain (e.g. ``ModelStopReason``).
    """

    value: TResult

    def replace(self, *, value: TResult) -> MiddlewareResult[TResult]:
        """Return a copy with ``value`` replaced.

        Convenience wrapper around ``dataclasses.replace`` so Output handlers don't need
        to import it:

            return result.replace(value=transformed_event)
        """
        return dataclasses.replace(self, value=value)


class MiddlewareInputPhase(Generic[TContext, TResult, TEvent]):
    """Phase sub-token for Input handlers — transforms context before execution."""

    __slots__ = ("_stage", "_phase")

    def __init__(self, stage: MiddlewareStage[TContext, TResult, TEvent]) -> None:
        self._stage = stage
        self._phase = "input"


class MiddlewareWrapPhase(Generic[TContext, TResult, TEvent]):
    """Phase sub-token for Wrap handlers — full async generator wrap."""

    __slots__ = ("_stage", "_phase")

    def __init__(self, stage: MiddlewareStage[TContext, TResult, TEvent]) -> None:
        self._stage = stage
        self._phase = "wrap"


class MiddlewareOutputPhase(Generic[TContext, TResult, TEvent]):
    """Phase sub-token for Output handlers — transforms result after execution."""

    __slots__ = ("_stage", "_phase")

    def __init__(self, stage: MiddlewareStage[TContext, TResult, TEvent]) -> None:
        self._stage = stage
        self._phase = "output"


class MiddlewareStage(Generic[TContext, TResult, TEvent]):
    """A stage token identifying a middleware interception point."""

    __slots__ = ("name", "Input", "Wrap", "Output")

    def __init__(self, name: str) -> None:
        self.name = name
        self.Input: MiddlewareInputPhase[TContext, TResult, TEvent] = MiddlewareInputPhase(self)
        self.Wrap: MiddlewareWrapPhase[TContext, TResult, TEvent] = MiddlewareWrapPhase(self)
        self.Output: MiddlewareOutputPhase[TContext, TResult, TEvent] = MiddlewareOutputPhase(self)

    def __repr__(self) -> str:
        return f"MiddlewareStage(name={self.name!r})"

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


MiddlewareNext = Callable[[Any], AsyncGenerator[Any, None]]
MiddlewareHandler = Callable[[Any, MiddlewareNext], AsyncGenerator[Any, None]]
MiddlewareInputHandler = Callable[[Any], Any | Awaitable[Any]]
# Output handlers take and return a MiddlewareResult wrapping the result event.
MiddlewareOutputHandler = Callable[
    ["MiddlewareResult[Any]"], "MiddlewareResult[Any] | Awaitable[MiddlewareResult[Any]]"
]
