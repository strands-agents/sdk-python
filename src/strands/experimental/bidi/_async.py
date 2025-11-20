"""Utilities for async operations."""

from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar, cast

from .types._async import Startable

F = TypeVar("F", bound=Callable[..., Awaitable[None]])


def start(func: F) -> F:
    """Call stop if pairing start call fails.

    Any resources that did successfully start will still have an opportunity to stop cleanly.

    Args:
        func: Start function to wrap.
    """

    @wraps(func)
    async def wrapper(self: Startable, *args: Any, **kwargs: Any) -> None:
        try:
            await func(self, *args, **kwargs)
        except Exception:
            await self.stop()
            raise

    return cast(F, wrapper)


async def stop(*funcs: F) -> None:
    """Call all stops in sequence and aggregate errors.

    A failure in one stop call will not block subsequent stop calls.

    Args:
        funcs: Stop functions to call in sequence.

    Raises:
        ExceptionGroup: If any stop function raises an exception.
    """
    exceptions = []
    for func in funcs:
        try:
            await func()
        except Exception as exception:
            exceptions.append(exception)

    if exceptions:
        raise ExceptionGroup("failed stop sequence", exceptions)  # type: ignore  # noqa: F821
