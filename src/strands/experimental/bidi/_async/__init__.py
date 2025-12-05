"""Utilities for async operations."""

from typing import Awaitable, Callable

from ._task_pool import _TaskPool

__all__ = ["_TaskPool"]


async def stop_all(*funcs: Callable[..., Awaitable[None]]) -> None:
    """Call all stops in sequence and aggregate errors.

    A failure in one stop call will not block subsequent stop calls.

    Args:
        funcs: Stop functions to call in sequence.

    Raises:
        RuntimeError: If any stop function raises an exception.
    """
    exceptions = []
    for func in funcs:
        try:
            await func()
        except Exception as exception:
            exceptions.append(exception)

    if exceptions:
        exceptions.append(RuntimeError("failed stop sequence"))
        for i in range(1, len(exceptions)):
            exceptions[i].__cause__ = exceptions[i - 1]

        raise exceptions[-1]
