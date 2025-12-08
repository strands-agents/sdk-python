"""Manage a group of async tasks.

This is intended to mimic the behaviors of asyncio.TaskGroup released in Python 3.11.

- Docs: https://docs.python.org/3/library/asyncio-task.html#task-groups
"""

import asyncio
from typing import Any, Coroutine


class _TaskGroup:
    """Shim of asyncio.TaskGroup for use in Python 3.10.

    Attributes:
        _tasks: List of tasks in group.
    """

    _tasks: list[asyncio.Task]

    def create_task(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
        """Create an async task and add to group.

        Returns:
            The created task.
        """
        task = asyncio.create_task(coro)
        self._tasks.append(task)
        return task

    async def __aenter__(self) -> "_TaskGroup":
        """Setup self managed task group context."""
        self._tasks = []
        return self

    async def __aexit__(self, *_: Any) -> None:
        """Execute tasks in group.

        The following execution rules are enforced:
        - The context stops executing all tasks if at least one task raises an Exception or the context is cancelled.
        - The context re-raises Exceptions to the caller.
        - The context re-raises CancelledErrors to the caller only if the context itself was cancelled.
        """
        try:
            await asyncio.gather(*self._tasks)

        except (Exception, asyncio.CancelledError) as error:
            for task in self._tasks:
                task.cancel()

            await asyncio.gather(*self._tasks, return_exceptions=True)

            if not isinstance(error, asyncio.CancelledError):
                raise

            context_task = asyncio.current_task()
            if context_task and context_task.cancelling() > 0:  # context itself was cancelled
                raise

        finally:
            self._tasks = []
