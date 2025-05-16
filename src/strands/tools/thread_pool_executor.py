"""Thread pool execution management for parallel tool calls."""

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Iterator, Optional

from ..types.event_loop import Future, ParallelToolExecutorInterface


class ThreadPoolExecutorWrapper(ParallelToolExecutorInterface):
    """Wrapper around ThreadPoolExecutor to implement the strands.types.event_loop.ParallelToolExecutorInterface.

    This class adapts Python's standard ThreadPoolExecutor to conform to the SDK's ParallelToolExecutorInterface,
    allowing it to be used for parallel tool execution within the agent event loop. It provides methods for submitting
    tasks, monitoring their completion, and shutting down the executor.

    Attributes:
        thread_pool: The underlying ThreadPoolExecutor instance.
    """

    def __init__(self, thread_pool: ThreadPoolExecutor):
        """Initialize with a ThreadPoolExecutor instance.

        Args:
            thread_pool: The ThreadPoolExecutor to wrap.
        """
        self.thread_pool = thread_pool

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future:
        """Submit a callable to be executed with the given arguments.

        This method schedules the callable to be executed as fn(*args, **kwargs)
        and returns a Future instance representing the execution of the callable.

        Args:
            fn: The callable to execute.
            *args: Positional arguments for the callable.
            **kwargs: Keyword arguments for the callable.

        Returns:
            A Future instance representing the execution of the callable.
        """
        return self.thread_pool.submit(fn, *args, **kwargs)

    def as_completed(self, futures: Iterable[Future], timeout: Optional[int] = None) -> Iterator[Future]:
        """Return an iterator over the futures as they complete.

        The returned iterator yields futures as they complete (finished or cancelled).

        Args:
            futures: The futures to iterate over.
            timeout: The maximum number of seconds to wait.
                None means no limit.

        Returns:
            An iterator yielding futures as they complete.

        Raises:
            concurrent.futures.TimeoutError: If the timeout is reached.
        """
        return concurrent.futures.as_completed(futures, timeout=timeout)  # type: ignore

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool executor.

        Args:
            wait: If True, waits until all running futures have finished executing.
        """
        self.thread_pool.shutdown(wait=wait)
