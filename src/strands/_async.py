"""Private async execution utilities."""

import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


def run_async(async_func: Callable[[], Awaitable[T]]) -> T:
    """Run an async function in a separate thread to avoid event loop conflicts.

    This utility handles the common pattern of running async code from sync contexts
    by using ThreadPoolExecutor to isolate the async execution.

    Args:
        async_func: A callable that returns an awaitable

    Returns:
        The result of the async function
    """

    async def execute_async() -> T:
        return await async_func()

    def execute() -> T:
        return asyncio.run(execute_async())

    with ThreadPoolExecutor() as executor:
        context = contextvars.copy_context()
        future = executor.submit(context.run, execute)
        return future.result()
