"""Private async execution utilities."""

import asyncio
import contextvars
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Awaitable, Callable, TypeVar

T = TypeVar("T")

# Try to import greenback; set availability flag
# Set STRANDS_DISABLE_GREENBACK=1 to force the ThreadPoolExecutor fallback (for testing/debugging)
_GREENBACK_AVAILABLE = False
try:
    import greenback

    if os.environ.get("STRANDS_DISABLE_GREENBACK", "").lower() not in ("1", "true", "yes"):
        _GREENBACK_AVAILABLE = True
except ImportError:
    greenback = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import greenback


def run_async(async_func: Callable[[], Awaitable[T]]) -> T:
    """Run an async function, using greenback if available or a separate thread otherwise.

    If greenback is installed and a portal has been set up (via `await greenback.ensure_portal()`),
    this function uses greenback to await the async function on the current event loop. This allows
    async tools to access resources bound to the main event loop.

    Otherwise, this function uses ThreadPoolExecutor to run the async code in a separate thread
    with a new event loop, which isolates the async execution but cannot access main-loop resources.

    Set the environment variable STRANDS_DISABLE_GREENBACK=1 to force the ThreadPoolExecutor
    fallback even when greenback is installed (useful for testing or debugging).

    Args:
        async_func: A callable that returns an awaitable.

    Returns:
        The result of the async function.
    """
    # Use greenback if available and a portal is active
    if _GREENBACK_AVAILABLE and greenback.has_portal():
        return greenback.await_(async_func())

    # Fall back to ThreadPoolExecutor approach
    async def execute_async() -> T:
        return await async_func()

    def execute() -> T:
        return asyncio.run(execute_async())

    with ThreadPoolExecutor() as executor:
        context = contextvars.copy_context()
        future = executor.submit(context.run, execute)
        return future.result()
