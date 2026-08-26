"""Linking of a caller-owned cancellation signal into the agent's internal one."""

import asyncio
import threading

_POLL_INTERVAL = 0.05


async def link_cancel_signal(external: threading.Event, internal: threading.Event) -> None:
    """Mirror an external cancellation event onto the agent's internal event.

    Args:
        external: Caller-owned event. Never set or cleared here.
        internal: The agent's event, which every cancellation checkpoint reads.
    """
    # threading.Event has no async notification hook. Poll on this loop rather than
    # running Event.wait() in an executor: cancelling that await cannot stop a worker
    # already blocked in Event.wait(), so completed invocations could strand worker threads.
    while not external.is_set():
        await asyncio.sleep(_POLL_INTERVAL)

    internal.set()
