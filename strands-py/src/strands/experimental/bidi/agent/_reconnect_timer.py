"""Proactive reconnect timer for bidirectional streaming.

``_BidiReconnectTimer`` fires a warning then a deadline callback at caller-supplied offsets;
it holds no reconnect policy. ``resolve_deadline_s`` derives the deadline from a provider's
declared ``BidiConnectionConfig``.
"""

import asyncio
import logging
from typing import Awaitable, Callable

from ..types.model import DEFAULT_RECONNECT_MARGIN_S, BidiConnectionConfig

logger = logging.getLogger(__name__)


def resolve_deadline_s(connection_config: BidiConnectionConfig) -> float | None:
    """Resolve the reconnect deadline in seconds from a connection config.

    The deadline is ``max_connection_s`` minus the reconnect margin. Returns ``None`` when
    ``max_connection_s`` is not declared (no proactive timer).

    Args:
        connection_config: Provider-declared connection limit.

    Returns:
        Seconds from now until reconnect should fire, or ``None`` if no limit is declared.
    """
    max_connection_s = connection_config.get("max_connection_s")
    if max_connection_s is None:
        return None

    margin = connection_config.get("reconnect_margin_s", DEFAULT_RECONNECT_MARGIN_S)
    # Clamp to zero so a limit smaller than the margin reconnects immediately.
    return max(max_connection_s - margin, 0.0)


class _BidiReconnectTimer:
    """Fire a warning then a deadline callback ahead of a provider's connection limit.

    The clock is injectable so tests can drive timing without wall time.

    Attributes:
        _sleep: Injectable async sleep, defaults to ``asyncio.sleep``.
    """

    def __init__(
        self,
        on_warning: Callable[[float], Awaitable[None]],
        on_deadline: Callable[[], Awaitable[None]],
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        """Initialize the timer.

        Args:
            on_warning: Awaitable called with seconds-left when the warning lead elapses.
            on_deadline: Awaitable called when the reconnect deadline elapses.
            sleep: Injectable async sleep (for tests). Defaults to ``asyncio.sleep``.
        """
        self._on_warning = on_warning
        self._on_deadline = on_deadline
        self._sleep = sleep or asyncio.sleep
        self._task: asyncio.Task | None = None

    def arm(self, deadline_s: float, warning_lead_s: float) -> None:
        """Arm the warning and deadline timers, cancelling any previously armed cycle.

        Args:
            deadline_s: Seconds from now until the deadline callback fires.
            warning_lead_s: Seconds before the deadline to fire the warning callback.
        """
        self.cancel()
        self._task = asyncio.create_task(self._run(deadline_s, warning_lead_s))
        logger.debug(
            "deadline_s=<%.1f>, warning_lead_s=<%.1f> | proactive reconnect timer armed",
            deadline_s,
            warning_lead_s,
        )

    def cancel(self) -> None:
        """Cancel the armed timer, if any. Safe to call when idle."""
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _run(self, deadline_s: float, warning_lead_s: float) -> None:
        """Sleep until the warning lead, fire the warning, then fire the deadline.

        The warning fires ``warning_lead_s`` before the deadline. When the lead is zero
        or exceeds the deadline, the warning is emitted immediately and the remaining
        wait runs down to the deadline.
        """
        warning_at_s = max(deadline_s - warning_lead_s, 0.0)

        await self._sleep(warning_at_s)
        time_left_s = deadline_s - warning_at_s
        await self._on_warning(time_left_s)

        await self._sleep(deadline_s - warning_at_s)
        # Detach before the callback re-arms this timer; cancelling a live self-reference
        # would abort the reconnect the callback runs.
        self._task = None
        await self._on_deadline()
