"""Extraction triggers that control *when* a store's extraction runs.

A trigger is a self-attaching value object (see :class:`ExtractionTrigger`):
:meth:`~ExtractionTrigger.attach` wires whatever agent hooks the trigger needs
and calls :attr:`ExtractionTriggerContext.fire` when extraction should happen.
This module provides the two built-in triggers:

* :class:`InvocationTrigger` -- fire after every agent invocation (highest
  fidelity, most expensive when an extractor is configured).
* :class:`IntervalTrigger` -- fire once every ``turns`` invocations (a
  controllable middle ground; the high-water mark still picks up the skipped
  turns when the trigger does fire).
"""

from __future__ import annotations

from ...hooks.events import AfterInvocationEvent
from ...hooks.registry import HookOrder
from .types import ExtractionTrigger, ExtractionTriggerContext


class InvocationTrigger(ExtractionTrigger):
    """Runs extraction after every agent invocation.

    The highest-fidelity option: nothing said in a turn is missed. Also the most
    expensive when an :class:`~strands.memory.extraction.types.Extractor` is
    configured (a model call per turn) -- for a server-side-extraction backend
    with no extractor it is just a per-turn write.

    Example:
        ```python
        ExtractionConfig(trigger=[InvocationTrigger()])
        ```
    """

    name = "invocation"

    def attach(self, context: ExtractionTriggerContext) -> None:
        """Register an after-invocation callback that fires extraction.

        The callback runs after the SDK's own after-invocation hooks (e.g.
        session persistence) so extraction sees the fully settled turn. It
        returns ``None`` synchronously; the actual save runs in a task scheduled
        by ``fire``, so the firing hook never blocks the agent.

        Args:
            context: The agent to attach to and the fire callback bound to this
                trigger's store.
        """
        context.agent.add_hook(
            lambda event: context.fire(),
            AfterInvocationEvent,
            order=HookOrder.SDK_LAST,
        )


class IntervalTrigger(ExtractionTrigger):
    """Runs extraction every ``turns`` agent invocations.

    A controllable middle ground: extraction (and any model call it entails)
    happens on a cadence rather than every turn, while the high-water mark
    guarantees the messages from the skipped turns are still processed when the
    trigger does fire.

    Example:
        ```python
        ExtractionConfig(trigger=[IntervalTrigger(turns=5)])
        ```

    Attributes:
        name: Stable identifier for this trigger kind (``interval``).
    """

    name = "interval"

    def __init__(self, turns: int) -> None:
        """Initialize the trigger with a firing cadence.

        Args:
            turns: Run extraction once every this many invocations. Must be a
                positive integer.

        Raises:
            ValueError: If ``turns`` is not a positive integer. ``bool`` values
                are rejected even though ``bool`` is a subclass of ``int``.
        """
        # Reject bool explicitly (bool is a subclass of int) and any value < 1.
        if not isinstance(turns, int) or isinstance(turns, bool) or turns < 1:
            raise ValueError(f"IntervalTrigger: turns must be a positive integer, got {turns}")
        self._turns = turns

    def attach(self, context: ExtractionTriggerContext) -> None:
        """Register an after-invocation callback that fires every ``turns`` turns.

        Each call to ``attach`` creates a fresh closure counter, so one trigger
        instance attached to multiple stores keeps an independent count per
        attachment -- each store fires on its own schedule.

        Args:
            context: The agent to attach to and the fire callback bound to this
                trigger's store.
        """
        # Per-attach counter: each store this trigger is configured on gets its
        # own count via a fresh closure, so two stores sharing one
        # IntervalTrigger instance still fire independently.
        count = 0

        def _callback(event: AfterInvocationEvent) -> None:
            nonlocal count
            count += 1
            # `fire` is fire-and-forget (returns None); it dispatches extraction
            # in the background.
            if count % self._turns == 0:
                context.fire()

        context.agent.add_hook(_callback, AfterInvocationEvent, order=HookOrder.SDK_LAST)
