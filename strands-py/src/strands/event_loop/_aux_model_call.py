"""Shared instrumentation for SDK-internal (auxiliary) model calls.

Auxiliary model calls — summarization, routing classification, memory extraction —
happen outside the agent's main event loop, so they bypass its hooks and metrics.
:func:`instrument_aux_model_call` wraps an auxiliary call's event stream to fire the
``Before/AfterAuxModelCallEvent`` hook pair and roll token usage into the owning
agent's :class:`~strands.telemetry.metrics.EventLoopMetrics`, tagged by source.
"""

import logging
from collections.abc import AsyncGenerator, AsyncIterable, Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from ..hooks import AfterAuxModelCallEvent, BeforeAuxModelCallEvent
from ..types.content import Messages
from ..types.exceptions import AuxModelCallCancelledException

if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)

TEvent = TypeVar("TEvent")


async def instrument_aux_model_call(
    events: AsyncIterable[TEvent],
    *,
    source: str,
    agent: "Agent | None",
    messages: Messages | None = None,
    invocation_state: dict[str, Any] | None = None,
) -> AsyncGenerator[TEvent, None]:
    """Wrap an auxiliary model call's event stream with hooks and metrics.

    Fires ``BeforeAuxModelCallEvent`` before consuming the stream and
    ``AfterAuxModelCallEvent`` when it completes (successfully or not), and adds the
    usage from the stream's terminal ``{"stop": (stop_reason, message, usage, metrics)}``
    event to ``agent.event_loop_metrics`` under ``source``. Events are yielded through
    unchanged. When ``agent`` is None (no owning agent is reachable from the call site),
    the stream passes through uninstrumented.

    Args:
        events: The auxiliary call's event stream (e.g. from ``process_stream``,
            ``stream_messages``, or ``Model.structured_output``). Streams without a
            ``"stop"`` event are passed through; hooks still fire, but no usage is
            recorded and ``stop_response`` is None.
        source: The auxiliary feature making the call, e.g. ``"summarization"``.
        agent: The agent this call is made on behalf of, or None.
        messages: The messages sent to the model, exposed on the Before event.
        invocation_state: Invocation state to expose on the hook events, when the call
            site has access to it.

    Yields:
        The events from the wrapped stream, unchanged.

    Raises:
        AuxModelCallCancelledException: If a ``BeforeAuxModelCallEvent`` callback set
            ``cancel``. The After event does not fire in this case, matching the
            hook-pair contract for short-circuited Before events.
    """
    if agent is None:
        async for event in events:
            yield event
        return

    resolved_invocation_state = invocation_state if invocation_state is not None else {}

    before_event = BeforeAuxModelCallEvent(
        agent=agent,
        source=source,
        messages=messages,
        invocation_state=resolved_invocation_state,
    )
    await agent.hooks.invoke_callbacks_async(before_event)

    if before_event.cancel:
        cancel_message = (
            before_event.cancel if isinstance(before_event.cancel, str) else "auxiliary model call cancelled by hook"
        )
        raise AuxModelCallCancelledException(cancel_message)

    stop: Any = None
    try:
        async for event in events:
            if isinstance(event, Mapping) and event.get("stop") is not None:
                stop = event["stop"]
            yield event
    except Exception as error:
        after_error_event = AfterAuxModelCallEvent(
            agent=agent,
            source=source,
            invocation_state=resolved_invocation_state,
            exception=error,
        )
        await agent.hooks.invoke_callbacks_async(after_error_event)
        raise

    stop_response: AfterAuxModelCallEvent.ModelStopResponse | None = None
    if stop is not None:
        stop_reason, message, usage, _metrics = stop
        agent.event_loop_metrics.update_usage(usage, source=source)
        stop_response = AfterAuxModelCallEvent.ModelStopResponse(
            message=message,
            stop_reason=stop_reason,
            usage=usage,
        )
    else:
        logger.debug("source=<%s> | auxiliary model call stream reported no stop event, skipping usage", source)

    after_event = AfterAuxModelCallEvent(
        agent=agent,
        source=source,
        invocation_state=resolved_invocation_state,
        stop_response=stop_response,
    )
    await agent.hooks.invoke_callbacks_async(after_event)
