"""Shared instrumentation for SDK-internal (auxiliary) model calls.

Auxiliary model calls — summarization, routing classification, memory extraction —
happen outside the agent's main event loop, so they bypass its hooks and metrics.
:func:`instrument_auxiliary_model_call` wraps an auxiliary call's event stream to fire the
``Before/AfterAuxiliaryModelCallEvent`` hook pair and roll token usage into the owning
agent's :class:`~strands.telemetry.metrics.EventLoopMetrics`, tagged by source.
"""

import logging
from collections.abc import AsyncGenerator, AsyncIterable, Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from ..hooks import AfterAuxiliaryModelCallEvent, BeforeAuxiliaryModelCallEvent
from ..types.content import Messages
from ..types.event_loop import AuxiliaryModelCallSource
from ..types.exceptions import AuxiliaryModelCallCancelledException

if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)

TEvent = TypeVar("TEvent")


async def instrument_auxiliary_model_call(
    events: AsyncIterable[TEvent],
    *,
    source: AuxiliaryModelCallSource,
    agent: "Agent | None",
    messages: Messages,
    invocation_state: dict[str, Any] | None = None,
) -> AsyncGenerator[TEvent, None]:
    """Wrap an auxiliary model call's event stream with hooks and metrics.

    Fires ``BeforeAuxiliaryModelCallEvent`` before consuming the stream and
    ``AfterAuxiliaryModelCallEvent`` when it completes (successfully or not), and adds the
    usage from the stream's terminal ``{"stop": (stop_reason, message, usage, metrics)}``
    event to ``agent.event_loop_metrics`` under ``source``. Events are yielded through
    unchanged. When ``agent`` is None (no owning agent is reachable from the call site),
    the stream passes through uninstrumented.

    Args:
        events: The auxiliary call's event stream (e.g. from ``process_stream``,
            ``stream_messages``, or ``Model.structured_output``). Pass the stream
            *unstarted*: the hook-before-model-call guarantee holds because async
            generator bodies run only on first iteration, which happens here after
            the Before event — not because this helper defers anything itself.
            Streams without a
            ``"stop"`` event are passed through; hooks still fire, but no usage is
            recorded and ``stop_response`` is None. Note: only providers whose
            ``structured_output`` forwards stream events (e.g. Bedrock, Anthropic) emit
            the ``"stop"`` event on structured-output calls; for others the classifier's
            usage is not recorded today.
        source: The auxiliary feature making the call, e.g. ``"summarization"``.
        agent: The agent this call is made on behalf of, or None.
        messages: The messages sent to the model, exposed on the Before event.
        invocation_state: Invocation state to expose on the hook events, when the call
            site has access to it.

    Yields:
        The events from the wrapped stream, unchanged.

    Raises:
        AuxiliaryModelCallCancelledException: If a ``BeforeAuxiliaryModelCallEvent`` callback set
            ``cancel``. The After event does not fire in this case, matching the
            hook-pair contract for short-circuited Before events.
    """
    if agent is None:
        async for event in events:
            yield event
        return

    resolved_invocation_state = invocation_state if invocation_state is not None else {}

    before_event = BeforeAuxiliaryModelCallEvent(
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
        raise AuxiliaryModelCallCancelledException(cancel_message)

    stop: Any = None
    stop_response: AfterAuxiliaryModelCallEvent.ModelStopResponse | None = None
    # BaseException, not Exception: the After event must also fire when the stream is
    # cancelled or the generator is closed (e.g. the routing classifier's asyncio.wait_for
    # timeout), or paired setup/teardown hooks would leak on every timeout.
    try:
        async for event in events:
            if isinstance(event, Mapping) and event.get("stop") is not None:
                stop = event["stop"]
            yield event

        # Shape-check: in-tree streams always yield (stop_reason, message, usage, metrics),
        # but a third-party ``structured_output`` may emit anything under "stop".
        if isinstance(stop, tuple) and len(stop) == 4 and _is_usable_usage(stop[2]):
            stop_reason, message, usage, _metrics = stop
            agent.event_loop_metrics.update_usage(usage, source=source)
            stop_response = AfterAuxiliaryModelCallEvent.ModelStopResponse(
                message=message,
                stop_reason=stop_reason,
                usage=usage,
            )
        else:
            logger.debug("source=<%s> | auxiliary model call reported no usable stop event, skipping usage", source)
    except BaseException as error:
        after_error_event = AfterAuxiliaryModelCallEvent(
            agent=agent,
            source=source,
            invocation_state=resolved_invocation_state,
            exception=error,
        )
        await agent.hooks.invoke_callbacks_async(after_error_event)
        raise

    after_event = AfterAuxiliaryModelCallEvent(
        agent=agent,
        source=source,
        invocation_state=resolved_invocation_state,
        stop_response=stop_response,
    )
    await agent.hooks.invoke_callbacks_async(after_event)


def _is_usable_usage(usage: Any) -> bool:
    """Return True if the stop event's usage payload has the required ``Usage`` keys."""
    required_keys = {"inputTokens", "outputTokens", "totalTokens"}
    return isinstance(usage, dict) and required_keys <= usage.keys()
