"""Intervention action types and factory functions.

Each action represents a typed decision that a handler returns after evaluating
an event. The framework uses these to compose decisions across multiple handlers.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..hooks.events import (
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
)

LifecycleEvent = (
    BeforeInvocationEvent | BeforeToolCallEvent | AfterToolCallEvent | BeforeModelCallEvent | AfterModelCallEvent
)

_APPROVED_RESPONSES = {"y", "yes"}


def default_evaluate(response: Any) -> bool:
    """Default evaluate function for the confirm action.

    Accepts: True, 'y'/'yes' (case-insensitive, whitespace-trimmed).

    Args:
        response: The human's response value to evaluate.

    Returns:
        True if the response is considered an approval, False otherwise.
    """
    if response is True:
        return True
    if isinstance(response, str):
        return response.lower().strip() in _APPROVED_RESPONSES
    return False


@dataclass(frozen=True)
class Proceed:
    """Allow the operation to continue unchanged.

    Args:
        reason: Optional metadata for debugging/logging. Not shown to the model.
    """

    type: str = field(default="proceed", init=False)
    reason: str | None = None


@dataclass(frozen=True)
class Deny:
    """Block the operation. The reason is shown to the model as the cancellation message."""

    type: str = field(default="deny", init=False)
    reason: str = ""


@dataclass(frozen=True)
class Guide:
    """Provide feedback to steer behavior.

    On beforeToolCall/beforeInvocation, sets cancel so the model sees the feedback.
    On beforeModelCall, injects feedback as a user message.
    On afterModelCall, the response is discarded and the model retries with feedback.

    .. warning::
        On ``after_model_call``, Guide triggers a model retry. Handlers **must** ensure
        convergence (e.g., by tracking retry count and escalating to Deny after repeated
        failures). The framework imposes no retry cap on guide-triggered retries.

    .. note::
        On ``before_model_call`` and ``after_model_call``, guidance messages are injected
        directly into ``agent.messages`` and bypass session management. Session managers
        will not track these injected messages.
    """

    type: str = field(default="guide", init=False)
    feedback: str = ""
    reason: str | None = None


@dataclass(frozen=True)
class Confirm:
    """Request human approval before proceeding. Only supported on beforeToolCall.

    Two modes depending on whether response is provided:
    - With response: passed as a preemptive value to the interrupt system, agent never pauses.
    - Without response: breaks out of the agent loop to pause for external resume.
    """

    type: str = field(default="confirm", init=False)
    prompt: str = ""
    reason: str | None = None
    response: Any = None
    evaluate: Callable[[Any], bool] = field(default=default_evaluate)


@dataclass(frozen=True)
class Transform:
    """Modify event content in-place.

    The apply function mutates the event before execution proceeds.
    Later handlers in the pipeline see the transformed content.
    """

    type: str = field(default="transform", init=False)
    apply: Callable[[LifecycleEvent], None] = field(default=lambda e: None)
    reason: str | None = None


InterventionAction = Proceed | Deny | Guide | Confirm | Transform
"""Union of all intervention actions a handler can return.

Action-to-event compatibility matrix::

    | Action    | before_invocation | before_tool_call | before_model_call | after_tool_call | after_model_call |
    |-----------|-------------------|------------------|-------------------|-----------------|------------------|
    | Proceed   | —                 | —                | —                 | —               | —                |
    | Deny      | cancel            | cancel           | cancel            | —               | —                |
    | Guide     | cancel+           | cancel+          | inject            | —               | inject + retry   |
    | Confirm   | —                 | confirm          | —                 | —               | —                |
    | Transform | apply             | apply            | apply             | apply           | apply            |

    — = no-op (warns at runtime)
    cancel = sets event.cancel/cancel_tool, short-circuits (remaining handlers skipped)
    cancel+ = sets cancel with accumulated feedback from all guiding handlers
    confirm = uses preemptive response or interrupt, checks with evaluate, sets cancel if denied
    inject = appends accumulated feedback as a user message so the model sees it on this call
    inject + retry = appends accumulated feedback and retries so the model sees guidance
    apply = calls action.apply(event) for in-place mutation, later handlers see the change
"""


def proceed(*, reason: str | None = None) -> Proceed:
    """Allow the operation to continue.

    Args:
        reason: Optional metadata for debugging/logging. Not shown to the model.
    """
    return Proceed(reason=reason)


def deny(reason: str) -> Deny:
    """Block the operation.

    Args:
        reason: Why the operation was blocked. Shown to the model as the cancellation message.
    """
    return Deny(reason=reason)


def guide(feedback: str, *, reason: str | None = None) -> Guide:
    """Provide feedback to steer behavior.

    Args:
        feedback: The guidance text shown to the model.
        reason: Optional metadata for debugging/logging. Not shown to the model.
    """
    return Guide(feedback=feedback, reason=reason)


def confirm(
    prompt: str,
    *,
    reason: str | None = None,
    response: Any = None,
    evaluate: Callable[[Any], bool] | None = None,
) -> Confirm:
    """Request human approval before proceeding (before_tool_call only).

    Args:
        prompt: Message shown to the user/system requesting approval.
        reason: Optional metadata for debugging/logging.
        response: Preemptive response value. If not None, the agent does not pause —
            the response is passed directly to ``evaluate``. None is reserved as the
            "no response provided" sentinel.
        evaluate: Predicate that determines approval. Receives the response value.
            Defaults to accepting True, 'y', or 'yes' (case-insensitive).
    """
    return Confirm(
        prompt=prompt,
        reason=reason,
        response=response,
        evaluate=evaluate if evaluate is not None else default_evaluate,
    )


def transform(apply: Callable[[LifecycleEvent], None], *, reason: str | None = None) -> Transform:
    """Modify event content in-place.

    Args:
        apply: Function that mutates the event. Later handlers see the change.
        reason: Optional metadata for debugging/logging. Not shown to the model.
    """
    return Transform(apply=apply, reason=reason)
