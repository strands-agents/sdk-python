"""Experimental hook functionality that has not yet reached stability."""

from typing import Any

from .events import (
    BidiAfterConnectionRestartEvent,
    BidiAfterInvocationEvent,
    BidiAgentInitializedEvent,
    BidiBeforeConnectionRestartEvent,
    BidiBeforeInvocationEvent,
    BidiInterruptionEvent,
    BidiMessageAddedEvent,
)

# Deprecated aliases are accessed via __getattr__ to emit warnings only on use


def __getattr__(name: str) -> Any:
    from . import events

    return getattr(events, name)


__all__ = [
    "BeforeToolInvocationEvent",
    "AfterToolInvocationEvent",
    "BeforeModelInvocationEvent",
    "AfterModelInvocationEvent",
    # BidiAgent hooks
    "BidiAgentInitializedEvent",
    "BidiBeforeInvocationEvent",
    "BidiAfterInvocationEvent",
    "BidiMessageAddedEvent",
    "BidiInterruptionEvent",
    "BidiBeforeConnectionRestartEvent",
    "BidiAfterConnectionRestartEvent",
]
