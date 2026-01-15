"""Multi-agent execution lifecycle events for hook system integration.

Deprecated: Use strands.hooks.multiagent instead.
"""

import warnings
from typing import Any

from ....hooks.multiagent.events import (
    AfterMultiAgentInvocationEvent,
    AfterNodeCallEvent,
    BeforeMultiAgentInvocationEvent,
    BeforeNodeCallEvent,
    MultiAgentInitializedEvent,
)

_DEPRECATED_ALIASES = {
    "MultiAgentInitializedEvent": MultiAgentInitializedEvent,
    "BeforeNodeCallEvent": BeforeNodeCallEvent,
    "BeforeMultiAgentInvocationEvent": BeforeMultiAgentInvocationEvent,
    "AfterNodeCallEvent": AfterNodeCallEvent,
    "AfterMultiAgentInvocationEvent": AfterMultiAgentInvocationEvent,
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_ALIASES:
        warnings.warn(
            f"{name} has been moved to production with an updated name. "
            f"Use {_DEPRECATED_ALIASES[name].__name__} from strands.hooks.multiagent instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEPRECATED_ALIASES[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
