"""Multi-agent execution lifecycle events for hook system integration.

Deprecated: Use strands.hooks.multiagent instead.
"""

import warnings

from ....hooks.multiagent.events import (
    AfterMultiAgentInvocationEvent,
    AfterNodeCallEvent,
    BeforeMultiAgentInvocationEvent,
    BeforeNodeCallEvent,
    MultiAgentInitializedEvent,
)

warnings.warn(
    "strands.experimental.hooks.multiagent.events is deprecated. Use strands.hooks.multiagent.events instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "AfterMultiAgentInvocationEvent",
    "AfterNodeCallEvent",
    "BeforeMultiAgentInvocationEvent",
    "BeforeNodeCallEvent",
    "MultiAgentInitializedEvent",
]
