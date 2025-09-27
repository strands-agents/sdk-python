"""Experimental hook events emitted as part of invoking Agents.

This module defines the events that are emitted as Agents run through the lifecycle of a request.
"""

import warnings
from typing import TypeAlias

from ...hooks.events import AfterModelCallEvent, AfterToolCallEvent, BeforeModelCallEvent, BeforeToolCallEvent

warnings.warn(
    "These events have been moved to production with updated names. Use BeforeModelCallEvent, "
    "AfterModelCallEvent, BeforeToolCallEvent, and AfterToolCallEvent from strands.hooks instead.",
    DeprecationWarning,
    stacklevel=2,
)

BeforeToolInvocationEvent: TypeAlias = BeforeToolCallEvent
AfterToolInvocationEvent: TypeAlias = AfterToolCallEvent
BeforeModelInvocationEvent: TypeAlias = BeforeModelCallEvent
AfterModelInvocationEvent: TypeAlias = AfterModelCallEvent
