"""Steering context protocols for contextual guidance.

Defines protocols for context callbacks and providers that populate
steering context data used by handlers to make guidance decisions.

Architecture:
    SteeringContextCallback → Handler.steering_context → SteeringHandler.steer()
            ↓                           ↓                         ↓
    Update local context        Store in handler         Access via self.steering_context

Context lifecycle:
    1. Handler registers context callbacks for hook events
    2. Callbacks update handler's local steering_context on events
    3. Handler accesses self.steering_context in steer() method
    4. Context persists across calls within handler instance

Implementation:
    Each handler maintains its own JSONSerializableDict context.
    Callbacks are registered per handler instance for isolation.
    Providers can supply multiple callbacks for different events.
"""

import logging
from dataclasses import dataclass, field
from typing import Protocol, TypeVar, get_args, get_origin

from ....hooks.registry import HookEvent
from ....types.json_dict import JSONSerializableDict

logger = logging.getLogger(__name__)


@dataclass
class SteeringContext:
    """Container for steering context data."""

    """Container for steering context data.
    
    This class should not be instantiated directly - it is intended for internal use only.
    """

    data: JSONSerializableDict = field(default_factory=JSONSerializableDict)


EventType = TypeVar("EventType", bound=HookEvent, contravariant=True)


class SteeringContextCallback(Protocol[EventType]):
    """Protocol for steering context update callbacks."""

    @property
    def event_type(self) -> type[EventType]:
        """Return the event type this callback handles."""
        for base in self.__class__.__orig_bases__:
            if get_origin(base) is SteeringContextCallback:
                return get_args(base)[0]
        raise ValueError("Could not determine event type from generic parameter")

    def __call__(self, event: EventType, steering_context: "SteeringContext", **kwargs) -> None:
        """Update steering context based on hook event.

        Args:
            event: The hook event that triggered the callback
            steering_context: The steering context to update
            **kwargs: Additional keyword arguments for context updates
        """
        ...


class SteeringContextProvider(Protocol):
    """Protocol for context providers that handle multiple event types."""

    def context_providers(self, **kwargs) -> list[SteeringContextCallback]:
        """Return list of context callbacks with event types extracted from generics."""
        ...
