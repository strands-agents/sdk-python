"""Hook registry system for managing event callbacks in the Strands Agent SDK.

This module provides the core infrastructure for the typed hook system, enabling
composable extension of agent functionality through strongly-typed event callbacks.
The registry manages the mapping between event types and their associated callback
functions, supporting both individual callback registration and bulk registration
via hook provider objects.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generator, Protocol, Type, TypeVar

if TYPE_CHECKING:
    from ..agent import Agent


@dataclass
class HookEvent:
    """Base class for all hook events.

    Attributes:
        agent: The agent instance that triggered this event.
    """

    agent: "Agent"

    @property
    def should_reverse_callbacks(self) -> bool:
        """Determine if callbacks for this event should be invoked in reverse order.

        Returns:
            False by default. Override to return True for events that should
            invoke callbacks in reverse order (e.g., cleanup/teardown events).
        """
        return False


T = TypeVar("T", bound=Callable)
TEvent = TypeVar("TEvent", bound=HookEvent)


class HookProvider(Protocol):
    """Protocol for objects that provide hook callbacks to an agent.

    Hook providers offer a composable way to extend agent functionality by
    subscribing to various events in the agent lifecycle. This protocol enables
    building reusable components that can hook into agent events.

    Example:
        ```python
        class MyHookProvider(HookProvider):
            def register_hooks(self, registry: HookRegistry) -> None:
                hooks.add_callback(StartRequestEvent, self.on_request_start)
                hooks.add_callback(EndRequestEvent, self.on_request_end)

        agent = Agent(hooks=[MyHookProvider()])
        ```
    """

    def register_hooks(self, registry: "HookRegistry") -> None:
        """Register callback functions for specific event types.

        Args:
            registry: The hook registry to register callbacks with.
        """
        ...


class HookCallback(Protocol):
    """Protocol for callback functions that handle hook events.

    Hook callbacks are functions that receive a single strongly-typed event
    argument and perform some action in response. They should not return
    values and any exceptions they raise will propagate to the caller.

    Example:
        ```python
        def my_callback(event: StartRequestEvent) -> None:
            print(f"Request started for agent: {event.agent.name}")
        ```
    """

    def __call__(self, event: Any) -> None:
        """Handle a hook event.

        Args:
            event: The strongly-typed event to handle.
        """
        ...


class HookRegistry:
    """Registry for managing hook callbacks associated with event types.

    The HookRegistry maintains a mapping of event types to callback functions
    and provides methods for registering callbacks and invoking them when
    events occur.

    The registry handles callback ordering, including reverse ordering for
    cleanup events, and provides type-safe event dispatching.
    """

    def __init__(self) -> None:
        """Initialize an empty hook registry."""
        self._registered_callbacks: dict[Type, list[HookCallback]] = {}

    def add_callback(self, event_type: Type, callback: HookCallback) -> None:
        """Register a callback function for a specific event type.

        Args:
            event_type: The class type of events this callback should handle.
            callback: The callback function to invoke when events of this type occur.

        Example:
            ```python
            def my_handler(event: StartRequestEvent):
                print("Request started")

            registry.add_callback(StartRequestEvent, my_handler)
            ```
        """
        callbacks = self._registered_callbacks.setdefault(event_type, [])
        callbacks.append(callback)

    def add_hook(self, hook: HookProvider) -> None:
        """Register all callbacks from a hook provider.

        This method allows bulk registration of callbacks by delegating to
        the hook provider's register_hooks method. This is the preferred
        way to register multiple related callbacks.

        Args:
            hook: The hook provider containing callbacks to register.

        Example:
            ```python
            class MyHooks(HookProvider):
                def register_hooks(self, registry: HookRegistry):
                    registry.add_callback(StartRequestEvent, self.on_start)
                    registry.add_callback(EndRequestEvent, self.on_end)

            registry.add_hook(MyHooks())
            ```
        """
        hook.register_hooks(self)

    def invoke_callbacks(self, event: TEvent) -> None:
        """Invoke all registered callbacks for the given event.

        This method finds all callbacks registered for the event's type and
        invokes them in the appropriate order. For events with is_after_callback=True,
        callbacks are invoked in reverse registration order.

        Args:
            event: The event to dispatch to registered callbacks.

        Raises:
            Any exceptions raised by callback functions will propagate to the caller.

        Example:
            ```python
            event = StartRequestEvent(agent=my_agent)
            registry.invoke_callbacks(event)
            ```
        """
        for callback in self.get_callbacks_for(event):
            callback(event)

    def get_callbacks_for(self, event: TEvent) -> Generator[HookCallback, None, None]:
        """Get callbacks registered for the given event in the appropriate order.

        This method returns callbacks in registration order for normal events,
        or reverse registration order for events that have is_after_callback=True.
        This enables proper cleanup ordering for teardown events.

        Args:
            event: The event to get callbacks for.

        Yields:
            Callback functions registered for this event type, in the appropriate order.

        Example:
            ```python
            event = EndRequestEvent(agent=my_agent)
            for callback in registry.get_callbacks_for(event):
                callback(event)
            ```
        """
        event_type = type(event)

        callbacks = self._registered_callbacks.get(event_type, [])
        if event.should_reverse_callbacks:
            yield from reversed(callbacks)
        else:
            yield from callbacks
