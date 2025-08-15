"""Hook registry system for managing event callbacks in the Strands Agent SDK.

This module provides the core infrastructure for the typed hook system, enabling
composable extension of agent functionality through strongly-typed event callbacks.
The registry manages the mapping between event types and their associated callback
functions, supporting both individual callback registration and bulk registration
via hook provider objects.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generator, Generic, Protocol, Type, TypeVar

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

    def _can_write(self, name: str) -> bool:
        """Check if the given property can be written to.

        Args:
            name: The name of the property to check.

        Returns:
            True if the property can be written to, False otherwise.
        """
        return False

    def __post_init__(self) -> None:
        """Disallow writes to non-approved properties."""
        # This is needed as otherwise the class can't be initialized at all, so we trigger
        # this after class initialization
        super().__setattr__("_disallow_writes", True)

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent setting attributes on hook events.

        Raises:
            AttributeError: Always raised to prevent setting attributes on hook events.
        """
        #  Allow setting attributes:
        #    - during init (when __dict__) doesn't exist
        #    - if the subclass specifically said the property is writable
        if not hasattr(self, "_disallow_writes") or self._can_write(name):
            return super().__setattr__(name, value)

        raise AttributeError(f"Property {name} is not writable")


TEvent = TypeVar("TEvent", bound=HookEvent, contravariant=True)
"""Generic for adding callback handlers - contravariant to allow adding handlers which take in base classes."""

TInvokeEvent = TypeVar("TInvokeEvent", bound=HookEvent)
"""Generic for invoking events - non-contravariant to enable returning events."""


class HookProvider(Protocol):
    """Protocol for objects that provide hook callbacks to an agent.

    Hook providers offer a composable way to extend agent functionality by
    subscribing to various events in the agent lifecycle. This protocol enables
    building reusable components that can hook into agent events.

    Example:
        ```python
        class MyHookProvider(HookProvider):
            def register_hooks(self, registry: HookRegistry) -> None:
                registry.add_callback(StartRequestEvent, self.on_request_start)
                registry.add_callback(EndRequestEvent, self.on_request_end)

        agent = Agent(hooks=[MyHookProvider()])
        ```
    """

    def register_hooks(self, registry: "HookRegistry", **kwargs: Any) -> None:
        """Register callback functions for specific event types.

        Args:
            registry: The hook registry to register callbacks with.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        ...


class HookCallback(Protocol, Generic[TEvent]):
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

    def __call__(self, event: TEvent) -> None:
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

    def add_callback(self, event_type: Type[TEvent], callback: HookCallback[TEvent]) -> None:
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

    def invoke_callbacks(self, event: TInvokeEvent) -> TInvokeEvent:
        """Invoke all registered callbacks for the given event.

        This method finds all callbacks registered for the event's type and
        invokes them in the appropriate order. For events with should_reverse_callbacks=True,
        callbacks are invoked in reverse registration order. Any exceptions raised by callback
        functions will propagate to the caller.

        Args:
            event: The event to dispatch to registered callbacks.

        Returns:
            The event dispatched to registered callbacks.

        Example:
            ```python
            event = StartRequestEvent(agent=my_agent)
            registry.invoke_callbacks(event)
            ```
        """
        for callback in self.get_callbacks_for(event):
            callback(event)

        return event

    def has_callbacks(self) -> bool:
        """Check if the registry has any registered callbacks.

        Returns:
            True if there are any registered callbacks, False otherwise.

        Example:
            ```python
            if registry.has_callbacks():
                print("Registry has callbacks registered")
            ```
        """
        return bool(self._registered_callbacks)

    def get_callbacks_for(self, event: TEvent) -> Generator[HookCallback[TEvent], None, None]:
        """Get callbacks registered for the given event in the appropriate order.

        This method returns callbacks in registration order for normal events,
        or reverse registration order for events that have should_reverse_callbacks=True.
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
