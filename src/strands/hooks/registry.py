"""Hook registry system for managing event callbacks in the Strands Agent SDK.

This module provides the core infrastructure for the typed hook system, enabling
composable extension of agent functionality through strongly-typed event callbacks.
The registry manages the mapping between event types and their associated callback
functions, supporting both individual callback registration and bulk registration
via hook provider objects.
"""

import inspect
import logging
from collections.abc import Awaitable, Generator
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    TypeVar,
    get_type_hints,
    overload,
    runtime_checkable,
)

from ..interrupt import Interrupt, InterruptException

if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)


@dataclass
class BaseHookEvent:
    """Base class for all hook events."""

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


@dataclass
class HookEvent(BaseHookEvent):
    """Base class for single agent hook events.

    Attributes:
        agent: The agent instance that triggered this event.
    """

    agent: "Agent"


TEvent = TypeVar("TEvent", bound=BaseHookEvent, contravariant=True)
"""Generic for adding callback handlers - contravariant to allow adding handlers which take in base classes."""

TInvokeEvent = TypeVar("TInvokeEvent", bound=BaseHookEvent)
"""Generic for invoking events - non-contravariant to enable returning events."""


@runtime_checkable
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

        # Or

        async def my_callback(event: StartRequestEvent) -> None:
            # await an async operation
        ```
    """

    def __call__(self, event: TEvent) -> None | Awaitable[None]:
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
        self._registered_callbacks: dict[type, list[HookCallback]] = {}

    @overload
    def add_callback(self, callback: HookCallback[TEvent]) -> None: ...

    @overload
    def add_callback(self, event_type: type[TEvent], callback: HookCallback[TEvent]) -> None: ...

    def add_callback(
        self,
        event_type: type[TEvent] | HookCallback[TEvent] | None = None,
        callback: HookCallback[TEvent] | None = None,
    ) -> None:
        """Register a callback function for a specific event type.

        This method supports two call patterns:
        1. ``add_callback(callback)`` - Event type inferred from callback's type hint
        2. ``add_callback(event_type, callback)`` - Event type specified explicitly

        Args:
            event_type: The class type of events this callback should handle.
                When using the single-argument form, pass the callback here instead.
            callback: The callback function to invoke when events of this type occur.

        Raises:
            ValueError: If event_type is not provided and cannot be inferred from
                the callback's type hints, or if AgentInitializedEvent is registered
                with an async callback.

        Example:
            ```python
            def my_handler(event: StartRequestEvent):
                print("Request started")

            # With explicit event type
            registry.add_callback(StartRequestEvent, my_handler)

            # With event type inferred from type hint
            registry.add_callback(my_handler)
            ```
        """
        resolved_callback: HookCallback[TEvent]
        resolved_event_type: type[TEvent]

        # Support both add_callback(callback) and add_callback(event_type, callback)
        if callback is None:
            if event_type is None:
                raise ValueError("callback is required")
            # First argument is actually the callback, infer event_type
            if callable(event_type) and not isinstance(event_type, type):
                resolved_callback = event_type
                resolved_event_type = self._infer_event_type(resolved_callback)
            else:
                raise ValueError("callback is required when event_type is a type")
        elif event_type is None:
            # callback provided but event_type is None - infer it
            resolved_callback = callback
            resolved_event_type = self._infer_event_type(callback)
        else:
            # Both provided - event_type should be a type
            if isinstance(event_type, type):
                resolved_callback = callback
                resolved_event_type = event_type
            else:
                raise ValueError("event_type must be a type when callback is provided")

        # Related issue: https://github.com/strands-agents/sdk-python/issues/330
        if resolved_event_type.__name__ == "AgentInitializedEvent" and inspect.iscoroutinefunction(resolved_callback):
            raise ValueError("AgentInitializedEvent can only be registered with a synchronous callback")

        callbacks = self._registered_callbacks.setdefault(resolved_event_type, [])
        callbacks.append(resolved_callback)

    def _infer_event_type(self, callback: HookCallback[TEvent]) -> type[TEvent]:
        """Infer the event type from a callback's type hints.

        Args:
            callback: The callback function to inspect.

        Returns:
            The event type inferred from the callback's first parameter type hint.

        Raises:
            ValueError: If the event type cannot be inferred from the callback's type hints.
        """
        try:
            hints = get_type_hints(callback)
        except Exception as e:
            logger.debug("callback=<%s>, error=<%s> | failed to get type hints", callback, e)
            raise ValueError(
                "failed to get type hints for callback | cannot infer event type, please provide event_type explicitly"
            ) from e

        # Get the first parameter's type hint
        sig = inspect.signature(callback)
        params = list(sig.parameters.values())

        if not params:
            raise ValueError(
                "callback has no parameters | cannot infer event type, please provide event_type explicitly"
            )

        first_param = params[0]
        type_hint = hints.get(first_param.name)

        if type_hint is None:
            raise ValueError(
                f"parameter=<{first_param.name}> has no type hint | "
                "cannot infer event type, please provide event_type explicitly"
            )

        # Handle single type
        if isinstance(type_hint, type) and issubclass(type_hint, BaseHookEvent):
            return type_hint  # type: ignore[return-value]

        raise ValueError(
            f"parameter=<{first_param.name}>, type=<{type_hint}> | type hint must be a subclass of BaseHookEvent"
        )

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

    async def invoke_callbacks_async(self, event: TInvokeEvent) -> tuple[TInvokeEvent, list[Interrupt]]:
        """Invoke all registered callbacks for the given event.

        This method finds all callbacks registered for the event's type and
        invokes them in the appropriate order. For events with should_reverse_callbacks=True,
        callbacks are invoked in reverse registration order. Any exceptions raised by callback
        functions will propagate to the caller.

        Additionally, this method aggregates interrupts raised by the user to instantiate human-in-the-loop workflows.

        Args:
            event: The event to dispatch to registered callbacks.

        Returns:
            The event dispatched to registered callbacks and any interrupts raised by the user.

        Raises:
            ValueError: If interrupt name is used more than once.

        Example:
            ```python
            event = StartRequestEvent(agent=my_agent)
            await registry.invoke_callbacks_async(event)
            ```
        """
        interrupts: dict[str, Interrupt] = {}

        for callback in self.get_callbacks_for(event):
            try:
                if inspect.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)

            except InterruptException as exception:
                interrupt = exception.interrupt
                if interrupt.name in interrupts:
                    message = f"interrupt_name=<{interrupt.name}> | interrupt name used more than once"
                    logger.error(message)
                    raise ValueError(message) from exception

                # Each callback is allowed to raise their own interrupt.
                interrupts[interrupt.name] = interrupt

        return event, list(interrupts.values())

    def invoke_callbacks(self, event: TInvokeEvent) -> tuple[TInvokeEvent, list[Interrupt]]:
        """Invoke all registered callbacks for the given event.

        This method finds all callbacks registered for the event's type and
        invokes them in the appropriate order. For events with should_reverse_callbacks=True,
        callbacks are invoked in reverse registration order. Any exceptions raised by callback
        functions will propagate to the caller.

        Additionally, this method aggregates interrupts raised by the user to instantiate human-in-the-loop workflows.

        Args:
            event: The event to dispatch to registered callbacks.

        Returns:
            The event dispatched to registered callbacks and any interrupts raised by the user.

        Raises:
            RuntimeError: If at least one callback is async.
            ValueError: If interrupt name is used more than once.

        Example:
            ```python
            event = StartRequestEvent(agent=my_agent)
            registry.invoke_callbacks(event)
            ```
        """
        callbacks = list(self.get_callbacks_for(event))
        interrupts: dict[str, Interrupt] = {}

        if any(inspect.iscoroutinefunction(callback) for callback in callbacks):
            raise RuntimeError(f"event=<{event}> | use invoke_callbacks_async to invoke async callback")

        for callback in callbacks:
            try:
                callback(event)
            except InterruptException as exception:
                interrupt = exception.interrupt
                if interrupt.name in interrupts:
                    message = f"interrupt_name=<{interrupt.name}> | interrupt name used more than once"
                    logger.error(message)
                    raise ValueError(message) from exception

                # Each callback is allowed to raise their own interrupt.
                interrupts[interrupt.name] = interrupt

        return event, list(interrupts.values())

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
