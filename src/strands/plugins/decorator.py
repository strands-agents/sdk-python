"""Hook decorator for Plugin methods.

This module provides the @hook decorator that marks methods as hook callbacks
for automatic registration when the plugin is attached to an agent.

The @hook decorator performs several functions:

1. Marks methods as hook callbacks for automatic discovery by Plugin base class
2. Infers event types from the callback's type hints (consistent with HookRegistry.add_callback)
3. Supports both @hook and @hook() syntax
4. Supports union types for multiple event types (e.g., BeforeModelCallEvent | AfterModelCallEvent)
5. Stores hook metadata on the decorated method for later discovery

Example:
    ```python
    from strands.plugins import Plugin, hook
    from strands.hooks import BeforeModelCallEvent, AfterModelCallEvent

    class MyPlugin(Plugin):
        name = "my-plugin"

        @hook
        def on_model_call(self, event: BeforeModelCallEvent):
            print(event)

        @hook
        def on_any_model_event(self, event: BeforeModelCallEvent | AfterModelCallEvent):
            print(event)
    ```
"""

from collections.abc import Callable
from typing import TypeVar, overload

from ..hooks._type_inference import infer_event_types

# Type for wrapped function
T = TypeVar("T", bound=Callable[..., object])


# Handle @hook
@overload
def hook(__func: T) -> T: ...


# Handle @hook()
@overload
def hook() -> Callable[[T], T]: ...


def hook(  # type: ignore[misc]
    func: T | None = None,
) -> T | Callable[[T], T]:
    """Decorator that marks a method as a hook callback for automatic registration.

    This decorator enables declarative hook registration in Plugin classes. When a
    Plugin is attached to an agent, methods marked with @hook are automatically
    discovered and registered with the agent's hook registry.

    The event type is inferred from the callback's type hint on the first parameter
    (after 'self' for instance methods). Union types are supported for registering
    a single callback for multiple event types.

    The decorator can be used in two ways:
    - As a simple decorator: `@hook`
    - With parentheses: `@hook()`

    Args:
        func: The function to decorate. When used as a simple decorator, this is
            the function being decorated. When used with parentheses, this will be None.

    Returns:
        The decorated function with hook metadata attached.

    Raises:
        ValueError: If the event type cannot be inferred from type hints, or if
            the type hint is not a valid HookEvent subclass.

    Example:
        ```python
        class MyPlugin(Plugin):
            name = "my-plugin"

            @hook
            def on_model_call(self, event: BeforeModelCallEvent):
                print(f"Model called: {event}")

            @hook
            def on_any_event(self, event: BeforeModelCallEvent | AfterModelCallEvent):
                print(f"Event: {type(event).__name__}")
        ```
    """

    def decorator(f: T) -> T:
        # Infer event types from type hints (skip 'self' for methods)
        event_types = infer_event_types(f, skip_self=True)

        # Store hook metadata on the function
        f._hook_event_types = event_types  # type: ignore[attr-defined]

        return f

    # Handle both @hook and @hook() syntax
    if func is None:
        return decorator

    return decorator(func)
