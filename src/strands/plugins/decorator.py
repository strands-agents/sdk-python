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

import functools
import inspect
import logging
import types
from collections.abc import Callable
from typing import TypeVar, Union, cast, get_args, get_origin, get_type_hints, overload

from ..hooks.registry import BaseHookEvent, HookCallback, TEvent

logger = logging.getLogger(__name__)

# Type for wrapped function
T = TypeVar("T", bound=Callable[..., object])


def _infer_event_types(callback: HookCallback[TEvent]) -> list[type[TEvent]]:
    """Infer the event type(s) from a callback's type hints.

    Supports both single types and union types (A | B or Union[A, B]).

    This logic is adapted from HookRegistry._infer_event_types to provide
    consistent behavior for event type inference.

    Args:
        callback: The callback function to inspect.

    Returns:
        A list of event types inferred from the callback's first parameter type hint.

    Raises:
        ValueError: If the event type cannot be inferred from the callback's type hints,
            or if a union contains None or non-BaseHookEvent types.
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
        raise ValueError("callback has no parameters | cannot infer event type, please provide event_type explicitly")

    # For methods, skip 'self' parameter
    first_param = params[0]
    if first_param.name == "self" and len(params) > 1:
        first_param = params[1]

    type_hint = hints.get(first_param.name)

    if type_hint is None:
        raise ValueError(
            f"parameter=<{first_param.name}> has no type hint | "
            "cannot infer event type, please provide event_type explicitly"
        )

    # Check if it's a Union type (Union[A, B] or A | B)
    origin = get_origin(type_hint)
    if origin is Union or origin is types.UnionType:
        event_types: list[type[TEvent]] = []
        for arg in get_args(type_hint):
            if arg is type(None):
                raise ValueError("None is not a valid event type in union")
            if not (isinstance(arg, type) and issubclass(arg, BaseHookEvent)):
                raise ValueError(f"Invalid type in union: {arg} | must be a subclass of BaseHookEvent")
            event_types.append(cast(type[TEvent], arg))
        return event_types

    # Handle single type
    if isinstance(type_hint, type) and issubclass(type_hint, BaseHookEvent):
        return [cast(type[TEvent], type_hint)]

    raise ValueError(
        f"parameter=<{first_param.name}>, type=<{type_hint}> | type hint must be a subclass of BaseHookEvent"
    )


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
        # Infer event types from type hints
        event_types = _infer_event_types(f)

        # Store hook metadata on the function
        f._hook_event_types = event_types

        # Preserve original function metadata
        @functools.wraps(f)
        def wrapper(*args: object, **kwargs: object) -> object:
            return f(*args, **kwargs)

        # Copy hook metadata to wrapper
        wrapper._hook_event_types = event_types

        return cast(T, wrapper)

    # Handle both @hook and @hook() syntax
    if func is None:
        return decorator

    return decorator(func)
