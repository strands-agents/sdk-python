"""Hook decorator for defining hooks as functions.

This module provides the @hook decorator that transforms Python functions into
HookProvider implementations with automatic event type detection from type hints.

Example:
    ```python
    from strands import Agent, hook
    from strands.hooks import BeforeToolCallEvent

    @hook
    def log_tool_calls(event: BeforeToolCallEvent) -> None:
        '''Log all tool calls before execution.'''
        print(f"Tool: {event.tool_use}")

    agent = Agent(hooks=[log_tool_calls])
    ```
"""

import functools
import inspect
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from .registry import BaseHookEvent, HookCallback, HookProvider, HookRegistry

TEvent = TypeVar("TEvent", bound=BaseHookEvent)


@dataclass
class HookMetadata:
    """Metadata extracted from a decorated hook function.

    Attributes:
        name: The name of the hook function.
        description: Description extracted from the function's docstring.
        event_types: List of event types this hook handles.
        is_async: Whether the hook function is async.
    """

    name: str
    description: str
    event_types: list[type[BaseHookEvent]]
    is_async: bool


class FunctionHookMetadata:
    """Helper class to extract and manage function metadata for hook decoration."""

    def __init__(
        self,
        func: Callable[..., Any],
    ) -> None:
        """Initialize with the function to process.

        Args:
            func: The function to extract metadata from.
        """
        self.func = func
        self.signature = inspect.signature(func)

        # Validate and extract event types
        self._event_types = self._resolve_event_types()
        self._validate_event_types()

    def _resolve_event_types(self) -> list[type[BaseHookEvent]]:
        """Resolve event types from type hints.

        Returns:
            List of event types this hook handles.

        Raises:
            ValueError: If no event type can be determined.
        """
        # Try to extract from type hints
        try:
            type_hints = get_type_hints(self.func)
        except Exception:
            # get_type_hints can fail for various reasons (forward refs, etc.)
            type_hints = {}

        # Find the first parameter's type hint (should be the event)
        # Skip 'self' and 'cls' for class methods
        params = list(self.signature.parameters.values())
        event_params = [p for p in params if p.name not in ("self", "cls")]

        if not event_params:
            raise ValueError(
                f"Hook function '{self.func.__name__}' must have at least one parameter "
                "for the event with a type hint."
            )

        first_param = event_params[0]
        event_type = type_hints.get(first_param.name)

        if event_type is None:
            # Check annotation directly (for cases where get_type_hints fails)
            if first_param.annotation is not inspect.Parameter.empty:
                event_type = first_param.annotation
            else:
                raise ValueError(
                    f"Hook function '{self.func.__name__}' must have a type hint for the event parameter."
                )

        # Handle Union types (e.g., BeforeToolCallEvent | AfterToolCallEvent)
        return self._extract_event_types_from_annotation(event_type)

    def _is_union_type(self, annotation: Any) -> bool:
        """Check if annotation is a Union type (typing.Union or types.UnionType)."""
        origin = get_origin(annotation)
        if origin is Union:
            return True

        # Python 3.10+ uses types.UnionType for `A | B` syntax
        if isinstance(annotation, types.UnionType):
            return True

        return False

    def _extract_event_types_from_annotation(self, annotation: Any) -> list[type[BaseHookEvent]]:
        """Extract event types from a type annotation."""
        # Handle Union types (Union[A, B] or A | B)
        if self._is_union_type(annotation):
            args = get_args(annotation)
            event_types = []
            for arg in args:
                # Skip NoneType in Optional[X]
                if arg is type(None):
                    continue
                if isinstance(arg, type) and issubclass(arg, BaseHookEvent):
                    event_types.append(arg)
                else:
                    raise ValueError(f"All types in Union must be subclasses of BaseHookEvent, got {arg}")
            return event_types

        # Single type
        if isinstance(annotation, type) and issubclass(annotation, BaseHookEvent):
            return [annotation]

        raise ValueError(f"Event type must be a subclass of BaseHookEvent, got {annotation}")

    def _validate_event_types(self) -> None:
        """Validate that all event types are valid."""
        if not self._event_types:
            raise ValueError(f"Hook function '{self.func.__name__}' must handle at least one event type.")

        for event_type in self._event_types:
            if not isinstance(event_type, type) or not issubclass(event_type, BaseHookEvent):
                raise ValueError(f"Event type must be a subclass of BaseHookEvent, got {event_type}")

    def extract_metadata(self) -> HookMetadata:
        """Extract metadata from the function to create hook specification."""
        return HookMetadata(
            name=self.func.__name__,
            description=inspect.getdoc(self.func) or self.func.__name__,
            event_types=self._event_types,
            is_async=inspect.iscoroutinefunction(self.func),
        )

    @property
    def event_types(self) -> list[type[BaseHookEvent]]:
        """Get the event types this hook handles."""
        return self._event_types


class DecoratedFunctionHook(HookProvider, Generic[TEvent]):
    """A HookProvider that wraps a function decorated with @hook."""

    _func: Callable[[TEvent], Any]
    _metadata: FunctionHookMetadata
    _hook_metadata: HookMetadata

    def __init__(
        self,
        func: Callable[[TEvent], Any],
        metadata: FunctionHookMetadata,
    ):
        """Initialize the decorated function hook.

        Args:
            func: The original function being decorated.
            metadata: The FunctionHookMetadata object with extracted function information.
        """
        self._func = func
        self._metadata = metadata
        self._hook_metadata = metadata.extract_metadata()

        # Preserve function metadata
        functools.update_wrapper(wrapper=self, wrapped=self._func)

    def __get__(self, instance: Any, obj_type: type[Any] | None = None) -> "DecoratedFunctionHook[TEvent]":
        """Descriptor protocol implementation for proper method binding."""
        if instance is not None and not inspect.ismethod(self._func):
            # Create a bound method
            bound_func = self._func.__get__(instance, instance.__class__)
            return DecoratedFunctionHook(bound_func, self._metadata)

        return self

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register callback functions for specific event types."""
        callback = cast(HookCallback[BaseHookEvent], self._func)
        for event_type in self._metadata.event_types:
            registry.add_callback(event_type, callback)

    def __call__(self, event: TEvent) -> Any:
        """Allow direct invocation for testing."""
        return self._func(event)

    @property
    def name(self) -> str:
        """Get the name of the hook."""
        return self._hook_metadata.name

    @property
    def description(self) -> str:
        """Get the description of the hook."""
        return self._hook_metadata.description

    @property
    def event_types(self) -> list[type[BaseHookEvent]]:
        """Get the event types this hook handles."""
        return self._hook_metadata.event_types

    @property
    def is_async(self) -> bool:
        """Check if this hook is async."""
        return self._hook_metadata.is_async

    def __repr__(self) -> str:
        """Return a string representation of the hook."""
        event_names = [e.__name__ for e in self._hook_metadata.event_types]
        return f"DecoratedFunctionHook({self._hook_metadata.name}, events={event_names})"


# Type variable for the decorated function
F = TypeVar("F", bound=Callable[..., Any])


def hook(
    func: F | None = None,
) -> DecoratedFunctionHook[Any] | Callable[[F], DecoratedFunctionHook[Any]]:
    """Decorator that transforms a function into a HookProvider.

    The decorated function can be passed directly to Agent(hooks=[...]).
    Event types are automatically detected from the function's type hints.

    Args:
        func: The function to decorate.

    Returns:
        A DecoratedFunctionHook that implements HookProvider.

    Raises:
        ValueError: If no event type can be determined from type hints.
        ValueError: If event types are not subclasses of BaseHookEvent.

    Example:
        ```python
        from strands import Agent, hook
        from strands.hooks import BeforeToolCallEvent

        @hook
        def log_tool_calls(event: BeforeToolCallEvent) -> None:
            print(f"Tool: {event.tool_use}")

        agent = Agent(hooks=[log_tool_calls])
        ```
    """

    def decorator(f: F) -> DecoratedFunctionHook[Any]:
        hook_meta = FunctionHookMetadata(f)
        return DecoratedFunctionHook(f, hook_meta)

    if func is None:
        return decorator

    return decorator(func)
