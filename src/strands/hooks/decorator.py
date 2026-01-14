"""Hook decorator for simplified hook definitions.

This module provides the @hook decorator that transforms Python functions into
HookProvider implementations with automatic event type detection from type hints.

The @hook decorator mirrors the ergonomics of the existing @tool decorator,
making hooks as easy to define and share via PyPI packages as tools are today.

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
import logging
import sys
import types
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from .registry import BaseHookEvent, HookProvider, HookRegistry

logger = logging.getLogger(__name__)


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
    event_types: list[Type[BaseHookEvent]]
    is_async: bool


class FunctionHookMetadata:
    """Helper class to extract and manage function metadata for hook decoration.

    This class handles the extraction of metadata from Python functions including:
    - Function name and description from docstrings
    - Event types from type hints
    - Async detection
    """

    def __init__(
        self,
        func: Callable[..., Any],
        event_types: Optional[Sequence[Type[BaseHookEvent]]] = None,
    ) -> None:
        """Initialize with the function to process.

        Args:
            func: The function to extract metadata from.
            event_types: Optional explicit event types. If not provided,
                        will be extracted from type hints.
        """
        self.func = func
        self.signature = inspect.signature(func)
        self._explicit_event_types = list(event_types) if event_types else None

        # Validate and extract event types
        self._event_types = self._resolve_event_types()
        self._validate_event_types()

    def _resolve_event_types(self) -> list[Type[BaseHookEvent]]:
        """Resolve event types from explicit parameter or type hints.

        Returns:
            List of event types this hook handles.

        Raises:
            ValueError: If no event type can be determined.
        """
        # Use explicit event types if provided
        if self._explicit_event_types:
            return self._explicit_event_types

        # Try to extract from type hints
        try:
            type_hints = get_type_hints(self.func)
        except Exception:
            # get_type_hints can fail for various reasons (forward refs, etc.)
            type_hints = {}

        # Find the first parameter's type hint (should be the event)
        params = list(self.signature.parameters.values())
        if not params:
            raise ValueError(
                f"Hook function '{self.func.__name__}' must have at least one parameter "
                "for the event. Use @hook(event=EventType) if type hints are unavailable."
            )

        first_param = params[0]
        event_type = type_hints.get(first_param.name)

        if event_type is None:
            # Check annotation directly (for cases where get_type_hints fails)
            if first_param.annotation is not inspect.Parameter.empty:
                event_type = first_param.annotation
            else:
                raise ValueError(
                    f"Hook function '{self.func.__name__}' must have a type hint for the event parameter, "
                    "or use @hook(event=EventType) to specify the event type explicitly."
                )

        # Handle Union types (e.g., BeforeToolCallEvent | AfterToolCallEvent)
        return self._extract_event_types_from_annotation(event_type)

    def _is_union_type(self, annotation: Any) -> bool:
        """Check if annotation is a Union type (typing.Union or types.UnionType).

        Args:
            annotation: The type annotation to check.

        Returns:
            True if the annotation is a Union type.
        """
        origin = get_origin(annotation)
        if origin is Union:
            return True

        # Python 3.10+ uses types.UnionType for `A | B` syntax
        if sys.version_info >= (3, 10):
            if isinstance(annotation, types.UnionType):
                return True

        return False

    def _extract_event_types_from_annotation(self, annotation: Any) -> list[Type[BaseHookEvent]]:
        """Extract event types from a type annotation.

        Handles Union types and single types.

        Args:
            annotation: The type annotation to extract from.

        Returns:
            List of event types.
        """
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
        """Validate that all event types are valid.

        Raises:
            ValueError: If any event type is invalid.
        """
        if not self._event_types:
            raise ValueError(f"Hook function '{self.func.__name__}' must handle at least one event type.")

        for event_type in self._event_types:
            if not isinstance(event_type, type) or not issubclass(event_type, BaseHookEvent):
                raise ValueError(f"Event type must be a subclass of BaseHookEvent, got {event_type}")

    def extract_metadata(self) -> HookMetadata:
        """Extract metadata from the function to create hook specification.

        Returns:
            HookMetadata containing the function's hook specification.
        """
        func_name = self.func.__name__

        # Extract description from docstring
        description = inspect.getdoc(self.func) or func_name

        # Check if async
        is_async = inspect.iscoroutinefunction(self.func)

        return HookMetadata(
            name=func_name,
            description=description,
            event_types=self._event_types,
            is_async=is_async,
        )

    @property
    def event_types(self) -> list[Type[BaseHookEvent]]:
        """Get the event types this hook handles."""
        return self._event_types


class DecoratedFunctionHook(HookProvider, Generic[TEvent]):
    """A HookProvider that wraps a function decorated with @hook.

    This class adapts Python functions decorated with @hook to the HookProvider
    interface, enabling them to be used with Agent's hooks parameter.

    The class is generic over the event type to maintain type safety.
    """

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

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register callback functions for specific event types.

        Args:
            registry: The hook registry to register callbacks with.
            **kwargs: Additional keyword arguments (unused, for protocol compatibility).
        """
        for event_type in self._metadata.event_types:
            registry.add_callback(event_type, self._func)

    def __call__(self, event: TEvent) -> Any:
        """Allow direct invocation for testing.

        Args:
            event: The event to process.

        Returns:
            The result of the hook function.
        """
        return self._func(event)

    @property
    def name(self) -> str:
        """Get the name of the hook.

        Returns:
            The hook name as a string.
        """
        return self._hook_metadata.name

    @property
    def description(self) -> str:
        """Get the description of the hook.

        Returns:
            The hook description as a string.
        """
        return self._hook_metadata.description

    @property
    def event_types(self) -> list[Type[BaseHookEvent]]:
        """Get the event types this hook handles.

        Returns:
            List of event types.
        """
        return self._hook_metadata.event_types

    @property
    def is_async(self) -> bool:
        """Check if this hook is async.

        Returns:
            True if the hook function is async.
        """
        return self._hook_metadata.is_async

    def __repr__(self) -> str:
        """Return a string representation of the hook."""
        event_names = [e.__name__ for e in self._hook_metadata.event_types]
        return f"DecoratedFunctionHook({self._hook_metadata.name}, events={event_names})"


# Type variable for the decorated function
F = TypeVar("F", bound=Callable[..., Any])


# Handle @hook
@overload
def hook(__func: F) -> DecoratedFunctionHook[Any]: ...


# Handle @hook(event=...)
@overload
def hook(
    event: Optional[Type[BaseHookEvent]] = None,
    events: Optional[Sequence[Type[BaseHookEvent]]] = None,
) -> Callable[[F], DecoratedFunctionHook[Any]]: ...


def hook(
    func: Optional[F] = None,
    event: Optional[Type[BaseHookEvent]] = None,
    events: Optional[Sequence[Type[BaseHookEvent]]] = None,
) -> Union[DecoratedFunctionHook[Any], Callable[[F], DecoratedFunctionHook[Any]]]:
    """Decorator that transforms a Python function into a Strands hook.

    This decorator enables simple, function-based hook definitions - mirroring
    the ergonomics of the existing @tool decorator. It extracts the event type
    from the function's type hints or from explicit parameters.

    When decorated, a function:
    1. Implements the HookProvider protocol automatically
    2. Can be passed directly to Agent(hooks=[...])
    3. Still works as a normal function when called directly
    4. Supports both sync and async hook functions

    The decorator can be used in several ways:

    1. Simple decorator with type hints:
        ```python
        @hook
        def my_hook(event: BeforeToolCallEvent) -> None:
            print(f"Tool: {event.tool_use}")
        ```

    2. With explicit event type:
        ```python
        @hook(event=BeforeToolCallEvent)
        def my_hook(event) -> None:
            print(f"Tool: {event.tool_use}")
        ```

    3. For multiple event types:
        ```python
        @hook(events=[BeforeToolCallEvent, AfterToolCallEvent])
        def my_hook(event: BeforeToolCallEvent | AfterToolCallEvent) -> None:
            print(f"Event: {event}")
        ```

    4. With Union type hint:
        ```python
        @hook
        def my_hook(event: BeforeToolCallEvent | AfterToolCallEvent) -> None:
            print(f"Event: {event}")
        ```

    Args:
        func: The function to decorate. When used as a simple decorator,
            this is the function being decorated. When used with parameters,
            this will be None.
        event: Optional single event type to handle. Takes precedence over
            type hint detection.
        events: Optional list of event types to handle. Takes precedence over
            both `event` parameter and type hint detection.

    Returns:
        A DecoratedFunctionHook that implements HookProvider and can be used
        directly with Agent(hooks=[...]).

    Raises:
        ValueError: If no event type can be determined from type hints or parameters.
        ValueError: If event types are not subclasses of BaseHookEvent.

    Example:
        ```python
        from strands import Agent, hook
        from strands.hooks import BeforeToolCallEvent, AfterToolCallEvent

        @hook
        def log_tool_calls(event: BeforeToolCallEvent) -> None:
            '''Log all tool calls before execution.'''
            print(f"Calling tool: {event.tool_use['name']}")

        @hook
        async def async_audit(event: AfterToolCallEvent) -> None:
            '''Async hook for auditing tool results.'''
            await send_to_audit_service(event.result)

        @hook(events=[BeforeToolCallEvent, AfterToolCallEvent])
        def tool_lifecycle(event: BeforeToolCallEvent | AfterToolCallEvent) -> None:
            '''Track the complete tool lifecycle.'''
            if isinstance(event, BeforeToolCallEvent):
                print("Starting tool...")
            else:
                print("Tool complete!")

        agent = Agent(hooks=[log_tool_calls, async_audit, tool_lifecycle])
        ```
    """

    def decorator(f: F) -> DecoratedFunctionHook[Any]:
        # Determine event types from parameters or type hints
        event_types: Optional[list[Type[BaseHookEvent]]] = None

        if events is not None:
            event_types = list(events)
        elif event is not None:
            event_types = [event]
        # Otherwise, let FunctionHookMetadata extract from type hints

        # Create function hook metadata
        hook_meta = FunctionHookMetadata(f, event_types)

        return DecoratedFunctionHook(f, hook_meta)

    # Handle both @hook and @hook() syntax
    if func is None:
        return decorator

    return decorator(func)
