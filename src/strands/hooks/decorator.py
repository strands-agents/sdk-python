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
import logging
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Optional,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from .registry import BaseHookEvent, HookCallback, HookEvent, HookProvider, HookRegistry

if TYPE_CHECKING:
    from ..agent import Agent

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
        has_agent_param: Whether the function has an 'agent' parameter for injection.
    """

    name: str
    description: str
    event_types: list[type[BaseHookEvent]]
    is_async: bool
    has_agent_param: bool = False


class FunctionHookMetadata:
    """Helper class to extract and manage function metadata for hook decoration.

    This class handles the extraction of metadata from Python functions including:
    - Function name and description from docstrings
    - Event types from type hints
    - Async detection
    - Agent parameter detection for automatic injection
    """

    def __init__(
        self,
        func: Callable[..., Any],
        event_types: Sequence[type[BaseHookEvent]] | None = None,
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

        # Check for agent parameter
        self._has_agent_param = self._check_agent_parameter()

    def _check_agent_parameter(self) -> bool:
        """Check if the function has an 'agent' parameter for injection.

        Returns:
            True if the function has an 'agent' parameter.
        """
        return "agent" in self.signature.parameters

    def _resolve_event_types(self) -> list[type[BaseHookEvent]]:
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
        # Skip 'self' and 'cls' for class methods
        params = list(self.signature.parameters.values())
        event_params = [p for p in params if p.name not in ("self", "cls")]

        if not event_params:
            raise ValueError(
                f"Hook function '{self.func.__name__}' must have at least one parameter "
                "for the event. Use @hook(event=EventType) if type hints are unavailable."
            )

        first_param = event_params[0]
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
        if isinstance(annotation, types.UnionType):
            return True

        return False

    def _extract_event_types_from_annotation(self, annotation: Any) -> list[type[BaseHookEvent]]:
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

    def _all_event_types_are_hook_events(self) -> bool:
        """Check if all event types extend HookEvent (which has .agent attribute).

        Returns:
            True if all event types are subclasses of HookEvent.
        """
        return all(issubclass(et, HookEvent) for et in self._event_types)

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
            has_agent_param=self._has_agent_param,
        )

    @property
    def event_types(self) -> list[type[BaseHookEvent]]:
        """Get the event types this hook handles."""
        return self._event_types

    @property
    def has_agent_param(self) -> bool:
        """Check if the function has an 'agent' parameter."""
        return self._has_agent_param


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

        Raises:
            ValueError: If agent injection is requested but event types don't support it.
        """
        self._func = func
        self._metadata = metadata
        self._hook_metadata = metadata.extract_metadata()

        # Validate agent injection compatibility
        if self._hook_metadata.has_agent_param and not metadata._all_event_types_are_hook_events():
            non_hook_events = [et.__name__ for et in metadata.event_types if not issubclass(et, HookEvent)]
            raise ValueError(
                f"Hook function '{func.__name__}' has an 'agent' parameter but handles event types "
                f"that don't have an 'agent' attribute: {non_hook_events}. "
                "Agent injection only works with events that extend HookEvent "
                "(e.g., BeforeToolCallEvent, AfterModelCallEvent). "
                "Multiagent events (e.g., BeforeNodeCallEvent, MultiAgentInitializedEvent) extend "
                "BaseHookEvent and have a 'source' attribute instead."
            )

        # Preserve function metadata
        functools.update_wrapper(wrapper=self, wrapped=self._func)

    def __get__(
        self, instance: Any, obj_type: type[Any] | None = None
    ) -> "DecoratedFunctionHook[TEvent]":
        """Descriptor protocol implementation for proper method binding.

        This method enables the decorated function to work correctly when used
        as a class method. It binds the instance to the function call when
        accessed through an instance.

        Args:
            instance: The instance through which the descriptor is accessed,
                or None when accessed through the class.
            obj_type: The class through which the descriptor is accessed.

        Returns:
            A new DecoratedFunctionHook with the instance bound to the function
            if accessed through an instance, otherwise returns self.

        Example:
            ```python
            class MyHooks:
                @hook
                def my_hook(self, event: BeforeToolCallEvent) -> None:
                    ...

            hooks = MyHooks()
            # Works correctly - 'self' is bound
            agent = Agent(hooks=[hooks.my_hook])
            ```
        """
        if instance is not None and not inspect.ismethod(self._func):
            # Create a bound method
            bound_func = self._func.__get__(instance, instance.__class__)
            return DecoratedFunctionHook(bound_func, self._metadata)

        return self

    def _create_callback_with_injection(self) -> HookCallback[BaseHookEvent]:
        """Create a callback that handles agent injection.

        Returns:
            A callback that wraps the original function with agent injection.
        """
        func = self._func
        has_agent_param = self._hook_metadata.has_agent_param

        if has_agent_param:
            # Create wrapper that injects agent
            # Safe to access event.agent here because we validated in __init__
            # that all event types are HookEvent subclasses
            if self._hook_metadata.is_async:

                async def async_callback_with_agent(event: BaseHookEvent) -> None:
                    # Cast is safe because we validated event types in __init__
                    hook_event = cast(HookEvent, event)
                    await func(event, agent=hook_event.agent)  # type: ignore[arg-type]

                return cast(HookCallback[BaseHookEvent], async_callback_with_agent)
            else:

                def sync_callback_with_agent(event: BaseHookEvent) -> None:
                    # Cast is safe because we validated event types in __init__
                    hook_event = cast(HookEvent, event)
                    func(event, agent=hook_event.agent)  # type: ignore[arg-type]

                return cast(HookCallback[BaseHookEvent], sync_callback_with_agent)
        else:
            # No injection needed, use function directly
            return cast(HookCallback[BaseHookEvent], func)

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register callback functions for specific event types.

        Args:
            registry: The hook registry to register callbacks with.
            **kwargs: Additional keyword arguments (unused, for protocol compatibility).
        """
        callback = self._create_callback_with_injection()
        for event_type in self._metadata.event_types:
            registry.add_callback(event_type, callback)

    def __call__(self, event: TEvent, agent: Optional["Agent"] = None) -> Any:
        """Allow direct invocation for testing.

        Args:
            event: The event to process.
            agent: Optional agent instance. If not provided and the hook
                   expects an agent parameter, it will be extracted from event.agent
                   (only works for HookEvent subclasses).

        Returns:
            The result of the hook function.
        """
        if self._hook_metadata.has_agent_param:
            # Use provided agent or fall back to event.agent
            # Safe because we validated in __init__ that event types support .agent
            if agent is not None:
                actual_agent = agent
            else:
                # Cast is safe because we validated event types in __init__
                hook_event = cast(HookEvent, event)
                actual_agent = hook_event.agent
            return self._func(event, agent=actual_agent)  # type: ignore[arg-type]
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
    def event_types(self) -> list[type[BaseHookEvent]]:
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

    @property
    def has_agent_param(self) -> bool:
        """Check if this hook has an agent parameter.

        Returns:
            True if the hook function expects an agent parameter.
        """
        return self._hook_metadata.has_agent_param

    def __repr__(self) -> str:
        """Return a string representation of the hook."""
        event_names = [e.__name__ for e in self._hook_metadata.event_types]
        agent_info = ", agent_injection=True" if self._hook_metadata.has_agent_param else ""
        return f"DecoratedFunctionHook({self._hook_metadata.name}, events={event_names}{agent_info})"


# Type variable for the decorated function
F = TypeVar("F", bound=Callable[..., Any])


# Handle @hook
@overload
def hook(__func: F) -> DecoratedFunctionHook[Any]: ...


# Handle @hook(event=...)
@overload
def hook(
    *,
    event: type[BaseHookEvent] | None = None,
    events: Sequence[type[BaseHookEvent]] | None = None,
) -> Callable[[F], DecoratedFunctionHook[Any]]: ...


def hook(
    func: F | None = None,
    event: type[BaseHookEvent] | None = None,
    events: Sequence[type[BaseHookEvent]] | None = None,
) -> DecoratedFunctionHook[Any] | Callable[[F], DecoratedFunctionHook[Any]]:
    """Decorator that transforms a function into a HookProvider.

    The decorated function can be passed directly to Agent(hooks=[...]).
    Event types are detected from type hints or can be specified explicitly.

    Args:
        func: The function to decorate.
        event: Single event type to handle.
        events: List of event types to handle.

    Returns:
        A DecoratedFunctionHook that implements HookProvider.

    Raises:
        ValueError: If no event type can be determined.
        ValueError: If event types are not subclasses of BaseHookEvent.
        ValueError: If agent injection is requested but event types don't support it.

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
        # Determine event types from parameters or type hints
        event_types: list[type[BaseHookEvent]] | None = None

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
