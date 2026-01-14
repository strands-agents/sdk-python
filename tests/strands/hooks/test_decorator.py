"""Tests for the @hook decorator."""

from typing import Union
from unittest.mock import MagicMock

import pytest

from strands.hooks import (
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeToolCallEvent,
    DecoratedFunctionHook,
    FunctionHookMetadata,
    HookMetadata,
    HookRegistry,
    hook,
)


class TestHookDecorator:
    """Tests for the @hook decorator function."""

    def test_basic_decorator_with_type_hint(self):
        """Test @hook with type hints extracts event type correctly."""

        @hook
        def my_hook(event: BeforeToolCallEvent) -> None:
            pass

        assert isinstance(my_hook, DecoratedFunctionHook)
        assert my_hook.name == "my_hook"
        assert my_hook.event_types == [BeforeToolCallEvent]
        assert not my_hook.is_async

    def test_decorator_with_explicit_event(self):
        """Test @hook(event=...) syntax."""

        @hook(event=BeforeToolCallEvent)
        def my_hook(event) -> None:
            pass

        assert isinstance(my_hook, DecoratedFunctionHook)
        assert my_hook.event_types == [BeforeToolCallEvent]

    def test_decorator_with_multiple_events(self):
        """Test @hook(events=[...]) syntax for multiple event types."""

        @hook(events=[BeforeToolCallEvent, AfterToolCallEvent])
        def my_hook(event: Union[BeforeToolCallEvent, AfterToolCallEvent]) -> None:
            pass

        assert isinstance(my_hook, DecoratedFunctionHook)
        assert set(my_hook.event_types) == {BeforeToolCallEvent, AfterToolCallEvent}

    def test_decorator_with_union_type_hint(self):
        """Test @hook with Union type hint extracts multiple event types."""

        @hook
        def my_hook(event: BeforeToolCallEvent | AfterToolCallEvent) -> None:
            pass

        assert isinstance(my_hook, DecoratedFunctionHook)
        assert set(my_hook.event_types) == {BeforeToolCallEvent, AfterToolCallEvent}

    def test_async_hook_detection(self):
        """Test that async hooks are detected correctly."""

        @hook
        async def async_hook(event: BeforeToolCallEvent) -> None:
            pass

        assert async_hook.is_async

        @hook
        def sync_hook(event: BeforeToolCallEvent) -> None:
            pass

        assert not sync_hook.is_async

    def test_docstring_extraction(self):
        """Test that docstring is extracted as description."""

        @hook
        def documented_hook(event: BeforeToolCallEvent) -> None:
            """This is a documented hook for testing."""
            pass

        assert documented_hook.description == "This is a documented hook for testing."

    def test_default_description(self):
        """Test that function name is used when no docstring."""

        @hook
        def undocumented_hook(event: BeforeToolCallEvent) -> None:
            pass

        assert undocumented_hook.description == "undocumented_hook"

    def test_direct_invocation(self):
        """Test that decorated hooks can be called directly."""
        mock_callback = MagicMock()

        @hook
        def my_hook(event: BeforeToolCallEvent) -> None:
            mock_callback(event)

        # Create a mock event
        mock_event = MagicMock(spec=BeforeToolCallEvent)

        # Direct invocation
        my_hook(mock_event)

        mock_callback.assert_called_once_with(mock_event)

    def test_hook_registration(self):
        """Test that hooks register correctly with HookRegistry."""
        callback_called = []

        @hook
        def my_hook(event: BeforeToolCallEvent) -> None:
            callback_called.append(event)

        registry = HookRegistry()
        my_hook.register_hooks(registry)

        # Verify callback is registered
        mock_agent = MagicMock()
        mock_tool_use = {"toolUseId": "test-123", "name": "test_tool", "input": {}}
        event = BeforeToolCallEvent(
            agent=mock_agent,
            selected_tool=None,
            tool_use=mock_tool_use,
            invocation_state={},
        )

        registry.invoke_callbacks(event)

        assert len(callback_called) == 1
        assert callback_called[0] is event

    def test_multi_event_registration(self):
        """Test that multi-event hooks register for all event types."""
        events_received = []

        @hook(events=[BeforeToolCallEvent, AfterToolCallEvent])
        def multi_hook(event: Union[BeforeToolCallEvent, AfterToolCallEvent]) -> None:
            events_received.append(type(event).__name__)

        registry = HookRegistry()
        multi_hook.register_hooks(registry)

        # Create mock events
        mock_agent = MagicMock()
        mock_tool_use = {"toolUseId": "test-123", "name": "test_tool", "input": {}}
        mock_result = {"toolUseId": "test-123", "status": "success", "content": []}

        before_event = BeforeToolCallEvent(
            agent=mock_agent,
            selected_tool=None,
            tool_use=mock_tool_use,
            invocation_state={},
        )
        after_event = AfterToolCallEvent(
            agent=mock_agent,
            selected_tool=None,
            tool_use=mock_tool_use,
            invocation_state={},
            result=mock_result,
        )

        registry.invoke_callbacks(before_event)
        registry.invoke_callbacks(after_event)

        assert "BeforeToolCallEvent" in events_received
        assert "AfterToolCallEvent" in events_received

    def test_repr(self):
        """Test string representation of decorated hook."""

        @hook
        def my_hook(event: BeforeToolCallEvent) -> None:
            pass

        repr_str = repr(my_hook)
        assert "DecoratedFunctionHook" in repr_str
        assert "my_hook" in repr_str
        assert "BeforeToolCallEvent" in repr_str


class TestHookDecoratorErrors:
    """Tests for error handling in @hook decorator."""

    def test_no_parameters_error(self):
        """Test error when function has no parameters."""
        with pytest.raises(ValueError, match="must have at least one parameter"):

            @hook
            def no_params() -> None:
                pass

    def test_no_type_hint_error(self):
        """Test error when no type hint and no explicit event type."""
        with pytest.raises(ValueError, match="must have a type hint"):

            @hook
            def no_hint(event) -> None:
                pass

    def test_invalid_event_type_error(self):
        """Test error when event type is not a BaseHookEvent subclass."""
        with pytest.raises(ValueError, match="must be a subclass of BaseHookEvent"):

            @hook(event=str)  # type: ignore
            def invalid_event(event) -> None:
                pass

    def test_invalid_union_type_error(self):
        """Test error when Union contains non-event types."""
        with pytest.raises(ValueError, match="must be subclasses of BaseHookEvent"):

            @hook
            def invalid_union(event: BeforeToolCallEvent | str) -> None:  # type: ignore
                pass


class TestFunctionHookMetadata:
    """Tests for FunctionHookMetadata class."""

    def test_metadata_extraction(self):
        """Test metadata extraction from function."""

        def my_func(event: BeforeToolCallEvent) -> None:
            """A test hook function."""
            pass

        metadata = FunctionHookMetadata(my_func)
        hook_meta = metadata.extract_metadata()

        assert isinstance(hook_meta, HookMetadata)
        assert hook_meta.name == "my_func"
        assert hook_meta.description == "A test hook function."
        assert hook_meta.event_types == [BeforeToolCallEvent]
        assert not hook_meta.is_async

    def test_explicit_event_types_override(self):
        """Test that explicit event types override type hints."""

        def my_func(event: BeforeToolCallEvent) -> None:
            pass

        # Explicitly specify different event type
        metadata = FunctionHookMetadata(my_func, event_types=[AfterToolCallEvent])

        assert metadata.event_types == [AfterToolCallEvent]


class TestDecoratedFunctionHook:
    """Tests for DecoratedFunctionHook class."""

    def test_hook_provider_protocol(self):
        """Test that DecoratedFunctionHook implements HookProvider."""

        @hook
        def my_hook(event: BeforeToolCallEvent) -> None:
            pass

        # Should have register_hooks method
        assert hasattr(my_hook, "register_hooks")
        assert callable(my_hook.register_hooks)

    def test_function_wrapper_preserves_metadata(self):
        """Test that functools.wraps preserves function metadata."""

        @hook
        def original_function(event: BeforeToolCallEvent) -> None:
            """Original docstring."""
            pass

        assert original_function.__name__ == "original_function"
        assert original_function.__doc__ == "Original docstring."


class TestMixedHooksUsage:
    """Tests for using decorated hooks alongside class-based hooks."""

    def test_mixed_hooks_in_registry(self):
        """Test using both decorator and class-based hooks together."""
        from strands.hooks import HookProvider, HookRegistry

        decorator_called = []
        class_called = []

        @hook
        def decorator_hook(event: BeforeInvocationEvent) -> None:
            decorator_called.append(event)

        class ClassHook(HookProvider):
            def register_hooks(self, registry: HookRegistry) -> None:
                registry.add_callback(BeforeInvocationEvent, self.on_event)

            def on_event(self, event: BeforeInvocationEvent) -> None:
                class_called.append(event)

        registry = HookRegistry()
        registry.add_hook(decorator_hook)
        registry.add_hook(ClassHook())

        # Create mock event
        mock_agent = MagicMock()
        event = BeforeInvocationEvent(agent=mock_agent)

        registry.invoke_callbacks(event)

        assert len(decorator_called) == 1
        assert len(class_called) == 1
