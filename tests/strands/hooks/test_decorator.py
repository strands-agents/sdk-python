"""Tests for the @hook decorator."""

from unittest.mock import MagicMock

import pytest

from strands.hooks import (
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeNodeCallEvent,
    BeforeToolCallEvent,
    DecoratedFunctionHook,
    FunctionHookMetadata,
    HookMetadata,
    HookProvider,
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
        def my_hook(event: BeforeToolCallEvent | AfterToolCallEvent) -> None:
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

    def test_decorator_with_typing_union(self):
        """Test @hook with typing.Union type hint."""

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
        received_events = []

        @hook
        def my_hook(event: BeforeToolCallEvent) -> None:
            received_events.append(event)

        mock_event = MagicMock(spec=BeforeToolCallEvent)
        my_hook(mock_event)

        assert len(received_events) == 1
        assert received_events[0] is mock_event

    def test_hook_registration(self):
        """Test that hooks register correctly with HookRegistry."""
        callback_called = []

        @hook
        def my_hook(event: BeforeToolCallEvent) -> None:
            callback_called.append(event)

        registry = HookRegistry()
        my_hook.register_hooks(registry)

        mock_agent = MagicMock()
        event = BeforeToolCallEvent(
            agent=mock_agent,
            selected_tool=None,
            tool_use={"toolUseId": "test-123", "name": "test_tool", "input": {}},
            invocation_state={},
        )

        registry.invoke_callbacks(event)

        assert len(callback_called) == 1
        assert callback_called[0] is event

    def test_multi_event_registration(self):
        """Test that multi-event hooks register for all event types."""
        events_received = []

        @hook(events=[BeforeToolCallEvent, AfterToolCallEvent])
        def multi_hook(event: BeforeToolCallEvent | AfterToolCallEvent) -> None:
            events_received.append(type(event).__name__)

        registry = HookRegistry()
        multi_hook.register_hooks(registry)

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

    def test_hook_parentheses_no_args(self):
        """Test @hook() syntax with empty parentheses."""

        @hook()
        def my_hook(event: BeforeToolCallEvent) -> None:
            pass

        assert isinstance(my_hook, DecoratedFunctionHook)
        assert my_hook.event_types == [BeforeToolCallEvent]

    def test_optional_type_hint_extracts_event_type(self):
        """Test that Optional[EventType] correctly extracts the event type."""

        @hook
        def optional_hook(event: BeforeToolCallEvent | None) -> None:
            pass

        assert isinstance(optional_hook, DecoratedFunctionHook)
        assert optional_hook.event_types == [BeforeToolCallEvent]


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

    def test_invalid_annotation_not_event_type(self):
        """Test error when annotation is a non-event class type."""

        class NotAnEvent:
            pass

        with pytest.raises(ValueError, match="must be a subclass of BaseHookEvent"):

            @hook
            def invalid_hook(event: NotAnEvent) -> None:
                pass

    def test_invalid_single_event_type_in_explicit_list(self):
        """Test error when explicit event list contains invalid type."""
        with pytest.raises(ValueError, match="must be a subclass of BaseHookEvent"):

            @hook(events=[str])  # type: ignore
            def invalid_events_hook(event) -> None:
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

        metadata = FunctionHookMetadata(my_func, event_types=[AfterToolCallEvent])
        assert metadata.event_types == [AfterToolCallEvent]

    def test_event_types_property(self):
        """Test FunctionHookMetadata.event_types property."""

        def my_func(event: BeforeToolCallEvent) -> None:
            pass

        metadata = FunctionHookMetadata(my_func)
        assert metadata.event_types == [BeforeToolCallEvent]


class TestDecoratedFunctionHook:
    """Tests for DecoratedFunctionHook class."""

    def test_hook_provider_protocol(self):
        """Test that DecoratedFunctionHook implements HookProvider."""

        @hook
        def my_hook(event: BeforeToolCallEvent) -> None:
            pass

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


class TestAsyncHooks:
    """Tests for async hook support."""

    def test_async_hook_direct_invocation(self):
        """Test async hook direct invocation."""
        import asyncio

        received_events = []

        @hook
        async def async_hook(event: BeforeToolCallEvent) -> None:
            received_events.append(event)

        mock_event = MagicMock(spec=BeforeToolCallEvent)
        asyncio.run(async_hook(mock_event))

        assert len(received_events) == 1
        assert received_events[0] is mock_event

    def test_async_hook_via_registry(self):
        """Test async hook when invoked via registry."""
        import asyncio

        received_events = []

        @hook
        async def async_hook(event: BeforeToolCallEvent) -> None:
            received_events.append(event)

        registry = HookRegistry()
        async_hook.register_hooks(registry)

        mock_agent = MagicMock()
        event = BeforeToolCallEvent(
            agent=mock_agent,
            selected_tool=None,
            tool_use={"toolUseId": "test-123", "name": "test_tool", "input": {}},
            invocation_state={},
        )

        async def run_callbacks():
            for callback in registry.get_callbacks_for(event):
                result = callback(event)
                if asyncio.iscoroutine(result):
                    await result

        asyncio.run(run_callbacks())

        assert len(received_events) == 1


class TestMixedHooksUsage:
    """Tests for using decorated hooks alongside class-based hooks."""

    def test_mixed_hooks_in_registry(self):
        """Test using both decorator and class-based hooks together."""
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

        mock_agent = MagicMock()
        event = BeforeInvocationEvent(agent=mock_agent)

        registry.invoke_callbacks(event)

        assert len(decorator_called) == 1
        assert len(class_called) == 1


class TestDescriptorProtocol:
    """Tests for the __get__ descriptor protocol implementation."""

    def test_hook_as_class_method(self):
        """Test that @hook works correctly on class methods."""
        results = []

        class MyHooks:
            def __init__(self, prefix: str):
                self.prefix = prefix

            @hook
            def my_hook(self, event: BeforeToolCallEvent) -> None:
                results.append(f"{self.prefix}: {event}")

        hooks_instance = MyHooks("test")
        bound_hook = hooks_instance.my_hook

        assert isinstance(bound_hook, DecoratedFunctionHook)

        mock_agent = MagicMock()
        event = BeforeToolCallEvent(
            agent=mock_agent,
            selected_tool=None,
            tool_use={"toolUseId": "test-123", "name": "test_tool", "input": {}},
            invocation_state={},
        )

        bound_hook(event)

        assert len(results) == 1
        assert results[0].startswith("test:")

    def test_hook_class_method_via_registry(self):
        """Test that class method hooks work with HookRegistry."""
        results = []

        class MyHooks:
            def __init__(self, name: str):
                self.name = name

            @hook
            def on_tool_call(self, event: BeforeToolCallEvent) -> None:
                results.append({"name": self.name, "event": event})

        hooks_instance = MyHooks("registry_test")

        registry = HookRegistry()
        registry.add_hook(hooks_instance.on_tool_call)

        mock_agent = MagicMock()
        event = BeforeToolCallEvent(
            agent=mock_agent,
            selected_tool=None,
            tool_use={"toolUseId": "test-123", "name": "test_tool", "input": {}},
            invocation_state={},
        )

        registry.invoke_callbacks(event)

        assert len(results) == 1
        assert results[0]["name"] == "registry_test"
        assert results[0]["event"] is event

    def test_hook_accessed_via_class_returns_self(self):
        """Test that accessing hook via class (not instance) returns the hook itself."""

        class MyHooks:
            @hook
            def my_hook(self, event: BeforeToolCallEvent) -> None:
                pass

        class_hook = MyHooks.my_hook
        assert isinstance(class_hook, DecoratedFunctionHook)

    def test_hook_different_instances_are_independent(self):
        """Test that hooks bound to different instances are independent."""
        results = []

        class MyHooks:
            def __init__(self, name: str):
                self.name = name

            @hook
            def my_hook(self, event: BeforeToolCallEvent) -> None:
                results.append(self.name)

        hooks1 = MyHooks("first")
        hooks2 = MyHooks("second")

        mock_agent = MagicMock()
        event = BeforeToolCallEvent(
            agent=mock_agent,
            selected_tool=None,
            tool_use={"toolUseId": "test-123", "name": "test_tool", "input": {}},
            invocation_state={},
        )

        hooks1.my_hook(event)
        hooks2.my_hook(event)

        assert results == ["first", "second"]


class TestMultiagentEvents:
    """Tests for multiagent event support."""

    def test_multiagent_hook_works(self):
        """Test that hooks for multiagent events work correctly."""
        events_received = []

        @hook
        def node_hook(event: BeforeNodeCallEvent) -> None:
            events_received.append(event)

        assert node_hook.event_types == [BeforeNodeCallEvent]

        mock_source = MagicMock()
        event = BeforeNodeCallEvent(
            source=mock_source,
            node_id="test-node",
            invocation_state={},
        )

        node_hook(event)

        assert len(events_received) == 1
        assert events_received[0] is event


class TestEdgeCases:
    """Edge case tests for remaining coverage gaps."""

    def test_get_type_hints_exception_fallback(self):
        """Test fallback when get_type_hints raises an exception."""
        import unittest.mock as mock

        def func_with_annotation(event: BeforeToolCallEvent) -> None:
            pass

        with mock.patch("strands.hooks.decorator.get_type_hints", side_effect=Exception("Type hint error")):
            metadata = FunctionHookMetadata(func_with_annotation)
            assert metadata.event_types == [BeforeToolCallEvent]

    def test_annotation_fallback_when_type_hints_empty(self):
        """Test annotation is used when get_type_hints returns empty dict."""
        import unittest.mock as mock

        def func_with_annotation(event: BeforeToolCallEvent) -> None:
            pass

        with mock.patch("strands.hooks.decorator.get_type_hints", return_value={}):
            metadata = FunctionHookMetadata(func_with_annotation)
            assert metadata.event_types == [BeforeToolCallEvent]
