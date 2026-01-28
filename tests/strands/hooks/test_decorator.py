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
    HookRegistry,
    MultiAgentInitializedEvent,
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
        def multi_hook(event: BeforeToolCallEvent | AfterToolCallEvent) -> None:
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


class TestAgentInjection:
    """Tests for automatic agent injection in @hook decorated functions."""

    def test_agent_param_detection(self):
        """Test that agent parameter is correctly detected."""
        from strands.agent import Agent

        @hook
        def with_agent(event: BeforeToolCallEvent, agent: Agent) -> None:
            pass

        @hook
        def without_agent(event: BeforeToolCallEvent) -> None:
            pass

        assert with_agent.has_agent_param is True
        assert without_agent.has_agent_param is False

    def test_agent_injection_in_repr(self):
        """Test that agent injection is shown in repr."""
        from strands.agent import Agent

        @hook
        def with_agent(event: BeforeToolCallEvent, agent: Agent) -> None:
            pass

        assert "agent_injection=True" in repr(with_agent)

    def test_hook_without_agent_param_not_injected(self):
        """Test that hooks without agent param work normally."""
        received_events = []

        @hook
        def simple_hook(event: BeforeToolCallEvent) -> None:
            received_events.append(event)

        # Create a mock event
        mock_agent = MagicMock()
        mock_event = MagicMock(spec=BeforeToolCallEvent)
        mock_event.agent = mock_agent

        # Call directly
        simple_hook(mock_event)

        assert len(received_events) == 1
        assert received_events[0] is mock_event

    def test_hook_with_agent_param_receives_agent(self):
        """Test that hooks with agent param receive agent via injection."""
        received_data = []

        @hook
        def hook_with_agent(event: BeforeToolCallEvent, agent) -> None:
            received_data.append({"event": event, "agent": agent})

        # Create mock event with agent
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_event = MagicMock(spec=BeforeToolCallEvent)
        mock_event.agent = mock_agent

        # Call directly - agent should be extracted from event.agent
        hook_with_agent(mock_event)

        assert len(received_data) == 1
        assert received_data[0]["event"] is mock_event
        assert received_data[0]["agent"] is mock_agent

    def test_direct_call_with_explicit_agent(self):
        """Test direct invocation with explicit agent parameter."""
        received_data = []

        @hook
        def hook_with_agent(event: BeforeToolCallEvent, agent) -> None:
            received_data.append({"event": event, "agent": agent})

        # Create mocks
        mock_event = MagicMock(spec=BeforeToolCallEvent)
        mock_event.agent = MagicMock(name="event_agent")
        explicit_agent = MagicMock(name="explicit_agent")

        # Call with explicit agent - should use explicit over event.agent
        hook_with_agent(mock_event, agent=explicit_agent)

        assert len(received_data) == 1
        assert received_data[0]["agent"] is explicit_agent

    def test_agent_injection_with_registry(self):
        """Test agent injection when registered with HookRegistry."""
        received_data = []

        @hook
        def hook_with_agent(event: BeforeToolCallEvent, agent) -> None:
            received_data.append({"event": event, "agent": agent})

        # Create registry and register hook
        registry = HookRegistry()
        hook_with_agent.register_hooks(registry)

        # Create a real BeforeToolCallEvent (not mock) since registry uses type()
        mock_agent = MagicMock()
        mock_agent.name = "registry_test_agent"

        # Create actual event instance
        mock_tool = MagicMock()
        event = BeforeToolCallEvent(
            agent=mock_agent,
            selected_tool=mock_tool,
            tool_use={"toolUseId": "test-123", "name": "test_tool", "input": {}},
            invocation_state={},
        )

        # Invoke callbacks through registry
        for callback in registry.get_callbacks_for(event):
            callback(event)

        assert len(received_data) == 1
        assert received_data[0]["agent"] is mock_agent

    def test_async_hook_with_agent_injection(self):
        """Test async hooks with agent injection."""
        import asyncio

        received_data = []

        @hook
        async def async_hook_with_agent(event: BeforeToolCallEvent, agent) -> None:
            received_data.append({"event": event, "agent": agent})

        assert async_hook_with_agent.has_agent_param is True
        assert async_hook_with_agent.is_async is True

        # Create mock event
        mock_agent = MagicMock()
        mock_event = MagicMock(spec=BeforeToolCallEvent)
        mock_event.agent = mock_agent

        # Run async hook
        asyncio.run(async_hook_with_agent(mock_event))

        assert len(received_data) == 1
        assert received_data[0]["agent"] is mock_agent

    def test_hook_metadata_includes_agent_param(self):
        """Test that HookMetadata correctly reflects agent parameter."""

        @hook
        def with_agent(event: BeforeToolCallEvent, agent) -> None:
            pass

        # Access internal metadata
        metadata = with_agent._hook_metadata

        assert metadata.has_agent_param is True
        assert metadata.name == "with_agent"

    def test_mixed_hooks_with_and_without_agent(self):
        """Test that hooks with and without agent params work together."""
        results = {"with_agent": [], "without_agent": []}

        @hook
        def without_agent_hook(event: BeforeToolCallEvent) -> None:
            results["without_agent"].append(event)

        @hook
        def with_agent_hook(event: BeforeToolCallEvent, agent) -> None:
            results["with_agent"].append({"event": event, "agent": agent})

        # Create mock event
        mock_agent = MagicMock()
        mock_event = MagicMock(spec=BeforeToolCallEvent)
        mock_event.agent = mock_agent

        # Call both hooks
        without_agent_hook(mock_event)
        with_agent_hook(mock_event)

        assert len(results["without_agent"]) == 1
        assert len(results["with_agent"]) == 1
        assert results["with_agent"][0]["agent"] is mock_agent


class TestAgentInjectionWithMultiagentEvents:
    """Tests for agent injection error handling with multiagent events."""

    def test_agent_injection_fails_with_multiagent_events(self):
        """Test that agent injection raises error for events without .agent attribute."""
        with pytest.raises(ValueError, match="don't have an 'agent' attribute"):

            @hook
            def bad_hook(event: BeforeNodeCallEvent, agent) -> None:
                pass

    def test_agent_injection_fails_with_multiagent_initialized_event(self):
        """Test that agent injection raises error for MultiAgentInitializedEvent."""
        with pytest.raises(ValueError, match="don't have an 'agent' attribute"):

            @hook
            def bad_hook(event: MultiAgentInitializedEvent, agent) -> None:
                pass

    def test_agent_injection_fails_with_mixed_events(self):
        """Test that agent injection raises error when mixing HookEvent and BaseHookEvent."""
        with pytest.raises(ValueError, match="don't have an 'agent' attribute"):

            @hook(events=[BeforeToolCallEvent, BeforeNodeCallEvent])
            def bad_hook(event, agent) -> None:
                pass

    def test_multiagent_hook_without_agent_param_works(self):
        """Test that multiagent hooks without agent param work correctly."""
        events_received = []

        @hook
        def node_hook(event: BeforeNodeCallEvent) -> None:
            events_received.append(event)

        assert node_hook.has_agent_param is False
        assert node_hook.event_types == [BeforeNodeCallEvent]

        # Create a mock multiagent event
        mock_source = MagicMock()
        event = BeforeNodeCallEvent(
            source=mock_source,
            node_id="test-node",
            invocation_state={},
        )

        # Direct invocation should work
        node_hook(event)

        assert len(events_received) == 1
        assert events_received[0] is event

    def test_error_message_lists_problematic_events(self):
        """Test that error message includes the event types that don't support injection."""
        with pytest.raises(ValueError) as exc_info:

            @hook(events=[BeforeToolCallEvent, BeforeNodeCallEvent, MultiAgentInitializedEvent])
            def bad_hook(event, agent) -> None:
                pass

        error_msg = str(exc_info.value)
        assert "BeforeNodeCallEvent" in error_msg
        assert "MultiAgentInitializedEvent" in error_msg
        # BeforeToolCallEvent supports agent injection, so it should NOT be in the error
        assert "BeforeToolCallEvent" not in error_msg or "extend HookEvent" in error_msg


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

        # Access the hook through the instance - should bind 'self'
        bound_hook = hooks_instance.my_hook

        # Should be a DecoratedFunctionHook
        assert isinstance(bound_hook, DecoratedFunctionHook)

        # Create a mock event and call
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

        # Register the bound method with registry
        registry = HookRegistry()
        registry.add_hook(hooks_instance.on_tool_call)

        # Create event and invoke
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

    def test_hook_class_method_with_agent_injection(self):
        """Test that class method hooks with agent injection work correctly."""
        results = []

        class MyHooks:
            @hook
            def with_agent(self, event: BeforeToolCallEvent, agent) -> None:
                results.append({"self": self, "event": event, "agent": agent})

        hooks_instance = MyHooks()
        bound_hook = hooks_instance.with_agent

        # Create mock event
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        event = BeforeToolCallEvent(
            agent=mock_agent,
            selected_tool=None,
            tool_use={"toolUseId": "test-123", "name": "test_tool", "input": {}},
            invocation_state={},
        )

        bound_hook(event)

        assert len(results) == 1
        assert results[0]["self"] is hooks_instance
        assert results[0]["agent"] is mock_agent

    def test_hook_accessed_via_class_returns_self(self):
        """Test that accessing hook via class (not instance) returns the hook itself."""

        class MyHooks:
            @hook
            def my_hook(self, event: BeforeToolCallEvent) -> None:
                pass

        # Access through class - should return the descriptor itself
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

        # Create event
        mock_agent = MagicMock()
        event = BeforeToolCallEvent(
            agent=mock_agent,
            selected_tool=None,
            tool_use={"toolUseId": "test-123", "name": "test_tool", "input": {}},
            invocation_state={},
        )

        # Call hooks from different instances
        hooks1.my_hook(event)
        hooks2.my_hook(event)

        assert results == ["first", "second"]


class TestCoverageGaps:
    """Additional tests to cover edge cases and improve coverage."""

    def test_optional_type_hint_extracts_event_type(self):
        """Test that Optional[EventType] correctly extracts the event type (skips NoneType)."""

        @hook
        def optional_hook(event: BeforeToolCallEvent | None) -> None:
            pass

        assert isinstance(optional_hook, DecoratedFunctionHook)
        assert optional_hook.event_types == [BeforeToolCallEvent]

    def test_async_hook_with_agent_via_registry(self):
        """Test async hook with agent injection when invoked via registry."""
        import asyncio

        received_data = []

        @hook
        async def async_hook_with_agent(event: BeforeToolCallEvent, agent) -> None:
            received_data.append({"event": event, "agent": agent})

        # Register with registry
        registry = HookRegistry()
        async_hook_with_agent.register_hooks(registry)

        # Create event
        mock_agent = MagicMock()
        mock_agent.name = "async_registry_agent"
        event = BeforeToolCallEvent(
            agent=mock_agent,
            selected_tool=None,
            tool_use={"toolUseId": "test-123", "name": "test_tool", "input": {}},
            invocation_state={},
        )

        # Get callbacks and invoke them (async)
        async def run_callbacks():
            for callback in registry.get_callbacks_for(event):
                result = callback(event)
                if asyncio.iscoroutine(result):
                    await result

        asyncio.run(run_callbacks())

        assert len(received_data) == 1
        assert received_data[0]["agent"] is mock_agent

    def test_sync_hook_with_agent_via_registry(self):
        """Test sync hook with agent injection when invoked via registry."""
        received_data = []

        @hook
        def sync_hook_with_agent(event: BeforeToolCallEvent, agent) -> None:
            received_data.append({"event": event, "agent": agent})

        # Register with registry
        registry = HookRegistry()
        sync_hook_with_agent.register_hooks(registry)

        # Create event
        mock_agent = MagicMock()
        mock_agent.name = "sync_registry_agent"
        event = BeforeToolCallEvent(
            agent=mock_agent,
            selected_tool=None,
            tool_use={"toolUseId": "test-123", "name": "test_tool", "input": {}},
            invocation_state={},
        )

        # Get callbacks and invoke them
        for callback in registry.get_callbacks_for(event):
            callback(event)

        assert len(received_data) == 1
        assert received_data[0]["agent"] is mock_agent

    def test_direct_call_without_agent_param_ignores_explicit_agent(self):
        """Test that hooks without agent param work even if explicit agent is passed."""
        received_events = []

        @hook
        def no_agent_hook(event: BeforeToolCallEvent) -> None:
            received_events.append(event)

        # Create mock event
        mock_event = MagicMock(spec=BeforeToolCallEvent)
        explicit_agent = MagicMock(name="explicit_agent")

        # Call with explicit agent - should be ignored since hook doesn't take agent
        no_agent_hook(mock_event, agent=explicit_agent)

        assert len(received_events) == 1
        assert received_events[0] is mock_event

    def test_get_type_hints_failure_fallback(self):
        """Test that annotation is used when get_type_hints fails."""
        # Create a function with a forward reference that might cause get_type_hints to fail
        # by directly testing FunctionHookMetadata with annotation

        def func_with_annotation(event: BeforeToolCallEvent) -> None:
            pass

        # This should work normally
        metadata = FunctionHookMetadata(func_with_annotation)
        assert metadata.event_types == [BeforeToolCallEvent]

    def test_hook_parentheses_no_args(self):
        """Test @hook() syntax with empty parentheses."""

        @hook()
        def my_hook(event: BeforeToolCallEvent) -> None:
            pass

        assert isinstance(my_hook, DecoratedFunctionHook)
        assert my_hook.event_types == [BeforeToolCallEvent]

    def test_union_with_typing_union(self):
        """Test Union from typing module explicitly."""

        @hook
        def union_hook(event: BeforeToolCallEvent | AfterToolCallEvent) -> None:
            pass

        assert isinstance(union_hook, DecoratedFunctionHook)
        assert set(union_hook.event_types) == {BeforeToolCallEvent, AfterToolCallEvent}

    def test_function_hook_metadata_event_types_property(self):
        """Test FunctionHookMetadata.event_types property."""

        def my_func(event: BeforeToolCallEvent) -> None:
            pass

        metadata = FunctionHookMetadata(my_func)
        # Access via property
        assert metadata.event_types == [BeforeToolCallEvent]

    def test_function_hook_metadata_has_agent_param_property(self):
        """Test FunctionHookMetadata.has_agent_param property."""

        def with_agent(event: BeforeToolCallEvent, agent) -> None:
            pass

        def without_agent(event: BeforeToolCallEvent) -> None:
            pass

        meta_with = FunctionHookMetadata(with_agent)
        meta_without = FunctionHookMetadata(without_agent)

        # Access via property
        assert meta_with.has_agent_param is True
        assert meta_without.has_agent_param is False


class TestAdditionalErrorCases:
    """Additional error case tests for complete coverage."""

    def test_invalid_annotation_not_event_type(self):
        """Test error when annotation is a non-event class type."""
        # This should trigger the error at line 216: "Event type must be a subclass of BaseHookEvent"

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


class TestEdgeCases:
    """Edge case tests for remaining coverage gaps."""

    def test_get_type_hints_exception_fallback(self):
        """Test fallback when get_type_hints raises an exception.

        This can happen with certain forward references or complex type annotations.
        """
        # Create a function with annotation that get_type_hints might struggle with
        # but that still has a valid annotation

        def func_with_annotation(event: BeforeToolCallEvent) -> None:
            pass

        # Manually test by mocking get_type_hints to raise
        import unittest.mock as mock

        with mock.patch("strands.hooks.decorator.get_type_hints", side_effect=Exception("Type hint error")):
            metadata = FunctionHookMetadata(func_with_annotation)
            # Should fall back to annotation
            assert metadata.event_types == [BeforeToolCallEvent]

    def test_annotation_fallback_when_type_hints_empty(self):
        """Test annotation is used when get_type_hints returns empty dict for param."""
        import unittest.mock as mock

        def func_with_annotation(event: BeforeToolCallEvent) -> None:
            pass

        # Mock get_type_hints to return empty dict (param not in hints)
        with mock.patch("strands.hooks.decorator.get_type_hints", return_value={}):
            metadata = FunctionHookMetadata(func_with_annotation)
            # Should fall back to first_param.annotation
            assert metadata.event_types == [BeforeToolCallEvent]

    def test_all_event_types_are_hook_events_helper(self):
        """Test the _all_event_types_are_hook_events helper method."""

        def hook_event_func(event: BeforeToolCallEvent) -> None:
            pass

        def base_event_func(event: BeforeNodeCallEvent) -> None:
            pass

        meta_hook = FunctionHookMetadata(hook_event_func)
        meta_base = FunctionHookMetadata(base_event_func)

        assert meta_hook._all_event_types_are_hook_events() is True
        assert meta_base._all_event_types_are_hook_events() is False
