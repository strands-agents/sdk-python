"""Tests for accepting callable hook callbacks in Agent constructor's hooks parameter."""

from unittest.mock import MagicMock

import pytest

from strands import Agent
from strands.hooks import (
    AfterInvocationEvent,
    AgentInitializedEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    HookProvider,
    HookRegistry,
)
from tests.fixtures.mocked_model_provider import MockedModelProvider


class TestHooksParamAcceptsCallables:
    """Test that the Agent constructor's hooks parameter accepts both HookProviders and callables."""

    def test_hooks_param_accepts_callable(self):
        """Verify that a plain callable can be passed via hooks parameter."""
        events_received = []

        def my_callback(event: AgentInitializedEvent) -> None:
            events_received.append(event)

        agent = Agent(hooks=[my_callback], callback_handler=None)

        assert len(events_received) == 1
        assert isinstance(events_received[0], AgentInitializedEvent)
        assert events_received[0].agent is agent

    def test_hooks_param_accepts_hook_provider(self):
        """Verify that HookProvider still works as before (backward compatibility)."""

        class MyProvider(HookProvider):
            def __init__(self):
                self.events = []

            def register_hooks(self, registry: HookRegistry) -> None:
                registry.add_callback(AgentInitializedEvent, self.on_init)

            def on_init(self, event: AgentInitializedEvent) -> None:
                self.events.append(event)

        provider = MyProvider()
        agent = Agent(hooks=[provider], callback_handler=None)

        assert len(provider.events) == 1
        assert isinstance(provider.events[0], AgentInitializedEvent)

    def test_hooks_param_accepts_mixed_list(self):
        """Verify that a mix of HookProviders and callables can be passed."""
        callback_events = []
        provider_events = []

        def my_callback(event: AgentInitializedEvent) -> None:
            callback_events.append(event)

        class MyProvider(HookProvider):
            def register_hooks(self, registry: HookRegistry) -> None:
                registry.add_callback(AgentInitializedEvent, lambda e: provider_events.append(e))

        agent = Agent(hooks=[MyProvider(), my_callback], callback_handler=None)

        assert len(callback_events) == 1
        assert len(provider_events) == 1
        assert callback_events[0].agent is agent
        assert provider_events[0].agent is agent

    def test_hooks_param_callable_invoked_during_agent_lifecycle(self):
        """Verify that callable hooks registered via hooks param fire during agent lifecycle."""
        before_events = []
        after_events = []

        def on_before(event: BeforeInvocationEvent) -> None:
            before_events.append(event)

        def on_after(event: AfterInvocationEvent) -> None:
            after_events.append(event)

        mock_model = MockedModelProvider(
            [{"role": "assistant", "content": [{"text": "Hello!"}]}]
        )

        agent = Agent(
            model=mock_model,
            hooks=[on_before, on_after],
            callback_handler=None,
        )
        agent("test prompt")

        assert len(before_events) == 1
        assert len(after_events) == 1
        assert isinstance(before_events[0], BeforeInvocationEvent)
        assert isinstance(after_events[0], AfterInvocationEvent)

    def test_hooks_param_invalid_hook_raises_error(self):
        """Verify that passing an invalid hook (not HookProvider or callable) raises ValueError."""
        with pytest.raises(ValueError, match="Invalid hook"):
            Agent(hooks=["not_a_hook"], callback_handler=None)  # type: ignore

    def test_hooks_param_none_is_valid(self):
        """Verify that passing None for hooks is still valid."""
        agent = Agent(hooks=None, callback_handler=None)
        assert agent is not None

    def test_hooks_param_empty_list_is_valid(self):
        """Verify that passing an empty list for hooks is still valid."""
        agent = Agent(hooks=[], callback_handler=None)
        assert agent is not None

    def test_hooks_param_callable_with_explicit_type_hint(self):
        """Verify that callables with typed event parameters work via hooks param."""
        model_call_events = []

        def on_model_call(event: BeforeModelCallEvent) -> None:
            model_call_events.append(event)

        mock_model = MockedModelProvider(
            [{"role": "assistant", "content": [{"text": "result"}]}]
        )

        agent = Agent(
            model=mock_model,
            hooks=[on_model_call],
            callback_handler=None,
        )
        agent("prompt")

        assert len(model_call_events) >= 1
        assert isinstance(model_call_events[0], BeforeModelCallEvent)

    def test_hooks_param_lambda_without_type_hint_raises_error(self):
        """Verify that lambda functions without type hints raise ValueError."""
        with pytest.raises(ValueError, match="cannot infer event type"):
            Agent(
                hooks=[lambda event: None],  # type: ignore
                callback_handler=None,
            )

    def test_hooks_param_multiple_callables(self):
        """Verify that multiple callables can be registered."""
        events_a = []
        events_b = []

        def callback_a(event: AgentInitializedEvent) -> None:
            events_a.append(event)

        def callback_b(event: AgentInitializedEvent) -> None:
            events_b.append(event)

        agent = Agent(hooks=[callback_a, callback_b], callback_handler=None)

        assert len(events_a) == 1
        assert len(events_b) == 1


class TestHooksParamAsyncCallables:
    """Test that the Agent constructor's hooks parameter accepts async callables."""

    def test_hooks_param_accepts_async_before_invocation_callback(self):
        """Verify that async callable hooks can be registered for non-init events."""
        events_received = []

        async def my_async_callback(event: BeforeInvocationEvent) -> None:
            events_received.append(event)

        mock_model = MockedModelProvider(
            [{"role": "assistant", "content": [{"text": "Hello!"}]}]
        )

        agent = Agent(
            model=mock_model,
            hooks=[my_async_callback],
            callback_handler=None,
        )
        agent("test")

        assert len(events_received) == 1
        assert isinstance(events_received[0], BeforeInvocationEvent)

    def test_hooks_param_rejects_async_agent_initialized_callback(self):
        """Verify that async callbacks for AgentInitializedEvent raise ValueError."""

        async def my_async_callback(event: AgentInitializedEvent) -> None:
            pass

        with pytest.raises(ValueError, match="AgentInitializedEvent can only be registered with a synchronous callback"):
            Agent(hooks=[my_async_callback], callback_handler=None)
