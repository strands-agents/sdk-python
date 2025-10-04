"""Tests for delegation hook events.

This module tests delegation-specific hook events and their integration
with the hook registry system.
"""

from unittest.mock import Mock

import pytest

from strands.hooks.events import SubAgentAddedEvent, SubAgentRemovedEvent
from strands.hooks.registry import HookRegistry
from strands.types._events import (
    DelegationCompleteEvent,
    DelegationProxyEvent,
    DelegationStartEvent,
    DelegationTimeoutEvent,
)


@pytest.mark.delegation
class TestDelegationHookEvents:
    """Test delegation event structures and hook integration."""

    @pytest.mark.parametrize(
        "event_class,event_data,expected_key,expected_fields",
        [
            (
                DelegationStartEvent,
                {"from_agent": "Orch", "to_agent": "Sub", "message": "Test"},
                "delegation_start",
                ["from_agent", "to_agent", "message"],
            ),
            (
                DelegationCompleteEvent,
                {"target_agent": "Sub", "result": Mock()},
                "delegation_complete",
                ["target_agent", "result"],
            ),
            (
                DelegationProxyEvent,
                {"original_event": Mock(), "from_agent": "A", "to_agent": "B"},
                "delegation_proxy",
                ["from_agent", "to_agent", "original_event"],
            ),
            (
                DelegationTimeoutEvent,
                {"target_agent": "Slow", "timeout_seconds": 30.0},
                "delegation_timeout",
                ["target_agent", "timeout_seconds"],
            ),
            (
                SubAgentAddedEvent,
                {"agent": Mock(), "sub_agent": Mock(), "sub_agent_name": "New"},
                None,  # SubAgentAddedEvent is a dataclass, not a TypedEvent
                ["agent", "sub_agent", "sub_agent_name"],
            ),
            (
                SubAgentRemovedEvent,
                {"agent": Mock(), "sub_agent_name": "Old", "removed_agent": Mock()},
                None,  # SubAgentRemovedEvent is a dataclass, not a TypedEvent
                ["agent", "sub_agent_name", "removed_agent"],
            ),
        ],
    )
    def test_delegation_event_structure(self, event_class, event_data, expected_key, expected_fields):
        """Test all delegation event structures with parametrization."""
        event = event_class(**event_data)

        # TypedEvent classes (dict-based)
        if expected_key:
            assert expected_key in event
            # Verify all expected fields present in the nested dict
            for field in expected_fields:
                assert field in event[expected_key]
        # Dataclass events (SubAgentAdded/Removed)
        else:
            # Verify all expected fields present as attributes
            for field in expected_fields:
                assert hasattr(event, field)

    def test_hook_registry_integration(self):
        """Test delegation events can be used with hook registry."""
        # HookRegistry expects HookEvent (dataclass), not TypedEvent (dict)
        # Test that SubAgentAdded/Removed events work with registry
        registry = HookRegistry()
        events_captured = []

        def capture(event):
            events_captured.append(event)

        registry.add_callback(SubAgentAddedEvent, capture)

        orchestrator = Mock()
        sub_agent = Mock()
        event = SubAgentAddedEvent(agent=orchestrator, sub_agent=sub_agent, sub_agent_name="NewAgent")
        registry.invoke_callbacks(event)

        assert len(events_captured) == 1
        captured = events_captured[0]
        assert captured.agent == orchestrator
        assert captured.sub_agent == sub_agent
        assert captured.sub_agent_name == "NewAgent"

    def test_delegation_start_event_properties(self):
        """Test DelegationStartEvent property accessors."""
        event = DelegationStartEvent(
            from_agent="Orchestrator", to_agent="SpecialistAgent", message="Handle complex task"
        )

        assert event.from_agent == "Orchestrator"
        assert event.to_agent == "SpecialistAgent"
        assert event.message == "Handle complex task"

    def test_delegation_complete_event_properties(self):
        """Test DelegationCompleteEvent property accessors."""
        mock_result = Mock()
        event = DelegationCompleteEvent(target_agent="SpecialistAgent", result=mock_result)

        assert event.target_agent == "SpecialistAgent"
        assert event.result == mock_result

    def test_delegation_proxy_event_properties(self):
        """Test DelegationProxyEvent property accessors."""
        original_event = Mock()
        event = DelegationProxyEvent(original_event=original_event, from_agent="Orchestrator", to_agent="SubAgent")

        assert event.original_event == original_event
        assert event.from_agent == "Orchestrator"
        assert event.to_agent == "SubAgent"

    def test_delegation_timeout_event_properties(self):
        """Test DelegationTimeoutEvent property accessors."""
        event = DelegationTimeoutEvent(target_agent="SlowAgent", timeout_seconds=300.0)

        assert event.target_agent == "SlowAgent"
        assert event.timeout_seconds == 300.0

    def test_sub_agent_added_event_attributes(self):
        """Test SubAgentAddedEvent dataclass attributes."""
        orchestrator = Mock()
        sub_agent = Mock()

        event = SubAgentAddedEvent(agent=orchestrator, sub_agent=sub_agent, sub_agent_name="NewAgent")

        assert event.agent == orchestrator
        assert event.sub_agent == sub_agent
        assert event.sub_agent_name == "NewAgent"

    def test_sub_agent_removed_event_attributes(self):
        """Test SubAgentRemovedEvent dataclass attributes."""
        orchestrator = Mock()
        removed_agent = Mock()

        event = SubAgentRemovedEvent(agent=orchestrator, sub_agent_name="OldAgent", removed_agent=removed_agent)

        assert event.agent == orchestrator
        assert event.sub_agent_name == "OldAgent"
        assert event.removed_agent == removed_agent

    def test_multiple_callbacks_for_delegation_events(self):
        """Test multiple callbacks can be registered for SubAgent events."""
        registry = HookRegistry()
        callback1_calls = []
        callback2_calls = []

        def callback1(event):
            callback1_calls.append(event)

        def callback2(event):
            callback2_calls.append(event)

        registry.add_callback(SubAgentAddedEvent, callback1)
        registry.add_callback(SubAgentAddedEvent, callback2)

        event = SubAgentAddedEvent(agent=Mock(), sub_agent=Mock(), sub_agent_name="TestAgent")
        registry.invoke_callbacks(event)

        assert len(callback1_calls) == 1
        assert len(callback2_calls) == 1

    def test_delegation_event_serialization(self):
        """Test delegation events can be serialized for logging."""
        event = DelegationStartEvent(from_agent="Orchestrator", to_agent="SubAgent", message="Test message")

        # TypedEvent is a dict subclass, should be JSON-serializable
        import json

        serialized = json.dumps(dict(event))
        deserialized = json.loads(serialized)

        assert deserialized["delegation_start"]["from_agent"] == "Orchestrator"
        assert deserialized["delegation_start"]["to_agent"] == "SubAgent"
        assert deserialized["delegation_start"]["message"] == "Test message"
