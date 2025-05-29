"""Tests for AGUI bridge functionality."""

import json
from unittest.mock import MagicMock, Mock

import pytest

from strands.agui.bridge import (
    AGUIEventType,
    StrandsAGUIBridge,
    StrandsAGUIEndpoint,
    create_strands_agui_setup,
)
from strands.agui.state_tools import StrandsStateManager


class TestAGUIEventType:
    """Test AGUI event type enumeration."""

    def test_event_types_exist(self):
        """Test that all expected event types are defined."""
        expected_events = {
            "RUN_STARTED",
            "RUN_FINISHED",
            "RUN_ERROR",
            "TEXT_MESSAGE_START",
            "TEXT_MESSAGE_CONTENT",
            "TEXT_MESSAGE_END",
            "TOOL_CALL_START",
            "TOOL_CALL_ARGS",
            "TOOL_CALL_END",
            "STATE_SNAPSHOT",
            "STATE_DELTA",
            "STEP_STARTED",
            "STEP_FINISHED",
            "CUSTOM",
            "RAW",
        }

        for event_name in expected_events:
            assert hasattr(AGUIEventType, event_name)

    def test_event_type_values(self):
        """Test that event types have correct string values."""
        assert AGUIEventType.RUN_STARTED == "RUN_STARTED"
        assert AGUIEventType.STATE_SNAPSHOT == "STATE_SNAPSHOT"
        assert AGUIEventType.STATE_DELTA == "STATE_DELTA"
        assert AGUIEventType.TOOL_CALL_START == "TOOL_CALL_START"

    def test_enum_inheritance(self):
        """Test that AGUIEventType properly inherits from str and Enum."""
        event = AGUIEventType.RUN_STARTED
        assert isinstance(event, str)
        assert event.value == "RUN_STARTED"


class TestStrandsAGUIBridge:
    """Test the StrandsAGUIBridge class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_agent = MagicMock()
        self.mock_state_manager = Mock(spec=StrandsStateManager)
        self.mock_state_manager.get_state.return_value = {}
        self.bridge = StrandsAGUIBridge(self.mock_agent, self.mock_state_manager)

    def test_initialization(self):
        """Test bridge initialization."""
        assert self.bridge.agent is self.mock_agent
        assert self.bridge.state_manager is self.mock_state_manager
        assert self.bridge.current_run_id is None
        assert self.bridge.current_thread_id is None
        assert self.bridge.current_message_id is None
        assert self.bridge.active_tool_calls == {}
        assert self.bridge.message_started is False

    def test_initialization_default_state_manager(self):
        """Test initialization without explicit state manager."""
        bridge = StrandsAGUIBridge(self.mock_agent)

        assert bridge.agent is self.mock_agent
        assert isinstance(bridge.state_manager, StrandsStateManager)

    def test_convert_strands_event_to_agui_text_content(self):
        """Test converting text content events."""
        strands_event = {"data": "Hello world"}

        agui_events = self.bridge._convert_strands_event_to_agui(strands_event)

        assert len(agui_events) == 2
        # First should be message start
        assert agui_events[0]["type"] == AGUIEventType.TEXT_MESSAGE_START
        # Second should be content
        assert agui_events[1]["type"] == AGUIEventType.TEXT_MESSAGE_CONTENT
        assert agui_events[1]["delta"] == "Hello world"

    def test_convert_strands_event_to_agui_tool_call(self):
        """Test converting tool call events."""
        strands_event = {
            "current_tool_use": {"toolUseId": "call_123", "name": "test_tool", "input": {"param": "value"}}
        }

        agui_events = self.bridge._convert_strands_event_to_agui(strands_event)

        # Should have tool call start and args events
        tool_events = [
            e for e in agui_events if e["type"] in [AGUIEventType.TOOL_CALL_START, AGUIEventType.TOOL_CALL_ARGS]
        ]
        assert len(tool_events) >= 2

    def test_convert_strands_event_to_agui_completion(self):
        """Test converting completion events."""
        # First add a tool call to track
        self.bridge.active_tool_calls["call_123"] = {"name": "test_tool"}

        strands_event = {"complete": True}
        agui_events = self.bridge._convert_strands_event_to_agui(strands_event)

        # Should have tool call end event
        end_events = [e for e in agui_events if e["type"] == AGUIEventType.TOOL_CALL_END]
        assert len(end_events) == 1

    def test_on_state_change(self):
        """Test state change callback handling."""
        new_state = {"key": "new_value"}
        updates = {"key": "new_value"}

        self.bridge._on_state_change(new_state, updates)

        # Should queue state change event
        assert len(self.bridge._state_change_queue) == 1
        event = self.bridge._state_change_queue[0]
        assert event["type"] == AGUIEventType.STATE_DELTA

    def test_dict_to_json_patch(self):
        """Test dictionary to JSON patch conversion."""
        updates = {"new_key": "new_value", "updated_key": "updated_value", "removed_key": None}

        patches = self.bridge._dict_to_json_patch(updates)

        assert len(patches) == 3

        # Check patch operations
        ops = {patch["op"] for patch in patches}
        assert "replace" in ops
        assert "remove" in ops

    def test_create_agui_event(self):
        """Test AGUI event creation."""
        event_type = AGUIEventType.RUN_STARTED
        data = {"thread_id": "test_thread"}

        event = self.bridge._create_agui_event(event_type, data)

        assert event["type"] == event_type.value
        assert event["thread_id"] == "test_thread"
        assert "timestamp" in event
        assert isinstance(event["timestamp"], int)

    def test_convert_strands_event_empty(self):
        """Test converting empty strands event."""
        strands_event = {}

        agui_events = self.bridge._convert_strands_event_to_agui(strands_event)

        # Should return empty list for empty event
        assert agui_events == []

    def test_message_state_tracking(self):
        """Test that message state is tracked correctly."""
        # Initially no message started
        assert self.bridge.message_started is False

        # Process text content event
        strands_event = {"data": "Hello"}
        self.bridge._convert_strands_event_to_agui(strands_event)

        # Should have started message
        assert self.bridge.message_started is True

        # Second text event should not start new message
        strands_event2 = {"data": " world"}
        agui_events2 = self.bridge._convert_strands_event_to_agui(strands_event2)

        # Should only have content event, not start event
        start_events = [e for e in agui_events2 if e["type"] == AGUIEventType.TEXT_MESSAGE_START]
        assert len(start_events) == 0


class TestStrandsAGUIEndpoint:
    """Test the StrandsAGUIEndpoint class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_agent = MagicMock()
        self.agents = {"test_agent": self.mock_agent}
        self.endpoint = StrandsAGUIEndpoint(self.agents)

    def test_initialization(self):
        """Test endpoint initialization."""
        assert self.endpoint.agents == self.agents
        assert "test_agent" in self.endpoint.bridges
        assert isinstance(self.endpoint.bridges["test_agent"], StrandsAGUIBridge)

    @pytest.mark.asyncio
    async def test_handle_request_agent_not_found(self):
        """Test handling request for non-existent agent."""
        request_data = {"agent": "nonexistent_agent", "messages": [{"role": "user", "content": "Hello"}]}

        responses = []
        async for response in self.endpoint.handle_request(request_data):
            responses.append(response)

        assert len(responses) == 1
        response_data = json.loads(responses[0][6:-2])  # Parse SSE data
        assert response_data["type"] == "RUN_ERROR"
        assert "Agent not found" in response_data["message"]

    @pytest.mark.asyncio
    async def test_handle_request_no_user_message(self):
        """Test handling request without user message."""
        request_data = {"agent": "test_agent", "messages": []}

        responses = []
        async for response in self.endpoint.handle_request(request_data):
            responses.append(response)

        assert len(responses) == 1
        response_data = json.loads(responses[0][6:-2])
        assert response_data["type"] == "RUN_ERROR"
        assert "No user message found" in response_data["message"]

    def test_message_content_extraction(self):
        """Test extraction of message content from various formats."""
        # Test simple string content
        messages = [{"role": "user", "content": "Simple message"}]
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        content = user_messages[-1].get("content", "")
        assert content == "Simple message"

        # Test array content
        messages2 = [{"role": "user", "content": [{"text": "Array message"}]}]
        user_messages2 = [msg for msg in messages2 if msg.get("role") == "user"]
        content2 = user_messages2[-1].get("content", "")
        if isinstance(content2, list) and content2:
            content2 = content2[0].get("text", "")
        assert content2 == "Array message"


class TestCreateStrandsAGUISetup:
    """Test the create_strands_agui_setup function."""

    def test_create_setup_basic(self):
        """Test basic setup creation."""
        mock_agent = MagicMock()
        mock_agent.tool_registry = MagicMock()
        agents = {"test_agent": mock_agent}

        endpoint = create_strands_agui_setup(agents)

        assert isinstance(endpoint, StrandsAGUIEndpoint)
        assert endpoint.agents == agents
        assert "test_agent" in endpoint.bridges

    def test_create_setup_with_initial_states(self):
        """Test setup with initial states."""
        mock_agent = MagicMock()
        mock_agent.tool_registry = MagicMock()
        agents = {"test_agent": mock_agent}
        initial_states = {"test_agent": {"initial": "value"}}

        endpoint = create_strands_agui_setup(agents, initial_states)

        assert isinstance(endpoint, StrandsAGUIEndpoint)
        # Should have called setup_agent_state_management for each agent
        mock_agent.tool_registry.register_tool.assert_called()

    def test_create_setup_multiple_agents(self):
        """Test setup with multiple agents."""
        mock_agent1 = MagicMock()
        mock_agent1.tool_registry = MagicMock()
        mock_agent2 = MagicMock()
        mock_agent2.tool_registry = MagicMock()

        agents = {"agent1": mock_agent1, "agent2": mock_agent2}

        endpoint = create_strands_agui_setup(agents)

        assert len(endpoint.bridges) == 2
        assert "agent1" in endpoint.bridges
        assert "agent2" in endpoint.bridges


class TestIntegration:
    """Integration tests for the complete AGUI bridge system."""

    def test_state_synchronization(self):
        """Test state synchronization between bridge and state manager."""
        mock_agent = MagicMock()
        bridge = StrandsAGUIBridge(mock_agent)

        # Verify callback was registered
        assert len(bridge.state_manager._callbacks) > 0

        # Trigger state change
        bridge.state_manager.update_state({"test": "value"})

        # Should have queued state event
        assert len(bridge._state_change_queue) > 0

        # Event should be STATE_DELTA type
        event = bridge._state_change_queue[0]
        assert event["type"] == AGUIEventType.STATE_DELTA

    def test_event_timestamp_consistency(self):
        """Test that events have consistent timestamp format."""
        mock_agent = MagicMock()
        bridge = StrandsAGUIBridge(mock_agent)

        # Create multiple events
        events = [
            bridge._create_agui_event(AGUIEventType.RUN_STARTED, {}),
            bridge._create_agui_event(AGUIEventType.STATE_SNAPSHOT, {"snapshot": {}}),
            bridge._create_agui_event(AGUIEventType.RUN_FINISHED, {}),
        ]

        # All should have timestamps
        for event in events:
            assert "timestamp" in event
            assert isinstance(event["timestamp"], int)
            # Should be reasonable timestamp (Unix milliseconds)
            assert event["timestamp"] > 1000000000000

    def test_json_patch_format(self):
        """Test JSON patch format consistency."""
        mock_agent = MagicMock()
        bridge = StrandsAGUIBridge(mock_agent)

        updates = {"key1": "value1", "key2": None, "key3": "value3"}
        patches = bridge._dict_to_json_patch(updates)

        # Should have proper JSON patch structure
        for patch in patches:
            assert "op" in patch
            assert "path" in patch
            assert patch["op"] in ["replace", "remove"]
            assert patch["path"].startswith("/")

    def test_bridge_endpoint_integration(self):
        """Test integration between bridge and endpoint."""
        mock_agent = MagicMock()
        mock_agent.tool_registry = MagicMock()
        agents = {"test_agent": mock_agent}

        endpoint = create_strands_agui_setup(agents)
        bridge = endpoint.bridges["test_agent"]

        # Bridge should have same agent
        assert bridge.agent is mock_agent

        # Bridge should be in endpoint
        assert endpoint.bridges["test_agent"] is bridge
