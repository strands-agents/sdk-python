"""Tests for AGUI state management tools."""

import json
import threading
import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from strands.agui.state_tools import (
    StrandsStateManager,
    emit_ui_update,
    get_agent_state,
    get_state_manager,
    set_agent_state,
    setup_agent_state_management,
    update_agent_state,
)
from strands.tools.tools import FunctionTool


class TestStrandsStateManager:
    """Test the StrandsStateManager class."""

    def test_initialization(self):
        """Test that StrandsStateManager initializes correctly."""
        manager = StrandsStateManager()

        assert manager._state == {}
        assert manager._callbacks == []
        assert manager._lock is not None

    def test_get_state_empty(self):
        """Test getting state when it's empty."""
        manager = StrandsStateManager()
        state = manager.get_state()

        assert state == {}

    def test_update_state_basic(self):
        """Test basic state update functionality."""
        manager = StrandsStateManager()
        updates = {"key1": "value1", "counter": 42}

        new_state = manager.update_state(updates)

        assert new_state == updates
        assert manager.get_state() == updates

    def test_update_state_multiple(self):
        """Test multiple state updates."""
        manager = StrandsStateManager()

        manager.update_state({"key1": "value1"})
        manager.update_state({"key2": "value2"})
        manager.update_state({"key1": "updated_value1"})

        final_state = manager.get_state()
        expected = {"key1": "updated_value1", "key2": "value2"}

        assert final_state == expected

    def test_set_state_replace(self):
        """Test that set_state completely replaces the state."""
        manager = StrandsStateManager()

        # Set initial state
        manager.update_state({"old_key": "old_value"})

        # Replace with new state
        new_state = {"new_key": "new_value"}
        result_state = manager.set_state(new_state)

        assert result_state == new_state
        assert manager.get_state() == new_state
        assert "old_key" not in manager.get_state()

    def test_state_isolation(self):
        """Test that returned state is isolated from internal state."""
        manager = StrandsStateManager()
        manager.update_state({"key": "value"})

        state1 = manager.get_state()
        state2 = manager.get_state()

        # Modify one copy
        state1["key"] = "modified"

        # Other copy should be unchanged
        assert state2["key"] == "value"
        assert manager.get_state()["key"] == "value"

    def test_callback_system(self):
        """Test state change callback system."""
        manager = StrandsStateManager()
        callback_calls = []

        def test_callback(new_state: Dict[str, Any], updates: Dict[str, Any]) -> None:
            callback_calls.append((new_state.copy(), updates.copy()))

        manager.add_callback(test_callback)
        manager.update_state({"key": "value"})

        assert len(callback_calls) == 1
        new_state, updates = callback_calls[0]
        assert new_state == {"key": "value"}
        assert updates == {"key": "value"}

    def test_multiple_callbacks(self):
        """Test that multiple callbacks are triggered."""
        manager = StrandsStateManager()
        calls1 = []
        calls2 = []

        def callback1(new_state, updates):
            calls1.append((new_state, updates))

        def callback2(new_state, updates):
            calls2.append((new_state, updates))

        manager.add_callback(callback1)
        manager.add_callback(callback2)
        manager.update_state({"test": "value"})

        assert len(calls1) == 1
        assert len(calls2) == 1

    def test_calculate_delta(self):
        """Test state delta calculation."""
        manager = StrandsStateManager()

        old_state = {"key1": "value1", "key2": "value2"}
        new_state = {"key1": "updated_value1", "key3": "value3"}

        delta = manager._calculate_delta(old_state, new_state)

        # Delta should include removed keys as None and new/updated keys
        expected = {"key1": "updated_value1", "key3": "value3", "key2": None}
        assert delta == expected

    def test_thread_safety(self):
        """Test that the state manager is thread-safe."""
        manager = StrandsStateManager()
        results = []
        errors = []

        def worker(worker_id: int) -> None:
            try:
                for i in range(10):
                    key = f"worker_{worker_id}_item_{i}"
                    manager.update_state({key: f"value_{i}"})
                    time.sleep(0.001)  # Small delay to increase chance of race conditions
                results.append(worker_id)
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")

        # Start multiple worker threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 3

        final_state = manager.get_state()
        # Should have 3 workers × 10 items each = 30 keys
        assert len(final_state) == 30


class TestGlobalStateManager:
    """Test the global state manager functions."""

    def test_get_state_manager_singleton(self):
        """Test that get_state_manager returns the same instance."""
        manager1 = get_state_manager()
        manager2 = get_state_manager()

        assert manager1 is manager2
        assert isinstance(manager1, StrandsStateManager)

    def test_state_persistence_across_calls(self):
        """Test that state persists across different calls."""
        manager1 = get_state_manager()
        manager1.update_state({"persistent": "value"})

        manager2 = get_state_manager()
        state = manager2.get_state()

        assert state["persistent"] == "value"


class TestStateTool:
    """Test the state management tool functions."""

    def test_get_agent_state_success(self):
        """Test successful state retrieval."""
        # Set up known state
        manager = get_state_manager()
        manager.set_state({"test_key": "test_value"})

        result = get_agent_state()

        assert result["status"] == "success"
        assert len(result["content"]) == 1
        assert "test_key" in result["content"][0]["text"]
        assert "test_value" in result["content"][0]["text"]

    def test_update_agent_state_success(self):
        """Test successful state update."""
        updates = {"new_key": "new_value", "counter": 100}

        result = update_agent_state(updates)

        assert result["status"] == "success"
        assert "Updated state with" in result["content"][0]["text"]

        # Verify state was actually updated
        manager = get_state_manager()
        state = manager.get_state()
        assert state["new_key"] == "new_value"
        assert state["counter"] == 100

    def test_set_agent_state_success(self):
        """Test successful state replacement."""
        # Set initial state
        get_state_manager().update_state({"old_key": "old_value"})

        new_state = {"replaced_key": "replaced_value"}
        result = set_agent_state(new_state)

        assert result["status"] == "success"
        assert "Set new state" in result["content"][0]["text"]

        # Verify state was replaced
        manager = get_state_manager()
        current_state = manager.get_state()
        assert current_state == new_state
        assert "old_key" not in current_state

    def test_emit_ui_update_success(self):
        """Test successful UI update emission."""
        component_name = "GameBoard"
        props = {"score": 100, "level": 5}

        result = emit_ui_update(component_name, props)

        assert result["status"] == "success"
        assert f"Emitted UI update for {component_name}" in result["content"][0]["text"]

        # Verify UI update was stored in state - checking actual implementation format
        manager = get_state_manager()
        state = manager.get_state()
        # Based on implementation, it stores as ui_{component_name} and last_ui_update
        assert f"ui_{component_name}" in state
        assert state[f"ui_{component_name}"] == props
        assert "last_ui_update" in state

    @patch("strands.agui.state_tools._state_manager")
    def test_update_agent_state_error_handling(self, mock_manager):
        """Test error handling in update_agent_state."""
        mock_manager.update_state.side_effect = Exception("Test error")

        result = update_agent_state({"test": "value"})

        assert result["status"] == "error"
        assert "Test error" in result["content"][0]["text"]

    @patch("strands.agui.state_tools._state_manager")
    def test_set_agent_state_error_handling(self, mock_manager):
        """Test error handling in set_agent_state."""
        mock_manager.set_state.side_effect = Exception("Set error")

        result = set_agent_state({"test": "value"})

        assert result["status"] == "error"
        assert "Set error" in result["content"][0]["text"]

    @patch("strands.agui.state_tools._state_manager")
    def test_emit_ui_update_error_handling(self, mock_manager):
        """Test error handling in emit_ui_update."""
        mock_manager.update_state.side_effect = Exception("UI error")

        result = emit_ui_update("TestComponent", {"prop": "value"})

        assert result["status"] == "error"
        assert "UI error" in result["content"][0]["text"]


class TestAgentSetup:
    """Test agent state management setup."""

    def test_setup_agent_state_management(self):
        """Test setting up state management for an agent."""
        # Mock agent with tool registry
        mock_tool_registry = MagicMock()
        mock_agent = MagicMock()
        mock_agent.tool_registry = mock_tool_registry

        initial_state = {"initial_key": "initial_value"}

        result_manager = setup_agent_state_management(mock_agent, initial_state)

        # Should return the state manager
        assert isinstance(result_manager, StrandsStateManager)

        # Should have registered 4 tools
        assert mock_tool_registry.register_tool.call_count == 4

        # Verify tools are FunctionTool instances
        for call in mock_tool_registry.register_tool.call_args_list:
            tool = call[0][0]
            assert isinstance(tool, FunctionTool)

        # Verify initial state was set
        current_state = result_manager.get_state()
        assert current_state["initial_key"] == "initial_value"

    def test_setup_agent_state_management_no_initial_state(self):
        """Test setup without initial state."""
        mock_tool_registry = MagicMock()
        mock_agent = MagicMock()
        mock_agent.tool_registry = mock_tool_registry

        result_manager = setup_agent_state_management(mock_agent)

        assert isinstance(result_manager, StrandsStateManager)
        assert mock_tool_registry.register_tool.call_count == 4

    def test_tool_names_registered(self):
        """Test that correct tool names are registered."""
        mock_tool_registry = MagicMock()
        mock_agent = MagicMock()
        mock_agent.tool_registry = mock_tool_registry

        setup_agent_state_management(mock_agent)

        # Extract tool names from registered tools
        registered_tools = []
        for call in mock_tool_registry.register_tool.call_args_list:
            tool = call[0][0]
            # Use _name attribute which is the actual attribute in FunctionTool
            registered_tools.append(tool._name)

        expected_tools = {"get_agent_state", "update_agent_state", "set_agent_state", "emit_ui_update"}

        assert set(registered_tools) == expected_tools


class TestIntegration:
    """Integration tests for the complete state management system."""

    def test_state_callback_integration(self):
        """Test integration between state updates and callbacks."""
        manager = get_state_manager()
        manager.set_state({})  # Clear state

        callback_events = []

        def test_callback(new_state, updates):
            callback_events.append({"new_state": new_state.copy(), "updates": updates.copy()})

        manager.add_callback(test_callback)

        # Perform operations using tools
        update_agent_state({"key1": "value1"})
        set_agent_state({"key2": "value2"})
        emit_ui_update("TestComponent", {"prop": "value"})

        # Should have triggered callbacks
        assert len(callback_events) >= 3

        # Final state should contain the UI update based on actual implementation
        final_state = manager.get_state()
        assert "ui_TestComponent" in final_state  # Actual format from implementation
        assert "TestComponent" in str(final_state)  # Component name should be in the state somehow

    def test_json_serialization(self):
        """Test that state can be JSON serialized."""
        manager = get_state_manager()
        manager.set_state(
            {
                "string": "value",
                "number": 42,
                "boolean": True,
                "null": None,
                "list": [1, 2, 3],
                "dict": {"nested": "value"},
            }
        )

        state = manager.get_state()
        json_str = json.dumps(state)

        # Should not raise exception
        parsed = json.loads(json_str)
        assert parsed == state
