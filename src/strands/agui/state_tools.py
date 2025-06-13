"""Strands State Management Tools for AG-UI Compatibility.

This module provides state management capabilities to Strands agents
through a tool-based approach, making them compatible with AG-UI frontends
that expect state synchronization.
"""

import json
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

# Use relative imports to avoid module name conflicts
from ..tools.decorator import tool
from ..types.tools import ToolResult, ToolUse

T = TypeVar("T")


class StrandsStateManager:
    """Thread-safe state manager for Strands agents."""

    def __init__(self) -> None:
        """Initialize the state manager with empty state."""
        self._state: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._callbacks: List[Callable[[Dict[str, Any], Dict[str, Any]], None]] = []

    def get_state(self) -> Dict[str, Any]:
        """Get current state snapshot."""
        with self._lock:
            return self._state.copy()

    def update_state(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update state and return the new state."""
        with self._lock:
            self._state.update(updates)
            new_state = self._state.copy()

        # Notify callbacks of state change
        for callback in self._callbacks:
            callback(new_state, updates)

        return new_state

    def set_state(self, new_state: Dict[str, Any]) -> Dict[str, Any]:
        """Replace entire state."""
        with self._lock:
            old_state = self._state.copy()
            self._state = new_state.copy()

        # Calculate delta for callbacks
        delta = self._calculate_delta(old_state, new_state)
        for callback in self._callbacks:
            callback(new_state, delta)

        return new_state

    def add_callback(self, callback: Any) -> None:
        """Add callback for state changes."""
        self._callbacks.append(callback)

    def _calculate_delta(self, old_state: Dict, new_state: Dict) -> Dict[str, Any]:
        """Calculate state delta between old and new state."""
        delta = {}

        # Find changed/added keys
        for key, value in new_state.items():
            if key not in old_state or old_state[key] != value:
                delta[key] = value

        # Find removed keys
        for key in old_state:
            if key not in new_state:
                delta[key] = None

        return delta


# Global state manager instance
_state_manager = StrandsStateManager()


def get_state_manager() -> StrandsStateManager:
    """Get the global state manager instance."""
    return _state_manager


@tool
def get_agent_state() -> Dict[str, Any]:
    """Get the current agent state.

    This tool allows the agent to read its current state,
    which is synchronized with the frontend.

    Returns:
        Dictionary containing the current agent state
    """
    state = _state_manager.get_state()
    return {"status": "success", "content": [{"text": f"Current agent state: {json.dumps(state, indent=2)}"}]}


@tool
def update_agent_state(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update specific keys in the agent state.

    This tool allows the agent to update its state, which will be
    synchronized with the frontend and trigger UI updates.

    Args:
        updates: Dictionary of state updates to apply

    Returns:
        Dictionary with the updated state
    """
    try:
        _state_manager.update_state(updates)
        return {"status": "success", "content": [{"text": f"Updated state with: {json.dumps(updates, indent=2)}"}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Failed to update state: {str(e)}"}]}


@tool
def set_agent_state(new_state: Dict[str, Any]) -> Dict[str, Any]:
    """Replace the entire agent state.

    This tool allows the agent to completely replace its state,
    which will be synchronized with the frontend.

    Args:
        new_state: Complete new state to set

    Returns:
        Dictionary with the new state
    """
    try:
        _state_manager.set_state(new_state)
        return {"status": "success", "content": [{"text": f"Set new state: {json.dumps(new_state, indent=2)}"}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Failed to set state: {str(e)}"}]}


@tool
def emit_ui_update(component_name: str, props: Dict[str, Any]) -> Dict[str, Any]:
    """Emit a UI update event for a specific component.

    This tool allows the agent to trigger specific UI updates
    by sending component props to the frontend.

    Args:
        component_name: Name of the UI component to update
        props: Properties/data to send to the component

    Returns:
        Confirmation of the UI update emission
    """
    try:
        # Update state with UI-specific data
        ui_updates = {
            f"ui_{component_name}": props,
            "last_ui_update": {"component": component_name, "timestamp": datetime.now().isoformat(), "props": props},
        }

        _state_manager.update_state(ui_updates)

        return {"status": "success", "content": [{"text": f"Emitted UI update for {component_name}"}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Failed to emit UI update: {str(e)}"}]}


def setup_agent_state_management(agent: Any, initial_state: Optional[Dict[str, Any]] = None) -> StrandsStateManager:
    """Set up state management for a Strands agent.

    Args:
        agent: The Strands agent instance
        initial_state: Optional initial state to set

    Returns:
        The configured state manager
    """
    # Import here to avoid circular imports
    from ..tools.tools import FunctionTool

    # Add state management tools to agent
    state_tools = [get_agent_state, update_agent_state, set_agent_state, emit_ui_update]

    # Add tools to agent's tool registry using FunctionTool wrapper
    for tool_func in state_tools:
        function_tool = FunctionTool(cast(Callable[[ToolUse], ToolResult], tool_func))
        agent.tool_registry.register_tool(function_tool)

    # Set initial state if provided
    if initial_state:
        _state_manager.set_state(initial_state)

    return _state_manager
