"""Agent state management."""

import copy
import json
from typing import Any, Dict, Optional


class AgentState:
    """Represents an Agent's stateful information outside of context provided to a model.

    Provides a key-value store for agent state with JSON serialization validation and persistence support.
    Key features:
    - JSON serialization validation on assignment
    - Get/set/delete operations
    """

    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        """Initialize AgentState."""
        self._state: Dict[str, Dict[str, Any]]
        if initial_state:
            self._validate_json_serializable(initial_state)
            self._state = copy.deepcopy(initial_state)
        else:
            self._state = {}

    def set(self, key: str, value: Any) -> None:
        """Set a value in the state.

        Args:
            key: The key to store the value under
            value: The value to store (must be JSON serializable)

        Raises:
            ValueError: If key is invalid, or if value is not JSON serializable
        """
        self._validate_key(key)
        self._validate_json_serializable(value)

        self._state[key] = copy.deepcopy(value)

    def get(self, key: Optional[str] = None) -> Any:
        """Get a value or entire state.

        Args:
            key: The key to retrieve (if None, returns entire state object)

        Returns:
            The stored value, entire state dict, or None if not found
        """
        if key is None:
            return copy.deepcopy(self._state)
        else:
            # Return specific key
            return copy.deepcopy(self._state.get(key))

    def delete(self, key: str) -> None:
        """Delete a specific key from the state.

        Args:
            key: The key to delete
        """
        self._validate_key(key)

        self._state.pop(key, None)

    def _validate_key(self, key: str) -> None:
        """Validate that a key is valid.

        Args:
            key: The key to validate

        Raises:
            ValueError: If key is invalid
        """
        if key is None:
            raise ValueError("Key cannot be None")
        if not isinstance(key, str):
            raise ValueError("Key must be a string")
        if not key.strip():
            raise ValueError("Key cannot be empty")

    def _validate_json_serializable(self, value: Any) -> None:
        """Validate that a value is JSON serializable.

        Args:
            value: The value to validate

        Raises:
            ValueError: If value is not JSON serializable
        """
        try:
            json.dumps(value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Value is not JSON serializable: {type(value).__name__}. "
                f"Only JSON-compatible types (str, int, float, bool, list, dict, None) are allowed."
            ) from e
