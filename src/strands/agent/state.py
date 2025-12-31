"""Agent state management.

Provides flexible state container with pluggable serialization and transient state support.
"""

import copy
from typing import Any

from .serializers import JSONSerializer, StateSerializer


class AgentState:
    """Flexible state container with pluggable serialization and transient state support.

    AgentState provides a key-value store for agent state with:
    - Pluggable serialization (JSON by default, Pickle for rich types)
    - Transient state support for runtime-only resources (persist=False)
    - Backward compatible API with existing code

    Example:
        Basic usage (backward compatible):
        ```python
        state = AgentState()
        state.set("count", 42)  # Persistent by default
        state.get("count")  # Returns 42
        ```

        Rich types with PickleSerializer:
        ```python
        from strands.agent.serializers import PickleSerializer
        from datetime import datetime

        state = AgentState(serializer=PickleSerializer())
        state.set("created_at", datetime.now())  # Works with Pickle
        ```

        Transient state for runtime resources:
        ```python
        state.set("db_connection", connection, persist=False)  # Not serialized
        state.get("db_connection")  # Returns the connection
        state.is_transient("db_connection")  # Returns True
        ```
    """

    def __init__(
        self,
        initial_state: dict[str, Any] | None = None,
        serializer: StateSerializer | None = None,
    ):
        """Initialize AgentState.

        Args:
            initial_state: Optional initial state dictionary
            serializer: Serializer to use for state persistence.
                Defaults to JSONSerializer for backward compatibility.

        Raises:
            ValueError: If initial_state contains non-serializable values (with JSONSerializer)
        """
        self._serializer = serializer if serializer is not None else JSONSerializer()
        self._transient_keys: set[str] = set()
        self._data: dict[str, Any]

        if initial_state:
            # Validate initial state
            self._serializer.validate(initial_state)
            self._data = copy.deepcopy(initial_state)
        else:
            self._data = {}

    @property
    def serializer(self) -> StateSerializer:
        """Get the current serializer.

        Returns:
            The serializer used for state persistence
        """
        return self._serializer

    @serializer.setter
    def serializer(self, value: StateSerializer) -> None:
        """Set the serializer.

        Args:
            value: New serializer to use for state persistence
        """
        self._serializer = value

    def set(self, key: str, value: Any, *, persist: bool = True) -> None:
        """Set a value in the store.

        Args:
            key: The key to store the value under
            value: The value to store
            persist: If False, value is transient (not serialized). Default True.

        Raises:
            ValueError: If key is invalid, or if value is not serializable
                (only when persist=True)
        """
        self._validate_key(key)

        if persist:
            # Validate serializable
            self._serializer.validate(value)
            self._transient_keys.discard(key)
        else:
            # Mark as transient - skip validation
            self._transient_keys.add(key)

        self._data[key] = copy.deepcopy(value)

    def get(self, key: str | None = None) -> Any:
        """Get a value or entire data.

        Works uniformly for both persistent and transient values.

        Args:
            key: The key to retrieve (if None, returns entire data dict)

        Returns:
            The stored value, entire data dict, or None if not found
        """
        if key is None:
            return copy.deepcopy(self._data)
        else:
            return copy.deepcopy(self._data.get(key))

    def delete(self, key: str) -> None:
        """Delete a specific key from the store.

        Args:
            key: The key to delete
        """
        self._validate_key(key)
        self._data.pop(key, None)
        self._transient_keys.discard(key)

    def is_transient(self, key: str) -> bool:
        """Check if a key is transient (not persisted).

        Args:
            key: The key to check

        Returns:
            True if the key is transient, False otherwise
        """
        return key in self._transient_keys

    def serialize(self) -> bytes:
        """Serialize only persistent keys.

        Returns:
            Serialized state as bytes (excludes transient keys)
        """
        persistent_data = {k: v for k, v in self._data.items() if k not in self._transient_keys}
        return self._serializer.serialize(persistent_data)

    def deserialize(self, data: bytes) -> None:
        """Deserialize persistent state.

        Transient keys are preserved if already in memory.

        Args:
            data: Serialized state bytes to restore
        """
        persistent_data = self._serializer.deserialize(data)
        # Keep transient keys in memory, replace persistent
        transient_data = {k: v for k, v in self._data.items() if k in self._transient_keys}
        self._data = {**persistent_data, **transient_data}

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
