"""JSON serializable dictionary utilities."""

import copy
import json
from typing import Any


class JSONSerializableDict:
    """A key-value store with JSON serialization validation.

    Provides a dict-like interface with automatic validation that all values
    are JSON serializable on assignment.
    """

    def __init__(self, initial_state: dict[str, Any] | None = None):
        """Initialize JSONSerializableDict."""
        self._data: dict[str, Any]
        self._version: int = 0
        if initial_state:
            self._validate_json_serializable(initial_state)
            self._data = copy.deepcopy(initial_state)
        else:
            self._data = {}

    def set(self, key: str, value: Any) -> None:
        """Set a value in the store.

        Args:
            key: The key to store the value under
            value: The value to store (must be JSON serializable)

        Raises:
            ValueError: If key is invalid, or if value is not JSON serializable
        """
        self._validate_key(key)
        self._validate_json_serializable(value)
        self._data[key] = copy.deepcopy(value)
        self._version += 1

    def get(self, key: str | None = None) -> Any:
        """Get a value or entire data.

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
        self._version += 1

    def _get_version(self) -> int:
        """Get the current version number of the store.

        The version is incremented each time set() or delete() is called.
        Consumers can compare versions to detect changes without requiring
        explicit dirty flag clearing.

        Returns:
            The current version number.
        """
        return self._version

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
