"""State serializers for agent state management.

This module provides pluggable serialization strategies for AgentState:
- JSONSerializer: Default serializer, backward compatible, validates on set()
- PickleSerializer: Supports any Python object, no validation on set()
- StateSerializer: Protocol for custom serializers
"""

import copy
import json
import pickle
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StateSerializer(Protocol):
    """Protocol for state serializers.

    Custom serializers can implement this protocol to provide
    alternative serialization strategies for agent state.
    """

    def serialize(self, data: dict[str, Any]) -> bytes:
        """Serialize state dict to bytes.

        Args:
            data: Dictionary of state data to serialize

        Returns:
            Serialized state as bytes
        """
        ...

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """Deserialize bytes back to state dict.

        Args:
            data: Serialized state bytes

        Returns:
            Deserialized state dictionary
        """
        ...

    def validate(self, value: Any) -> None:
        """Validate a value can be serialized.

        Serializers that accept any value should implement this as a no-op.

        Args:
            value: The value to validate

        Raises:
            ValueError: If value cannot be serialized by this serializer
        """
        ...


class JSONSerializer:
    """JSON-based state serializer.

    Default serializer that provides:
    - Human-readable serialization format
    - Validation on set() to maintain current behavior
    - Backward compatibility with existing code
    """

    def serialize(self, data: dict[str, Any]) -> bytes:
        """Serialize state dict to JSON bytes.

        Args:
            data: Dictionary of state data to serialize

        Returns:
            JSON serialized state as bytes
        """
        return json.dumps(data).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """Deserialize JSON bytes back to state dict.

        Args:
            data: JSON serialized state bytes

        Returns:
            Deserialized state dictionary
        """
        result: dict[str, Any] = json.loads(data.decode("utf-8"))
        return result

    def validate(self, value: Any) -> None:
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


class PickleSerializer:
    """Pickle-based state serializer.

    Provides:
    - Support for any Python object (datetime, UUID, dataclass, Pydantic models, etc.)
    - No validation on set() (accepts anything)

    Security Warning:
        Pickle can execute arbitrary code during deserialization.
        Only unpickle data from trusted sources.
    """

    def serialize(self, data: dict[str, Any]) -> bytes:
        """Serialize state dict using pickle.

        Args:
            data: Dictionary of state data to serialize

        Returns:
            Pickle serialized state as bytes
        """
        return pickle.dumps(copy.deepcopy(data))

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """Deserialize pickle bytes back to state dict.

        Args:
            data: Pickle serialized state bytes

        Returns:
            Deserialized state dictionary
        """
        result: dict[str, Any] = pickle.loads(data)  # noqa: S301
        return result

    def validate(self, value: Any) -> None:
        """No-op validation - pickle accepts any Python object.

        Args:
            value: The value to validate (ignored)
        """
        pass
