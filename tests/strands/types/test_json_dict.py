"""Tests for JSONSerializableDict class."""

import pytest

from strands.types.json_dict import JSONSerializableDict


def test_set_and_get():
    """Test basic set and get operations."""
    state = JSONSerializableDict()
    state.set("key", "value")
    assert state.get("key") == "value"


def test_get_nonexistent_key():
    """Test getting nonexistent key returns None."""
    state = JSONSerializableDict()
    assert state.get("nonexistent") is None


def test_get_entire_state():
    """Test getting entire state when no key specified."""
    state = JSONSerializableDict()
    state.set("key1", "value1")
    state.set("key2", "value2")

    result = state.get()
    assert result == {"key1": "value1", "key2": "value2"}


def test_initialize_and_get_entire_state():
    """Test getting entire state when no key specified."""
    state = JSONSerializableDict({"key1": "value1", "key2": "value2"})

    result = state.get()
    assert result == {"key1": "value1", "key2": "value2"}


def test_initialize_with_error():
    with pytest.raises(ValueError, match="not JSON serializable"):
        JSONSerializableDict({"object", object()})


def test_delete():
    """Test deleting keys."""
    state = JSONSerializableDict()
    state.set("key1", "value1")
    state.set("key2", "value2")

    state.delete("key1")

    assert state.get("key1") is None
    assert state.get("key2") == "value2"


def test_delete_nonexistent_key():
    """Test deleting nonexistent key doesn't raise error."""
    state = JSONSerializableDict()
    state.delete("nonexistent")  # Should not raise


def test_json_serializable_values():
    """Test that only JSON-serializable values are accepted."""
    state = JSONSerializableDict()

    # Valid JSON types
    state.set("string", "test")
    state.set("int", 42)
    state.set("bool", True)
    state.set("list", [1, 2, 3])
    state.set("dict", {"nested": "value"})
    state.set("null", None)

    # Invalid JSON types should raise ValueError
    with pytest.raises(ValueError, match="not JSON serializable"):
        state.set("function", lambda x: x)

    with pytest.raises(ValueError, match="not JSON serializable"):
        state.set("object", object())


def test_key_validation():
    """Test key validation for set and delete operations."""
    state = JSONSerializableDict()

    # Invalid keys for set
    with pytest.raises(ValueError, match="Key cannot be None"):
        state.set(None, "value")

    with pytest.raises(ValueError, match="Key cannot be empty"):
        state.set("", "value")

    with pytest.raises(ValueError, match="Key must be a string"):
        state.set(123, "value")

    # Invalid keys for delete
    with pytest.raises(ValueError, match="Key cannot be None"):
        state.delete(None)

    with pytest.raises(ValueError, match="Key cannot be empty"):
        state.delete("")


def test_initial_state():
    """Test initialization with initial state."""
    initial = {"key1": "value1", "key2": "value2"}
    state = JSONSerializableDict(initial_state=initial)

    assert state.get("key1") == "value1"
    assert state.get("key2") == "value2"
    assert state.get() == initial


# ============================================================================
# Dirty Flag Tests
# ============================================================================


def test_is_dirty_returns_false_after_initialization():
    """Test that _is_dirty() returns False after initialization."""
    state = JSONSerializableDict()
    assert state._is_dirty() is False


def test_is_dirty_returns_false_after_initialization_with_initial_state():
    """Test that _is_dirty() returns False when initialized with initial_state."""
    state = JSONSerializableDict(initial_state={"key": "value"})
    assert state._is_dirty() is False


def test_is_dirty_returns_true_after_set():
    """Test that _is_dirty() returns True after set() is called."""
    state = JSONSerializableDict()
    state.set("key", "value")
    assert state._is_dirty() is True


def test_is_dirty_returns_true_after_delete():
    """Test that _is_dirty() returns True after delete() is called."""
    state = JSONSerializableDict(initial_state={"key": "value"})
    state.delete("key")
    assert state._is_dirty() is True


def test_is_dirty_returns_true_after_delete_nonexistent_key():
    """Test that _is_dirty() returns True after delete() on nonexistent key."""
    state = JSONSerializableDict()
    state.delete("nonexistent")
    assert state._is_dirty() is True


def test_clear_dirty_resets_flag_to_false():
    """Test that _clear_dirty() resets dirty flag to False."""
    state = JSONSerializableDict()
    state.set("key", "value")
    assert state._is_dirty() is True

    state._clear_dirty()
    assert state._is_dirty() is False


def test_dirty_flag_set_after_clear():
    """Test that dirty flag is set after clear and subsequent set()."""
    state = JSONSerializableDict()
    state.set("key", "value")
    state._clear_dirty()

    state.set("key2", "value2")
    assert state._is_dirty() is True


def test_dirty_flag_set_after_clear_and_delete():
    """Test that dirty flag is set after clear and subsequent delete()."""
    state = JSONSerializableDict(initial_state={"key": "value"})
    state._clear_dirty()

    state.delete("key")
    assert state._is_dirty() is True
