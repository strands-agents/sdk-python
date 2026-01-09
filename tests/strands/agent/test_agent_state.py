"""Tests for AgentState class."""

from datetime import datetime
from uuid import uuid4

import pytest

from strands import Agent, JSONSerializer, PickleSerializer, tool
from strands.agent.state import AgentState
from strands.types.content import Messages

from ...fixtures.mocked_model_provider import MockedModelProvider


def test_set_and_get():
    """Test basic set and get operations."""
    state = AgentState()
    state.set("key", "value")
    assert state.get("key") == "value"


def test_get_nonexistent_key():
    """Test getting nonexistent key returns None."""
    state = AgentState()
    assert state.get("nonexistent") is None


def test_get_entire_state():
    """Test getting entire state when no key specified."""
    state = AgentState()
    state.set("key1", "value1")
    state.set("key2", "value2")

    result = state.get()
    assert result == {"key1": "value1", "key2": "value2"}


def test_initialize_and_get_entire_state():
    """Test getting entire state when no key specified."""
    state = AgentState({"key1": "value1", "key2": "value2"})

    result = state.get()
    assert result == {"key1": "value1", "key2": "value2"}


def test_initialize_with_error():
    with pytest.raises(ValueError, match="not JSON serializable"):
        AgentState({"object", object()})


def test_delete():
    """Test deleting keys."""
    state = AgentState()
    state.set("key1", "value1")
    state.set("key2", "value2")

    state.delete("key1")

    assert state.get("key1") is None
    assert state.get("key2") == "value2"


def test_delete_nonexistent_key():
    """Test deleting nonexistent key doesn't raise error."""
    state = AgentState()
    state.delete("nonexistent")  # Should not raise


def test_json_serializable_values():
    """Test that only JSON-serializable values are accepted."""
    state = AgentState()

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
    state = AgentState()

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
    state = AgentState(initial_state=initial)

    assert state.get("key1") == "value1"
    assert state.get("key2") == "value2"
    assert state.get() == initial


def test_agent_state_update_from_tool():
    @tool
    def update_state(agent: Agent):
        agent.state.set("hello", "world")
        agent.state.set("foo", "baz")

    agent_messages: Messages = [
        {
            "role": "assistant",
            "content": [{"toolUse": {"name": "update_state", "toolUseId": "123", "input": {}}}],
        },
        {"role": "assistant", "content": [{"text": "I invoked a tool!"}]},
    ]
    mocked_model_provider = MockedModelProvider(agent_messages)

    agent = Agent(
        model=mocked_model_provider,
        tools=[update_state],
        state={"foo": "bar"},
    )

    assert agent.state.get("hello") is None
    assert agent.state.get("foo") == "bar"

    agent("Invoke Mocked!")

    assert agent.state.get("hello") == "world"
    assert agent.state.get("foo") == "baz"


def test_default_serializer_is_json():
    """Test that default serializer is JSONSerializer."""
    state = AgentState()
    assert isinstance(state.serializer, JSONSerializer)


def test_pickle_serializer_allows_rich_types():
    """Test that PickleSerializer allows datetime, UUID, and other rich types."""
    state = AgentState(serializer=PickleSerializer())

    # Rich types that don't work with JSONSerializer
    now = datetime.now()
    user_id = uuid4()

    state.set("created_at", now)
    state.set("user_id", user_id)
    state.set("config", {"nested": now})

    assert state.get("created_at") == now
    assert state.get("user_id") == user_id


def test_json_serializer_rejects_rich_types():
    """Test that JSONSerializer rejects datetime and other non-JSON types."""
    state = AgentState(serializer=JSONSerializer())

    with pytest.raises(ValueError, match="not JSON serializable"):
        state.set("created_at", datetime.now())


def test_serialize_deserialize_json():
    """Test serialize and deserialize with JSONSerializer."""
    state = AgentState(serializer=JSONSerializer())
    state.set("name", "test")
    state.set("count", 42)

    # Serialize
    data = state.serialize()
    assert isinstance(data, bytes)

    # Deserialize into new state
    new_state = AgentState(serializer=JSONSerializer())
    new_state.deserialize(data)

    assert new_state.get("name") == "test"
    assert new_state.get("count") == 42


def test_serialize_deserialize_pickle():
    """Test serialize and deserialize with PickleSerializer."""
    state = AgentState(serializer=PickleSerializer())
    now = datetime.now()
    user_id = uuid4()
    state.set("created_at", now)
    state.set("user_id", user_id)

    # Serialize
    data = state.serialize()
    assert isinstance(data, bytes)

    # Deserialize into new state
    new_state = AgentState(serializer=PickleSerializer())
    new_state.deserialize(data)

    assert new_state.get("created_at") == now
    assert new_state.get("user_id") == user_id


def test_serializer_property():
    """Test serializer property getter and setter."""
    state = AgentState(serializer=JSONSerializer())
    assert isinstance(state.serializer, JSONSerializer)

    # Change serializer
    state.serializer = PickleSerializer()
    assert isinstance(state.serializer, PickleSerializer)


def test_agent_state_with_pickle_allows_datetime():
    """Test using datetime in agent state with PickleSerializer."""
    agent_messages: Messages = [
        {"role": "assistant", "content": [{"text": "Hello!"}]},
    ]
    mocked_model_provider = MockedModelProvider(agent_messages)

    now = datetime.now()
    agent = Agent(
        model=mocked_model_provider,
        state=AgentState(serializer=PickleSerializer()),
    )

    agent.state.set("created_at", now)
    assert agent.state.get("created_at") == now


def test_pickle_serializer_rejects_unpicklable():
    """Test that PickleSerializer rejects unpicklable objects like DB connections."""
    import sqlite3

    state = AgentState(serializer=PickleSerializer())
    conn = sqlite3.connect(":memory:")

    with pytest.raises(ValueError, match="not picklable"):
        state.set("connection", conn)

    conn.close()
