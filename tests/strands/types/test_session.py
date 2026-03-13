import json
import unittest.mock
from uuid import uuid4

from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from strands.agent.state import AgentState
from strands.interrupt import _InterruptState
from strands.types.session import (
    Session,
    SessionAgent,
    SessionMessage,
    SessionType,
    decode_bytes_values,
    encode_bytes_values,
)


def test_session_json_serializable():
    session = Session(session_id=str(uuid4()), session_type=SessionType.AGENT)
    # json dumps will fail if its not json serializable
    session_json_string = json.dumps(session.to_dict())
    loaded_session = Session.from_dict(json.loads(session_json_string))
    assert loaded_session is not None


def test_agent_json_serializable():
    agent = SessionAgent(
        agent_id=str(uuid4()), state={"foo": "bar"}, conversation_manager_state=NullConversationManager().get_state()
    )
    # json dumps will fail if its not json serializable
    agent_json_string = json.dumps(agent.to_dict())
    loaded_agent = SessionAgent.from_dict(json.loads(agent_json_string))
    assert loaded_agent is not None


def test_message_json_serializable():
    message = SessionMessage(message={"role": "user", "content": [{"text": "Hello!"}]}, message_id=0)
    # json dumps will fail if its not json serializable
    message_json_string = json.dumps(message.to_dict())
    loaded_message = SessionMessage.from_dict(json.loads(message_json_string))
    assert loaded_message is not None


def test_bytes_encoding_decoding():
    # Test simple bytes
    test_bytes = b"Hello, world!"
    encoded = encode_bytes_values(test_bytes)
    assert isinstance(encoded, dict)
    assert encoded["__bytes_encoded__"] is True
    decoded = decode_bytes_values(encoded)
    assert decoded == test_bytes

    # Test nested structure with bytes
    test_data = {
        "text": "Hello",
        "binary": b"Binary data",
        "nested": {"more_binary": b"More binary data", "list_with_binary": [b"Item 1", "Text item", b"Item 3"]},
    }

    encoded = encode_bytes_values(test_data)
    # Verify it's JSON serializable
    json_str = json.dumps(encoded)
    # Deserialize and decode
    decoded = decode_bytes_values(json.loads(json_str))

    # Verify the decoded data matches the original
    assert decoded["text"] == test_data["text"]
    assert decoded["binary"] == test_data["binary"]
    assert decoded["nested"]["more_binary"] == test_data["nested"]["more_binary"]
    assert decoded["nested"]["list_with_binary"][0] == test_data["nested"]["list_with_binary"][0]
    assert decoded["nested"]["list_with_binary"][1] == test_data["nested"]["list_with_binary"][1]
    assert decoded["nested"]["list_with_binary"][2] == test_data["nested"]["list_with_binary"][2]


def test_session_message_with_bytes():
    # Create a message with bytes content
    message = {
        "role": "user",
        "content": [{"text": "Here is some binary data"}, {"binary_data": b"This is binary data"}],
    }

    # Create a SessionMessage
    session_message = SessionMessage.from_message(message, 0)

    # Verify it's JSON serializable
    message_json_string = json.dumps(session_message.to_dict())

    # Load it back
    loaded_message = SessionMessage.from_dict(json.loads(message_json_string))

    # Convert back to original message and verify
    original_message = loaded_message.to_message()

    assert original_message["role"] == message["role"]
    assert original_message["content"][0]["text"] == message["content"][0]["text"]
    assert original_message["content"][1]["binary_data"] == message["content"][1]["binary_data"]


def test_session_agent_from_agent():
    agent = unittest.mock.Mock()
    agent.agent_id = "a1"
    agent.conversation_manager = unittest.mock.Mock(get_state=lambda: {"test": "conversation"})
    agent.state = AgentState({"test": "state"})
    agent._interrupt_state = _InterruptState(interrupts={}, context={}, activated=False)

    tru_session_agent = SessionAgent.from_agent(agent)
    exp_session_agent = SessionAgent(
        agent_id="a1",
        conversation_manager_state={"test": "conversation"},
        state={"test": "state"},
        _internal_state={"interrupt_state": {"interrupts": {}, "context": {}, "activated": False}},
        created_at=unittest.mock.ANY,
        updated_at=unittest.mock.ANY,
    )
    assert tru_session_agent == exp_session_agent


def test_session_agent_initialize_internal_state():
    agent = unittest.mock.Mock()
    session_agent = SessionAgent(
        agent_id="a1",
        conversation_manager_state={},
        state={},
        _internal_state={"interrupt_state": {"interrupts": {}, "context": {"test": "init"}, "activated": False}},
    )

    session_agent.initialize_internal_state(agent)

    tru_interrupt_state = agent._interrupt_state
    exp_interrupt_state = _InterruptState(interrupts={}, context={"test": "init"}, activated=False)
    assert tru_interrupt_state == exp_interrupt_state


def test_session_agent_with_bytes():
    """SessionAgent.to_dict() must encode bytes so json.dumps() doesn't crash.

    This is the root cause of issue #1864: S3SessionManager fails when agent state
    contains binary document content (e.g., inline PDF bytes from multimodal prompts).
    """
    state_with_bytes = {
        "documents": [
            {
                "format": "pdf",
                "name": "document.pdf",
                "source": {"bytes": b"fake-pdf-binary-content"},
            }
        ]
    }

    session_agent = SessionAgent(
        agent_id="a1",
        conversation_manager_state={},
        state=state_with_bytes,
    )

    # Must be JSON-serializable (this is where #1864 crashes without the fix)
    agent_dict = session_agent.to_dict()
    json_str = json.dumps(agent_dict)

    # Round-trip: from_dict must decode bytes back
    loaded_agent = SessionAgent.from_dict(json.loads(json_str))
    assert loaded_agent.state["documents"][0]["source"]["bytes"] == b"fake-pdf-binary-content"
    assert loaded_agent.agent_id == "a1"


def test_session_with_bytes_in_session_type():
    """Session.to_dict() must encode any bytes values to remain JSON-safe."""
    session = Session(session_id="test-id", session_type=SessionType.AGENT)
    session_dict = session.to_dict()
    # Should not raise
    json.dumps(session_dict)
