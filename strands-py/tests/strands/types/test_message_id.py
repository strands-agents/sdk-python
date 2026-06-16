"""Tests for the durable message id, its generator, and get_message_id."""

from uuid import UUID

from strands.types.content import Message, _generate_message_id, get_message_id


def test_get_message_id_without_id():
    msg: Message = {"role": "assistant", "content": [{"text": "hello"}]}
    assert get_message_id(msg) is None


def test_get_message_id_with_id():
    msg: Message = {"role": "assistant", "content": [{"text": "hello"}], "id": "abc123"}
    assert get_message_id(msg) == "abc123"


def test_generate_message_id_is_unique():
    ids = {_generate_message_id() for _ in range(1000)}
    assert len(ids) == 1000


def test_generate_message_id_is_canonical_uuid_v4():
    message_id = _generate_message_id()
    assert isinstance(message_id, str)
    parsed = UUID(message_id)  # raises if not a valid UUID
    # Canonical hyphenated form, matching the TypeScript SDK's crypto.randomUUID() shape.
    assert str(parsed) == message_id
    assert parsed.version == 4


def test_id_does_not_affect_role_and_content():
    msg: Message = {"role": "assistant", "content": [{"text": "hello"}], "id": "abc123"}
    assert msg["role"] == "assistant"
    assert msg["content"] == [{"text": "hello"}]
