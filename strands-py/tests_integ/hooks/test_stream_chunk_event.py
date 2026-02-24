"""Integration tests for BeforeStreamChunkEvent hook."""

import pytest

from strands import Agent
from strands.hooks import BeforeStreamChunkEvent


@pytest.fixture
def chunks_intercepted():
    return []


@pytest.fixture
def agent_with_stream_hook(chunks_intercepted):
    """Create an agent with BeforeStreamChunkEvent hook registered."""

    async def intercept_chunks(event: BeforeStreamChunkEvent):
        chunks_intercepted.append(event.chunk.copy())

    agent = Agent(system_prompt="Be very brief. Reply with one word only.")
    agent.hooks.add_callback(BeforeStreamChunkEvent, intercept_chunks)
    return agent


def test_before_stream_chunk_event_fires(agent_with_stream_hook, chunks_intercepted):
    """Test that BeforeStreamChunkEvent fires for each stream chunk."""
    agent_with_stream_hook("Say hello")

    # Should have intercepted multiple chunks
    assert len(chunks_intercepted) > 0

    # Should have message start, content blocks, and message stop
    chunk_types = set()
    for chunk in chunks_intercepted:
        chunk_types.update(chunk.keys())

    assert "messageStart" in chunk_types
    assert "messageStop" in chunk_types


@pytest.mark.asyncio
async def test_before_stream_chunk_event_modification():
    """Test that chunk modifications affect both stream events and final message."""
    modified_text = "[REDACTED]"

    async def redact_chunks(event: BeforeStreamChunkEvent):
        if "contentBlockDelta" in event.chunk:
            delta = event.chunk.get("contentBlockDelta", {}).get("delta", {})
            if "text" in delta:
                event.chunk = {"contentBlockDelta": {"delta": {"text": modified_text}}}

    agent = Agent(system_prompt="Say exactly: secret123")
    agent.hooks.add_callback(BeforeStreamChunkEvent, redact_chunks)

    text_events = []
    result = None

    async for event in agent.stream_async("go"):
        if "data" in event:
            text_events.append(event["data"])
        if "result" in event:
            result = event["result"]

    # All text events should be the modified text
    assert all(text == modified_text for text in text_events)

    # Final message should only contain modified text
    final_text = result.message["content"][0].get("text", "")
    assert modified_text in final_text
    assert "secret" not in final_text.lower()


@pytest.mark.asyncio
async def test_before_stream_chunk_event_skip():
    """Test that skip=True excludes chunks from processing and final message."""

    async def skip_content_deltas(event: BeforeStreamChunkEvent):
        # Skip all content block deltas (text content)
        if "contentBlockDelta" in event.chunk:
            event.skip = True

    agent = Agent(system_prompt="Say hello")
    agent.hooks.add_callback(BeforeStreamChunkEvent, skip_content_deltas)

    text_events = []
    result = None

    async for event in agent.stream_async("go"):
        if "data" in event:
            text_events.append(event["data"])
        if "result" in event:
            result = event["result"]

    # No text events should be yielded
    assert len(text_events) == 0

    # Final message should have no content (all text was skipped)
    assert result.message["content"] == []


@pytest.mark.asyncio
async def test_before_stream_chunk_event_has_invocation_state():
    """Test that invocation_state is accessible in BeforeStreamChunkEvent."""
    received_states = []

    async def capture_state(event: BeforeStreamChunkEvent):
        received_states.append(event.invocation_state.copy())

    agent = Agent(system_prompt="Be brief")
    agent.hooks.add_callback(BeforeStreamChunkEvent, capture_state)

    custom_state = {"session_id": "test-123", "user_id": "user-456"}

    async for _ in agent.stream_async("hi", invocation_state=custom_state):
        pass

    # All captured states should have our custom keys
    assert len(received_states) > 0
    for state in received_states:
        assert state.get("session_id") == "test-123"
        assert state.get("user_id") == "user-456"
