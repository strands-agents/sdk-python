"""Test BeforeStreamChunkEvent with a real OpenAI model.

Tests three scenarios:
1. Chunk capture - intercept and log all stream chunks
2. Chunk modification - redact text in-flight
3. Chunk skipping - suppress content deltas entirely
"""

import asyncio
import os

from strands import Agent
from strands.hooks import BeforeStreamChunkEvent
from strands.models.openai import OpenAIModel


def make_model():
    return OpenAIModel(
        model_id="gpt-4o-mini",
        client_args={"api_key": os.environ["OPENAI_API_KEY"]},
    )


# ---------- Test 1: capture chunks ----------
def test_capture():
    print("=" * 60)
    print("TEST 1: Capture every stream chunk")
    print("=" * 60)

    chunks = []

    async def on_chunk(event: BeforeStreamChunkEvent):
        chunks.append(event.chunk.copy())

    agent = Agent(
        model=make_model(),
        system_prompt="Reply in exactly 5 words.",
    )
    agent.hooks.add_callback(BeforeStreamChunkEvent, on_chunk)

    result = agent("Say hello")

    print(f"\nFinal text : {result.message['content'][0].get('text', '')}")
    print(f"Chunks seen: {len(chunks)}")

    chunk_types = set()
    for c in chunks:
        chunk_types.update(c.keys())
    print(f"Chunk types: {sorted(chunk_types)}")

    assert len(chunks) > 0, "Expected at least one chunk"
    assert "messageStart" in chunk_types, "Missing messageStart"
    assert "contentBlockDelta" in chunk_types, "Missing contentBlockDelta"
    print("PASSED\n")


# ---------- Test 2: modify (redact) chunks ----------
def test_modify():
    print("=" * 60)
    print("TEST 2: Redact text in-flight")
    print("=" * 60)

    async def redact(event: BeforeStreamChunkEvent):
        if "contentBlockDelta" in event.chunk:
            delta = event.chunk.get("contentBlockDelta", {}).get("delta", {})
            if "text" in delta:
                event.chunk = {"contentBlockDelta": {"delta": {"text": "[REDACTED]"}}}

    agent = Agent(
        model=make_model(),
        system_prompt="Say exactly: the secret code is 12345",
    )
    agent.hooks.add_callback(BeforeStreamChunkEvent, redact)

    text_events = []
    result = None

    async def run():
        nonlocal result
        async for event in agent.stream_async("go"):
            if "data" in event:
                text_events.append(event["data"])
            if "result" in event:
                result = event["result"]

    asyncio.run(run())

    final_text = result.message["content"][0].get("text", "")
    print(f"\nStreamed text events: {text_events}")
    print(f"Final message text : {final_text}")

    assert all(t == "[REDACTED]" for t in text_events), "Some text events were not redacted"
    assert "secret" not in final_text.lower(), "Final message still contains secret"
    assert "[REDACTED]" in final_text, "Final message missing redaction marker"
    print("PASSED\n")


# ---------- Test 3: skip content deltas ----------
def test_skip():
    print("=" * 60)
    print("TEST 3: Skip content deltas entirely")
    print("=" * 60)

    async def skip_deltas(event: BeforeStreamChunkEvent):
        if "contentBlockDelta" in event.chunk:
            event.skip = True

    agent = Agent(
        model=make_model(),
        system_prompt="Reply with a greeting",
    )
    agent.hooks.add_callback(BeforeStreamChunkEvent, skip_deltas)

    text_events = []
    result = None

    async def run():
        nonlocal result
        async for event in agent.stream_async("hi"):
            if "data" in event:
                text_events.append(event["data"])
            if "result" in event:
                result = event["result"]

    asyncio.run(run())

    print(f"\nText events received: {len(text_events)}")
    print(f"Final content       : {result.message['content']}")

    assert len(text_events) == 0, f"Expected 0 text events, got {len(text_events)}"
    assert result.message["content"] == [], "Expected empty content"
    print("PASSED\n")


if __name__ == "__main__":
    test_capture()
    test_modify()
    test_skip()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
