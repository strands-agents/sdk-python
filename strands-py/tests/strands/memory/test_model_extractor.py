"""Tests for ``ModelExtractor`` parsing and behavior.

Ported from ``strands-ts/src/memory/extraction/__tests__/model-extractor.test.ts``.
Each TS ``it(...)`` maps to a ``test_...`` case here.

The extractor is driven through its public ``extract`` API: a fake ``Model``
(``MockedModelProvider``) yields stream events that
``strands.event_loop.streaming.stream_messages`` aggregates into a chosen
assistant message, so we exercise the real aggregate-and-parse path.

These tests are written test-first: ``ModelExtractor`` lands in Task 6, so they
are expected to fail with an ``ImportError`` until that implementation exists.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands.memory.extraction.model_extractor import ModelExtractor
from strands.memory.extraction.types import ExtractionResult, ExtractorContext
from strands.types.content import Message
from tests.fixtures.mocked_model_provider import MockedModelProvider


def _user_turn(text: str) -> Message:
    """Build a single user turn, mirroring the TS ``userTurn`` helper."""
    return {"role": "user", "content": [{"text": text}]}


def _assistant_text(text: str) -> Message:
    """Build an assistant response carrying ``text`` for the fake model to emit."""
    return {"role": "assistant", "content": [{"text": text}]}


class _RecordingModel(MockedModelProvider):
    """A fake model that counts how many times ``stream`` is invoked.

    Used to assert the extractor short-circuits an empty batch without ever
    calling the model.
    """

    def __init__(self, agent_responses: Any) -> None:
        super().__init__(agent_responses)
        self.stream_calls = 0

    async def stream(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        self.stream_calls += 1
        async for event in super().stream(*args, **kwargs):
            yield event


@pytest.mark.asyncio
async def test_parses_a_json_array_of_entries_from_the_model_response():
    model = MockedModelProvider(
        [_assistant_text('[{"content": "User prefers dark mode"}, {"content": "Lives in Berlin"}]')]
    )
    extractor = ModelExtractor(model=model)

    entries = await extractor.extract([_user_turn("I like dark mode and live in Berlin")])

    assert entries == [
        ExtractionResult(content="User prefers dark mode"),
        ExtractionResult(content="Lives in Berlin"),
    ]


@pytest.mark.asyncio
async def test_extracts_a_json_array_even_when_wrapped_in_prose_or_a_code_fence():
    model = MockedModelProvider(
        [_assistant_text('Here are the facts:\n```json\n[{"content": "fact"}]\n```\nHope that helps.')]
    )
    extractor = ModelExtractor(model=model)

    entries = await extractor.extract([_user_turn("something")])

    assert entries == [ExtractionResult(content="fact")]


@pytest.mark.asyncio
async def test_preserves_entry_metadata():
    model = MockedModelProvider([_assistant_text('[{"content": "fact", "metadata": {"topic": "pref"}}]')])
    extractor = ModelExtractor(model=model)

    entries = await extractor.extract([_user_turn("x")])

    assert entries == [ExtractionResult(content="fact", metadata={"topic": "pref"})]


@pytest.mark.asyncio
async def test_returns_no_entries_on_malformed_json_without_throwing():
    model = MockedModelProvider([_assistant_text("not json at all")])
    extractor = ModelExtractor(model=model)

    entries = await extractor.extract([_user_turn("x")])

    assert entries == []


@pytest.mark.asyncio
async def test_drops_entries_without_a_string_content_and_empty_strings():
    model = MockedModelProvider([_assistant_text('[{"content": "keep"}, {"content": ""}, {"foo": "bar"}, "loose"]')])
    extractor = ModelExtractor(model=model)

    entries = await extractor.extract([_user_turn("x")])

    assert entries == [ExtractionResult(content="keep")]


@pytest.mark.asyncio
async def test_returns_empty_for_an_empty_message_batch_without_calling_the_model():
    model = _RecordingModel([])
    extractor = ModelExtractor(model=model)

    entries = await extractor.extract([])

    assert entries == []
    assert model.stream_calls == 0


@pytest.mark.asyncio
async def test_falls_back_to_the_default_model_from_context_when_none_configured():
    model = MockedModelProvider([_assistant_text('[{"content": "fact"}]')])
    extractor = ModelExtractor()

    entries = await extractor.extract([_user_turn("x")], ExtractorContext(default_model=model))

    assert entries == [ExtractionResult(content="fact")]


@pytest.mark.asyncio
async def test_raises_when_no_model_is_configured_and_no_default_is_provided():
    extractor = ModelExtractor()

    with pytest.raises(Exception, match="no model configured"):
        await extractor.extract([_user_turn("x")])
