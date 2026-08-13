"""Cache-point placement ahead of per-call content.

A point landing after per-call content writes a new entry every request and never reads one — which
total token counts do not reveal, so only these placement assertions catch it.
"""

from unittest.mock import MagicMock

import pytest

from strands.models import BedrockModel
from strands.models.model import CacheConfig

_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"


def _document(fmt: str = "csv") -> dict:
    return {"document": {"format": fmt, "name": "d", "source": {"bytes": b"a,b"}}}


@pytest.fixture
def model():
    built = BedrockModel(model_id=_MODEL_ID, cache_config=CacheConfig(strategy="auto"))
    built.client = MagicMock()
    return built


def _keys(content: list[dict]) -> list[str]:
    return [next(iter(block)) for block in content]


def test_places_cache_point_ahead_of_per_call_content(model):
    messages = [{"role": "user", "content": [{"text": "durable ask"}, {"text": "PER-CALL"}]}]

    formatted = model._format_bedrock_messages(messages, per_call_trailing_blocks=1)

    assert _keys(formatted[0]["content"]) == ["text", "cachePoint", "text"]


def test_places_cache_point_ahead_of_several_per_call_blocks(model):
    messages = [{"role": "user", "content": [{"text": "durable"}, {"text": "STATUS"}, {"text": "INJECTED"}]}]

    formatted = model._format_bedrock_messages(messages, per_call_trailing_blocks=2)

    assert _keys(formatted[0]["content"]) == ["text", "cachePoint", "text", "text"]


def test_appends_at_the_end_when_no_per_call_content(model):
    messages = [{"role": "user", "content": [{"text": "durable ask"}]}]

    formatted = model._format_bedrock_messages(messages, per_call_trailing_blocks=0)

    assert _keys(formatted[0]["content"]) == ["text", "cachePoint"]


def test_skips_cache_point_when_every_block_is_per_call(model):
    # Nothing durable ahead of the boundary, so there is no prefix worth caching.
    messages = [{"role": "user", "content": [{"text": "PER-CALL"}]}]

    formatted = model._format_bedrock_messages(messages, per_call_trailing_blocks=1)

    assert _keys(formatted[0]["content"]) == ["text"]


def test_per_call_boundary_steps_back_over_a_non_pdf_document(model):
    # Bedrock rejects a point directly after a non-PDF document.
    messages = [{"role": "user", "content": [{"text": "a"}, _document(), {"text": "PER-CALL"}]}]

    formatted = model._format_bedrock_messages(messages, per_call_trailing_blocks=1)

    keys = _keys(formatted[0]["content"])
    assert keys == ["text", "cachePoint", "document", "text"]


def test_per_call_boundary_dropped_when_a_document_leads_the_message(model):
    messages = [{"role": "user", "content": [_document(), {"text": "PER-CALL"}]}]

    formatted = model._format_bedrock_messages(messages, per_call_trailing_blocks=1)

    assert "cachePoint" not in _keys(formatted[0]["content"])


def test_per_call_boundary_keeps_a_pdf_document_in_the_cached_prefix(model):
    messages = [{"role": "user", "content": [{"text": "a"}, _document("pdf"), {"text": "PER-CALL"}]}]

    formatted = model._format_bedrock_messages(messages, per_call_trailing_blocks=1)

    assert _keys(formatted[0]["content"]) == ["text", "document", "cachePoint", "text"]


def test_per_call_boundary_carries_the_configured_ttl():
    built = BedrockModel(model_id=_MODEL_ID, cache_config=CacheConfig(strategy="auto", ttl="1h"))
    built.client = MagicMock()
    messages = [{"role": "user", "content": [{"text": "durable"}, {"text": "PER-CALL"}]}]

    formatted = built._format_bedrock_messages(messages, per_call_trailing_blocks=1)

    assert formatted[0]["content"][1] == {"cachePoint": {"type": "default", "ttl": "1h"}}


def test_no_cache_point_without_cache_config():
    built = BedrockModel(model_id=_MODEL_ID)
    built.client = MagicMock()
    messages = [{"role": "user", "content": [{"text": "durable"}, {"text": "PER-CALL"}]}]

    formatted = built._format_bedrock_messages(messages, per_call_trailing_blocks=1)

    assert "cachePoint" not in _keys(formatted[0]["content"])
