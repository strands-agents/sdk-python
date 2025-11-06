"""Tests for bidirectional streaming event types.

This module tests JSON serialization for all bidirectional streaming event types.
"""

import base64
import json

import pytest

from strands.experimental.bidirectional_streaming.types.bidirectional_streaming import (
    AudioInputEvent,
    AudioStreamEvent,
    ConnectionCloseEvent,
    ConnectionStartEvent,
    ErrorEvent,
    ImageInputEvent,
    InterruptionEvent,
    ResponseCompleteEvent,
    ResponseStartEvent,
    TextInputEvent,
    TranscriptStreamEvent,
    UsageEvent,
)


@pytest.mark.parametrize(
    "event_class,kwargs,expected_type",
    [
        # Input events
        (TextInputEvent, {"text": "Hello", "role": "user"}, "bidirectional_text_input"),
        (
            AudioInputEvent,
            {
                "audio": base64.b64encode(b"audio").decode("utf-8"),
                "format": "pcm",
                "sample_rate": 16000,
                "channels": 1,
            },
            "bidirectional_audio_input",
        ),
        (
            ImageInputEvent,
            {"image": base64.b64encode(b"image").decode("utf-8"), "mime_type": "image/jpeg"},
            "bidirectional_image_input",
        ),
        # Output events
        (
            ConnectionStartEvent,
            {"connection_id": "c1", "model": "m1", "capabilities": ["audio"]},
            "bidirectional_connection_start",
        ),
        (ResponseStartEvent, {"response_id": "r1"}, "bidirectional_response_start"),
        (
            AudioStreamEvent,
            {
                "audio": base64.b64encode(b"audio").decode("utf-8"),
                "format": "pcm",
                "sample_rate": 24000,
                "channels": 1,
            },
            "bidirectional_audio_stream",
        ),
        (
            TranscriptStreamEvent,
            {"text": "Hello", "role": "assistant", "is_final": True},
            "bidirectional_transcript_stream",
        ),
        (InterruptionEvent, {"reason": "user_speech", "turn_id": None}, "bidirectional_interruption"),
        (
            ResponseCompleteEvent,
            {"response_id": "r1", "stop_reason": "complete"},
            "bidirectional_response_complete",
        ),
        (
            UsageEvent,
            {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            "bidirectional_usage",
        ),
        (
            ConnectionCloseEvent,
            {"connection_id": "c1", "reason": "complete"},
            "bidirectional_connection_close",
        ),
        (ErrorEvent, {"error": ValueError("test"), "details": None}, "bidirectional_error"),
    ],
)
def test_event_json_serialization(event_class, kwargs, expected_type):
    """Test that all event types are JSON serializable and deserializable."""
    # Create event
    event = event_class(**kwargs)

    # Verify type field
    assert event["type"] == expected_type

    # Serialize to JSON
    json_str = json.dumps(event)

    # Deserialize back
    data = json.loads(json_str)

    # Verify type preserved
    assert data["type"] == expected_type

    # Verify all non-private keys preserved
    for key in event.keys():
        if not key.startswith("_"):
            assert key in data
