import logging
import unittest.mock
from enum import IntEnum
from typing import cast

import pytest

import strands
import strands.event_loop
from strands.types._events import ModelStopReason, TypedEvent
from strands.types._usage import USAGE_COUNTERS, accumulate_usage, context_token_count, prompt_token_count
from strands.types.content import Message, Messages
from strands.types.streaming import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    MessageStartEvent,
    MessageStopEvent,
)


class _CountEnum(IntEnum):
    """A vendor may model a count as an enum member, which is an int subclass."""

    TEN = 10


@pytest.fixture(autouse=True)
def moto_autouse(moto_env, moto_mock_aws):
    _ = moto_env
    _ = moto_mock_aws


@pytest.mark.parametrize(
    ("messages", "exp_result"),
    [
        (
            [
                {"role": "assistant", "content": [{"text": "a"}, {"text": " \n"}, {"toolUse": {}}]},
                {"role": "assistant", "content": [{"text": ""}, {"toolUse": {}}]},
                {"role": "assistant", "content": [{"text": "a"}, {"text": " \n"}]},
                {"role": "assistant", "content": []},
                {"role": "assistant"},
                {"role": "user", "content": [{"text": " \n"}]},
            ],
            [
                {"role": "assistant", "content": [{"text": "a"}, {"toolUse": {}}]},
                {"role": "assistant", "content": [{"toolUse": {}}]},
                {"role": "assistant", "content": [{"text": "a"}, {"text": "[blank text]"}]},
                {"role": "assistant", "content": [{"text": "[blank text]"}]},
                {"role": "assistant"},
                {"role": "user", "content": [{"text": " \n"}]},
            ],
        ),
        (
            [],
            [],
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:remove_blank_messages_content_text is deprecated:DeprecationWarning")
def test_remove_blank_messages_content_text(messages, exp_result):
    tru_result = strands.event_loop.streaming.remove_blank_messages_content_text(messages)

    assert tru_result == exp_result


@pytest.mark.parametrize(
    ("messages", "exp_result"),
    [
        pytest.param(
            [
                {"role": "assistant", "content": [{"text": "a"}, {"text": " \n"}, {"toolUse": {"name": "a_name"}}]},
                {"role": "assistant", "content": [{"text": ""}, {"toolUse": {"name": "a_name"}}]},
                {"role": "assistant", "content": [{"text": "a"}, {"text": " \n"}]},
                {"role": "assistant", "content": []},
                {"role": "assistant"},
                {"role": "user", "content": [{"text": " \n"}]},
            ],
            [
                {"role": "assistant", "content": [{"text": "a"}, {"toolUse": {"name": "a_name"}}]},
                {"role": "assistant", "content": [{"toolUse": {"name": "a_name"}}]},
                {"role": "assistant", "content": [{"text": "a"}, {"text": "[blank text]"}]},
                {"role": "assistant", "content": [{"text": "[blank text]"}]},
                {"role": "assistant"},
                {"role": "user", "content": [{"text": " \n"}]},
            ],
            id="blank messages",
        ),
        pytest.param(
            [],
            [],
            id="empty messages",
        ),
        pytest.param(
            [
                {"role": "assistant", "content": [{"toolUse": {"name": "invalid tool"}}]},
            ],
            [
                {"role": "assistant", "content": [{"toolUse": {"name": "INVALID_TOOL_NAME"}}]},
            ],
            id="invalid tool name",
        ),
        pytest.param(
            [
                {"role": "assistant", "content": [{"toolUse": {}}]},
            ],
            [
                {"role": "assistant", "content": [{"toolUse": {"name": "INVALID_TOOL_NAME"}}]},
            ],
            id="missing tool name",
        ),
        pytest.param(
            [
                {"role": "assistant", "content": [{"text": None}, {"toolUse": {"name": "a_name"}}]},
            ],
            [
                {"role": "assistant", "content": [{"toolUse": {"name": "a_name"}}]},
            ],
            id="none text with tool use",
        ),
        pytest.param(
            [
                {"role": "assistant", "content": [{"text": None}]},
            ],
            [
                {"role": "assistant", "content": [{"text": "[blank text]"}]},
            ],
            id="none text without tool use",
        ),
    ],
)
def test_normalize_blank_messages_content_text(messages, exp_result):
    tru_result = strands.event_loop.streaming._normalize_messages(messages)

    assert tru_result == exp_result


def test_handle_message_start():
    event: MessageStartEvent = {"role": "test"}

    tru_message = strands.event_loop.streaming.handle_message_start(event, {})
    exp_message = {"role": "test"}

    assert tru_message == exp_message


@pytest.mark.parametrize(
    ("chunk", "exp_tool_use"),
    [
        ({"start": {}}, {}),
        (
            {"start": {"toolUse": {"toolUseId": "test", "name": "test"}}},
            {"toolUseId": "test", "name": "test", "input": ""},
        ),
        (
            {"start": {"toolUse": {"toolUseId": "test", "name": "test", "reasoningSignature": "YWJj"}}},
            {"toolUseId": "test", "name": "test", "input": "", "reasoningSignature": "YWJj"},
        ),
    ],
)
def test_handle_content_block_start(chunk: ContentBlockStartEvent, exp_tool_use):
    tru_tool_use = strands.event_loop.streaming.handle_content_block_start(chunk)

    assert tru_tool_use == exp_tool_use


@pytest.mark.parametrize(
    ("event", "event_type", "state", "exp_updated_state", "callback_args"),
    [
        # Tool Use - Existing input
        (
            {"delta": {"toolUse": {"input": '"value"}'}}},
            {"type": "tool_use_stream"},
            {"current_tool_use": {"input": '{"key": '}},
            {"current_tool_use": {"input": '{"key": "value"}'}},
            {"current_tool_use": {"input": '{"key": "value"}'}},
        ),
        # Tool Use - New input
        (
            {"delta": {"toolUse": {"input": '{"key": '}}},
            {"type": "tool_use_stream"},
            {"current_tool_use": {}},
            {"current_tool_use": {"input": '{"key": '}},
            {"current_tool_use": {"input": '{"key": '}},
        ),
        # Text
        (
            {"delta": {"text": " world"}},
            {},
            {"text": "hello"},
            {"text": "hello world"},
            {"data": " world"},
        ),
        # Reasoning - Text - Existing
        (
            {"delta": {"reasoningContent": {"text": "king"}}},
            {},
            {"reasoningText": "thin"},
            {"reasoningText": "thinking"},
            {"reasoningText": "king", "reasoning": True},
        ),
        # Reasoning - Text - New
        (
            {"delta": {"reasoningContent": {"text": "thin"}}},
            {},
            {},
            {"reasoningText": "thin"},
            {"reasoningText": "thin", "reasoning": True},
        ),
        # Reasoning - Signature - Existing
        (
            {"delta": {"reasoningContent": {"signature": "ue"}}},
            {},
            {"signature": "val"},
            {"signature": "value"},
            {"reasoning_signature": "ue", "reasoning": True},
        ),
        # Reasoning - Signature - New
        (
            {"delta": {"reasoningContent": {"signature": "val"}}},
            {},
            {},
            {"signature": "val"},
            {"reasoning_signature": "val", "reasoning": True},
        ),
        # Reasoning - redactedContent - New
        pytest.param(
            {"delta": {"reasoningContent": {"redactedContent": b"encoded"}}},
            {},
            {},
            {"redactedContent": b"encoded"},
            {"reasoningRedactedContent": b"encoded", "reasoning": True},
        ),
        # Reasoning - redactedContent - Existing
        pytest.param(
            {"delta": {"reasoningContent": {"redactedContent": b"data"}}},
            {},
            {"redactedContent": b"encoded_"},
            {"redactedContent": b"encoded_data"},
            {"reasoningRedactedContent": b"data", "reasoning": True},
        ),
        # Reasoning - Empty
        (
            {"delta": {"reasoningContent": {}}},
            {},
            {},
            {},
            {},
        ),
        # Citation - New
        (
            {
                "delta": {
                    "citation": {
                        "location": {"documentChar": {"documentIndex": 0, "start": 10, "end": 20}},
                        "title": "Test Doc",
                    }
                }
            },
            {},
            {},
            {
                "citationsContent": [
                    {"location": {"documentChar": {"documentIndex": 0, "start": 10, "end": 20}}, "title": "Test Doc"}
                ]
            },
            {
                "citation": {
                    "location": {"documentChar": {"documentIndex": 0, "start": 10, "end": 20}},
                    "title": "Test Doc",
                }
            },
        ),
        # Citation - Existing
        (
            {
                "delta": {
                    "citation": {
                        "location": {"documentPage": {"documentIndex": 1, "start": 5, "end": 6}},
                        "title": "Another Doc",
                    }
                }
            },
            {},
            {
                "citationsContent": [
                    {"location": {"documentChar": {"documentIndex": 0, "start": 10, "end": 20}}, "title": "Test Doc"}
                ]
            },
            {
                "citationsContent": [
                    {"location": {"documentChar": {"documentIndex": 0, "start": 10, "end": 20}}, "title": "Test Doc"},
                    {"location": {"documentPage": {"documentIndex": 1, "start": 5, "end": 6}}, "title": "Another Doc"},
                ]
            },
            {
                "citation": {
                    "location": {"documentPage": {"documentIndex": 1, "start": 5, "end": 6}},
                    "title": "Another Doc",
                }
            },
        ),
        # Empty
        (
            {"delta": {}},
            {},
            {},
            {},
            {},
        ),
    ],
)
def test_handle_content_block_delta(event: ContentBlockDeltaEvent, event_type, state, exp_updated_state, callback_args):
    exp_callback_event = {**event_type, **callback_args, "delta": event["delta"]} if callback_args else {}

    tru_updated_state, tru_callback_event = strands.event_loop.streaming.handle_content_block_delta(event, state)

    assert tru_updated_state == exp_updated_state
    assert tru_callback_event == exp_callback_event


@pytest.mark.parametrize(
    ("state", "exp_updated_state"),
    [
        # Tool Use - Existing input
        (
            {
                "content": [],
                "current_tool_use": {"toolUseId": "123", "name": "test", "input": '{"key": "value"}'},
                "text": "",
                "reasoningText": "",
                "citationsContent": [],
                "redactedContent": b"",
            },
            {
                "content": [{"toolUse": {"toolUseId": "123", "name": "test", "input": {"key": "value"}}}],
                "current_tool_use": {},
                "text": "",
                "reasoningText": "",
                "citationsContent": [],
                "redactedContent": b"",
            },
        ),
        # Tool Use - With reasoningSignature
        (
            {
                "content": [],
                "current_tool_use": {
                    "toolUseId": "123",
                    "name": "test",
                    "input": '{"key": "value"}',
                    "reasoningSignature": "YWJj",
                },
                "text": "",
                "reasoningText": "",
                "citationsContent": [],
                "redactedContent": b"",
            },
            {
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "123",
                            "name": "test",
                            "input": {"key": "value"},
                            "reasoningSignature": "YWJj",
                        }
                    }
                ],
                "current_tool_use": {},
                "text": "",
                "reasoningText": "",
                "citationsContent": [],
                "redactedContent": b"",
            },
        ),
        # Tool Use - Missing input
        (
            {
                "content": [],
                "current_tool_use": {"toolUseId": "123", "name": "test"},
                "text": "",
                "reasoningText": "",
                "citationsContent": [],
                "redactedContent": b"",
            },
            {
                "content": [{"toolUse": {"toolUseId": "123", "name": "test", "input": {}}}],
                "current_tool_use": {},
                "text": "",
                "reasoningText": "",
                "citationsContent": [],
                "redactedContent": b"",
            },
        ),
        # Text
        (
            {
                "content": [],
                "current_tool_use": {},
                "text": "test",
                "reasoningText": "",
                "citationsContent": [],
                "redactedContent": b"",
            },
            {
                "content": [{"text": "test"}],
                "current_tool_use": {},
                "text": "",
                "reasoningText": "",
                "citationsContent": [],
                "redactedContent": b"",
            },
        ),
        # Text with Citations
        (
            {
                "content": [],
                "current_tool_use": {},
                "text": "This is cited text",
                "reasoningText": "",
                "citationsContent": [
                    {"location": {"documentChar": {"documentIndex": 0, "start": 10, "end": 20}}, "title": "Test Doc"}
                ],
                "redactedContent": b"",
            },
            {
                "content": [
                    {
                        "citationsContent": {
                            "citations": [
                                {
                                    "location": {"documentChar": {"documentIndex": 0, "start": 10, "end": 20}},
                                    "title": "Test Doc",
                                }
                            ],
                            "content": [{"text": "This is cited text"}],
                        }
                    }
                ],
                "current_tool_use": {},
                "text": "",
                "reasoningText": "",
                "citationsContent": [],
                "redactedContent": b"",
            },
        ),
        # Citations without text (should not create content block)
        (
            {
                "content": [],
                "current_tool_use": {},
                "text": "",
                "reasoningText": "",
                "citationsContent": [
                    {"location": {"documentChar": {"documentIndex": 0, "start": 10, "end": 20}}, "title": "Test Doc"}
                ],
                "redactedContent": b"",
            },
            {
                "content": [],
                "current_tool_use": {},
                "text": "",
                "reasoningText": "",
                "citationsContent": [
                    {"location": {"documentChar": {"documentIndex": 0, "start": 10, "end": 20}}, "title": "Test Doc"}
                ],
                "redactedContent": b"",
            },
        ),
        # Reasoning
        (
            {
                "content": [],
                "current_tool_use": {},
                "text": "",
                "reasoningText": "test",
                "signature": "123",
                "citationsContent": [],
                "redactedContent": b"",
            },
            {
                "content": [{"reasoningContent": {"reasoningText": {"text": "test", "signature": "123"}}}],
                "current_tool_use": {},
                "text": "",
                "reasoningText": "",
                "citationsContent": [],
                "redactedContent": b"",
            },
        ),
        # Reasoning without signature
        (
            {
                "content": [],
                "current_tool_use": {},
                "text": "",
                "reasoningText": "test",
                "citationsContent": [],
                "redactedContent": b"",
            },
            {
                "content": [{"reasoningContent": {"reasoningText": {"text": "test"}}}],
                "current_tool_use": {},
                "text": "",
                "reasoningText": "",
                "citationsContent": [],
                "redactedContent": b"",
            },
        ),
        # Reasoning with empty text but non-empty signature
        (
            {
                "content": [],
                "current_tool_use": {},
                "text": "",
                "reasoningText": "",
                "signature": "123",
                "citationsContent": [],
                "redactedContent": b"",
            },
            {
                "content": [{"reasoningContent": {"reasoningText": {"text": "", "signature": "123"}}}],
                "current_tool_use": {},
                "text": "",
                "reasoningText": "",
                "citationsContent": [],
                "redactedContent": b"",
            },
        ),
        # redactedContent
        (
            {
                "content": [],
                "current_tool_use": {},
                "text": "",
                "reasoningText": "",
                "redactedContent": b"encoded_data",
                "citationsContent": [],
            },
            {
                "content": [{"reasoningContent": {"redactedContent": b"encoded_data"}}],
                "current_tool_use": {},
                "text": "",
                "reasoningText": "",
                "redactedContent": b"",
                "citationsContent": [],
            },
        ),
        # Empty
        (
            {
                "content": [],
                "current_tool_use": {},
                "text": "",
                "reasoningText": "",
                "citationsContent": [],
                "redactedContent": b"",
            },
            {
                "content": [],
                "current_tool_use": {},
                "text": "",
                "reasoningText": "",
                "citationsContent": [],
                "redactedContent": b"",
            },
        ),
    ],
)
def test_handle_content_block_stop(state, exp_updated_state):
    tru_updated_state = strands.event_loop.streaming.handle_content_block_stop(state)

    assert tru_updated_state == exp_updated_state


def test_handle_message_stop():
    event: MessageStopEvent = {"stopReason": "end_turn"}

    tru_reason = strands.event_loop.streaming.handle_message_stop(event, [])
    exp_reason = "end_turn"

    assert tru_reason == exp_reason


def test_handle_message_stop_overrides_end_turn_when_tool_use_present():
    event: MessageStopEvent = {"stopReason": "end_turn"}
    content = [{"toolUse": {"toolUseId": "t1", "name": "myTool", "input": {}}}]

    tru_reason = strands.event_loop.streaming.handle_message_stop(event, content)

    assert tru_reason == "tool_use"


def test_handle_message_stop_keeps_tool_use_unchanged():
    event: MessageStopEvent = {"stopReason": "tool_use"}
    content = [{"toolUse": {"toolUseId": "t1", "name": "myTool", "input": {}}}]

    tru_reason = strands.event_loop.streaming.handle_message_stop(event, content)

    assert tru_reason == "tool_use"


def test_extract_usage_metrics():
    event = {
        "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
        "metrics": {"latencyMs": 0},
    }

    tru_usage, tru_metrics = strands.event_loop.streaming.extract_usage_metrics(event)
    exp_usage, exp_metrics = event["usage"], event["metrics"]

    assert tru_usage == exp_usage and tru_metrics == exp_metrics


def test_extract_usage_metrics_with_cache_tokens():
    event = {
        "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0, "cacheReadInputTokens": 0},
        "metrics": {"latencyMs": 0},
    }

    tru_usage, tru_metrics = strands.event_loop.streaming.extract_usage_metrics(event)
    exp_usage, exp_metrics = event["usage"], event["metrics"]

    assert tru_usage == exp_usage and tru_metrics == exp_metrics


def test_extract_usage_metrics_without_metrics():
    """Test extract_usage_metrics when metrics field is missing."""
    event = {
        "usage": {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7},
    }

    tru_usage, tru_metrics = strands.event_loop.streaming.extract_usage_metrics(event)
    exp_usage = {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7}
    exp_metrics = {"latencyMs": 0}

    assert tru_usage == exp_usage and tru_metrics == exp_metrics


def test_extract_usage_metrics_without_usage():
    """Test extract_usage_metrics when usage field is missing."""
    event = {
        "metrics": {"latencyMs": 100},
    }

    tru_usage, tru_metrics = strands.event_loop.streaming.extract_usage_metrics(event)
    exp_usage = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
    exp_metrics = {"latencyMs": 100}

    assert tru_usage == exp_usage and tru_metrics == exp_metrics


def test_extract_usage_metrics_empty_metadata():
    """Test extract_usage_metrics when both fields are missing."""
    event = {}

    tru_usage, tru_metrics = strands.event_loop.streaming.extract_usage_metrics(event)
    exp_usage = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
    exp_metrics = {"latencyMs": 0}

    assert tru_usage == exp_usage and tru_metrics == exp_metrics


def test_extract_usage_metrics_reads_the_event_and_each_counter_once(caplog):
    """Reading either the event or a counter twice would assemble a usage out of separate answers.

    Neither accessor need answer with the same value twice, so the coercion and the usability check
    read from one snapshot. Reading twice reports the difference between the two answers as a count
    the model failed to make usable, and reading the event twice can compose one usage out of
    several, satisfying no invariant.
    """

    class CountingUsage(dict):
        def __init__(self):
            super().__init__(inputTokens=10, outputTokens=5, totalTokens=15)
            self.reads = {}

        def get(self, key, default=None):
            self.reads[key] = self.reads.get(key, 0) + 1
            return super().get(key, default)

    class CountingEvent(dict):
        def __init__(self, usage):
            super().__init__(metrics={"latencyMs": 100})
            self.usage = usage
            self.reads = 0

        def get(self, key, default=None):
            if key == "usage":
                self.reads += 1
                return self.usage
            return super().get(key, default)

    reported = CountingUsage()
    event = CountingEvent(reported)

    tru_usage, _ = strands.event_loop.streaming.extract_usage_metrics(event)

    assert event.reads == 1
    assert reported.reads == dict.fromkeys(USAGE_COUNTERS, 1)
    assert tru_usage == {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}
    assert "not usable numbers" not in caplog.text


@pytest.mark.parametrize(
    ("response", "exp_events"),
    [
        # Standard Message
        (
            [
                {"messageStart": {"role": "assistant"}},
                {
                    "contentBlockStart": {"start": {"toolUse": {"toolUseId": "123", "name": "test"}}},
                },
                {
                    "contentBlockDelta": {"delta": {"toolUse": {"input": '{"key": "value"}'}}},
                },
                {"contentBlockStop": {}},
                {
                    "messageStop": {"stopReason": "tool_use"},
                },
                {
                    "metadata": {
                        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 1},
                        "metrics": {"latencyMs": 1},
                    }
                },
            ],
            [
                {
                    "event": {
                        "messageStart": {
                            "role": "assistant",
                        },
                    },
                },
                {
                    "event": {
                        "contentBlockStart": {
                            "start": {
                                "toolUse": {
                                    "name": "test",
                                    "toolUseId": "123",
                                },
                            },
                        },
                    },
                },
                {
                    "event": {
                        "contentBlockDelta": {
                            "delta": {
                                "toolUse": {
                                    "input": '{"key": "value"}',
                                },
                            },
                        },
                    },
                },
                {
                    "current_tool_use": {
                        "input": {
                            "key": "value",
                        },
                        "name": "test",
                        "toolUseId": "123",
                    },
                    "delta": {
                        "toolUse": {
                            "input": '{"key": "value"}',
                        },
                    },
                    "type": "tool_use_stream",
                },
                {
                    "event": {
                        "contentBlockStop": {},
                    },
                },
                {
                    "event": {
                        "messageStop": {
                            "stopReason": "tool_use",
                        },
                    },
                },
                {
                    "event": {
                        "metadata": {
                            "metrics": {
                                "latencyMs": 1,
                            },
                            "usage": {
                                "inputTokens": 1,
                                "outputTokens": 1,
                                "totalTokens": 1,
                            },
                        },
                    },
                },
                {
                    "stop": (
                        "tool_use",
                        {
                            "role": "assistant",
                            "content": [{"toolUse": {"toolUseId": "123", "name": "test", "input": {"key": "value"}}}],
                        },
                        {"inputTokens": 1, "outputTokens": 1, "totalTokens": 1},
                        {"latencyMs": 1},
                    )
                },
            ],
        ),
        # Message with Citations
        (
            [
                {"messageStart": {"role": "assistant"}},
                {"contentBlockStart": {"start": {}}},
                {"contentBlockDelta": {"delta": {"text": "This is cited text"}}},
                {
                    "contentBlockDelta": {
                        "delta": {
                            "citation": {
                                "location": {"documentChar": {"documentIndex": 0, "start": 10, "end": 20}},
                                "title": "Test Doc",
                            }
                        }
                    }
                },
                {
                    "contentBlockDelta": {
                        "delta": {
                            "citation": {
                                "location": {"documentPage": {"documentIndex": 1, "start": 5, "end": 6}},
                                "title": "Another Doc",
                            }
                        }
                    }
                },
                {"contentBlockStop": {}},
                {"messageStop": {"stopReason": "end_turn"}},
                {
                    "metadata": {
                        "usage": {"inputTokens": 5, "outputTokens": 10, "totalTokens": 15},
                        "metrics": {"latencyMs": 100},
                    }
                },
            ],
            [
                {"event": {"messageStart": {"role": "assistant"}}},
                {"event": {"contentBlockStart": {"start": {}}}},
                {"event": {"contentBlockDelta": {"delta": {"text": "This is cited text"}}}},
                {"data": "This is cited text", "delta": {"text": "This is cited text"}},
                {
                    "event": {
                        "contentBlockDelta": {
                            "delta": {
                                "citation": {
                                    "location": {"documentChar": {"documentIndex": 0, "start": 10, "end": 20}},
                                    "title": "Test Doc",
                                }
                            }
                        }
                    }
                },
                {
                    "citation": {
                        "location": {"documentChar": {"documentIndex": 0, "start": 10, "end": 20}},
                        "title": "Test Doc",
                    },
                    "delta": {
                        "citation": {
                            "location": {"documentChar": {"documentIndex": 0, "start": 10, "end": 20}},
                            "title": "Test Doc",
                        }
                    },
                },
                {
                    "event": {
                        "contentBlockDelta": {
                            "delta": {
                                "citation": {
                                    "location": {"documentPage": {"documentIndex": 1, "start": 5, "end": 6}},
                                    "title": "Another Doc",
                                }
                            }
                        }
                    }
                },
                {
                    "citation": {
                        "location": {"documentPage": {"documentIndex": 1, "start": 5, "end": 6}},
                        "title": "Another Doc",
                    },
                    "delta": {
                        "citation": {
                            "location": {"documentPage": {"documentIndex": 1, "start": 5, "end": 6}},
                            "title": "Another Doc",
                        }
                    },
                },
                {"event": {"contentBlockStop": {}}},
                {"event": {"messageStop": {"stopReason": "end_turn"}}},
                {
                    "event": {
                        "metadata": {
                            "usage": {"inputTokens": 5, "outputTokens": 10, "totalTokens": 15},
                            "metrics": {"latencyMs": 100},
                        }
                    }
                },
                {
                    "stop": (
                        "end_turn",
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "citationsContent": {
                                        "citations": [
                                            {
                                                "location": {
                                                    "documentChar": {"documentIndex": 0, "start": 10, "end": 20}
                                                },
                                                "title": "Test Doc",
                                            },
                                            {
                                                "location": {
                                                    "documentPage": {"documentIndex": 1, "start": 5, "end": 6}
                                                },
                                                "title": "Another Doc",
                                            },
                                        ],
                                        "content": [{"text": "This is cited text"}],
                                    }
                                }
                            ],
                        },
                        {"inputTokens": 5, "outputTokens": 10, "totalTokens": 15},
                        {"latencyMs": 100},
                    )
                },
            ],
        ),
        # Empty Message
        (
            [{}],
            [
                {
                    "event": {},
                },
                {
                    "stop": (
                        "end_turn",
                        {
                            "role": "assistant",
                            "content": [],
                        },
                        {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                        {"latencyMs": 0, "timeToFirstByteMs": 0},
                    ),
                },
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_process_stream(response, exp_events, agenerator, alist):
    stream = strands.event_loop.streaming.process_stream(agenerator(response))

    tru_events = await alist(stream)
    assert tru_events == exp_events

    # Ensure that we're getting typed events coming out of process_stream
    non_typed_events = [event for event in tru_events if not isinstance(event, TypedEvent)]
    assert non_typed_events == []


@pytest.mark.parametrize(
    ("response", "exp_events"),
    [
        # Redacted Message
        (
            [
                {"messageStart": {"role": "assistant"}},
                {
                    "contentBlockStart": {"start": {}},
                },
                {
                    "contentBlockDelta": {"delta": {"text": "Hello!"}},
                },
                {"contentBlockStop": {}},
                {
                    "messageStop": {"stopReason": "guardrail_intervened"},
                },
                {
                    "redactContent": {
                        "redactUserContentMessage": "REDACTED",
                        "redactAssistantContentMessage": "REDACTED.",
                    }
                },
                {
                    "metadata": {
                        "usage": {
                            "inputTokens": 1,
                            "outputTokens": 1,
                            "totalTokens": 1,
                        },
                        "metrics": {"latencyMs": 1},
                    }
                },
            ],
            [
                {"event": {"messageStart": {"role": "assistant"}}},
                {"event": {"contentBlockStart": {"start": {}}}},
                {"event": {"contentBlockDelta": {"delta": {"text": "Hello!"}}}},
                {"data": "Hello!", "delta": {"text": "Hello!"}},
                {"event": {"contentBlockStop": {}}},
                {"event": {"messageStop": {"stopReason": "guardrail_intervened"}}},
                {
                    "event": {
                        "redactContent": {
                            "redactUserContentMessage": "REDACTED",
                            "redactAssistantContentMessage": "REDACTED.",
                        }
                    }
                },
                {
                    "event": {
                        "metadata": {
                            "usage": {
                                "inputTokens": 1,
                                "outputTokens": 1,
                                "totalTokens": 1,
                            },
                            "metrics": {"latencyMs": 1},
                        }
                    }
                },
                {
                    "stop": (
                        "guardrail_intervened",
                        {"role": "assistant", "content": [{"text": "REDACTED."}]},
                        {"inputTokens": 1, "outputTokens": 1, "totalTokens": 1},
                        {"latencyMs": 1},
                    )
                },
            ],
        ),
        (
            [
                {"messageStart": {"role": "assistant"}},
                {
                    "contentBlockStart": {"start": {}},
                },
                {
                    "contentBlockDelta": {"delta": {"reasoningContent": {"redactedContent": b"encoded_data"}}},
                },
                {"contentBlockStop": {}},
                {
                    "messageStop": {"stopReason": "end_turn"},
                },
                {
                    "metadata": {
                        "usage": {
                            "inputTokens": 1,
                            "outputTokens": 1,
                            "totalTokens": 1,
                        },
                        "metrics": {"latencyMs": 1},
                    }
                },
            ],
            [
                {"event": {"messageStart": {"role": "assistant"}}},
                {"event": {"contentBlockStart": {"start": {}}}},
                {"event": {"contentBlockDelta": {"delta": {"reasoningContent": {"redactedContent": b"encoded_data"}}}}},
                {
                    "reasoningRedactedContent": b"encoded_data",
                    "delta": {"reasoningContent": {"redactedContent": b"encoded_data"}},
                    "reasoning": True,
                },
                {"event": {"contentBlockStop": {}}},
                {"event": {"messageStop": {"stopReason": "end_turn"}}},
                {
                    "event": {
                        "metadata": {
                            "usage": {
                                "inputTokens": 1,
                                "outputTokens": 1,
                                "totalTokens": 1,
                            },
                            "metrics": {"latencyMs": 1},
                        }
                    }
                },
                {
                    "stop": (
                        "end_turn",
                        {
                            "role": "assistant",
                            "content": [{"reasoningContent": {"redactedContent": b"encoded_data"}}],
                        },
                        {"inputTokens": 1, "outputTokens": 1, "totalTokens": 1},
                        {"latencyMs": 1},
                    )
                },
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_process_stream_redacted(response, exp_events, agenerator, alist):
    stream = strands.event_loop.streaming.process_stream(agenerator(response))

    tru_events = await alist(stream)
    assert tru_events == exp_events

    # Ensure that we're getting typed events coming out of process_stream
    non_typed_events = [event for event in tru_events if not isinstance(event, TypedEvent)]
    assert non_typed_events == []


def _get_message_from_event(event: ModelStopReason) -> Message:
    return cast(Message, event["stop"][1])


@pytest.mark.asyncio
async def test_process_stream_with_no_signature(agenerator, alist):
    response = [
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockDelta": {
                "delta": {"reasoningContent": {"text": 'User asks: "Reason about 2+2" so I will do that'}},
                "contentBlockIndex": 0,
            }
        },
        {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "."}}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {
            "contentBlockDelta": {
                "delta": {"text": "Sure! Let’s do it"},
                "contentBlockIndex": 1,
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "end_turn"}},
        {
            "metadata": {
                "usage": {"inputTokens": 112, "outputTokens": 764, "totalTokens": 876},
                "metrics": {"latencyMs": 2970},
            }
        },
    ]

    stream = strands.event_loop.streaming.process_stream(agenerator(response))

    last_event = cast(ModelStopReason, (await alist(stream))[-1])

    message = _get_message_from_event(last_event)

    assert "signature" not in message["content"][0]["reasoningContent"]["reasoningText"]
    assert message["content"][1]["text"] == "Sure! Let’s do it"


@pytest.mark.asyncio
async def test_process_stream_with_signature(agenerator, alist):
    response = [
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockDelta": {
                "delta": {"reasoningContent": {"text": 'User asks: "Reason about 2+2" so I will do that'}},
                "contentBlockIndex": 0,
            }
        },
        {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "."}}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"signature": "test-"}}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"signature": "signature"}}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {
            "contentBlockDelta": {
                "delta": {"text": "Sure! Let’s do it"},
                "contentBlockIndex": 1,
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "end_turn"}},
        {
            "metadata": {
                "usage": {"inputTokens": 112, "outputTokens": 764, "totalTokens": 876},
                "metrics": {"latencyMs": 2970},
            }
        },
    ]

    stream = strands.event_loop.streaming.process_stream(agenerator(response))

    last_event = cast(ModelStopReason, (await alist(stream))[-1])

    message = _get_message_from_event(last_event)

    assert message["content"][0]["reasoningContent"]["reasoningText"]["signature"] == "test-signature"
    assert message["content"][1]["text"] == "Sure! Let’s do it"


@pytest.mark.asyncio
async def test_process_stream_with_multiple_signed_reasoning_blocks(agenerator, alist):
    """Each signed reasoning block keeps only its own signature.

    Guards against https://github.com/strands-agents/harness-sdk/issues/3425: the signature accumulator
    was never cleared when a reasoning block finalized, so later blocks carried a cumulative concatenation
    ("SIG1", "SIG1SIG2", ...) that providers reject as a modified thinking signature on the next request.
    """
    response = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "first"}}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"signature": "SIG1"}}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "second"}}, "contentBlockIndex": 1}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"signature": "SIG2"}}, "contentBlockIndex": 1}},
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]

    stream = strands.event_loop.streaming.process_stream(agenerator(response))

    last_event = cast(ModelStopReason, (await alist(stream))[-1])

    message = _get_message_from_event(last_event)

    tru_signatures = [block["reasoningContent"]["reasoningText"]["signature"] for block in message["content"]]
    exp_signatures = ["SIG1", "SIG2"]
    assert tru_signatures == exp_signatures


@pytest.mark.asyncio
async def test_process_stream_signature_does_not_leak_into_later_empty_block(agenerator, alist):
    """A signed reasoning block does not turn a later empty content block into a spurious reasoning block.

    Guards against the secondary effect of https://github.com/strands-agents/harness-sdk/issues/3425: a stale
    signature left in state kept the ``"signature" in state`` guard true, so an otherwise-empty trailing block
    was materialized as a reasoning block carrying the leaked signature.
    """
    response = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "thinking"}}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"signature": "SIG1"}}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"contentBlockStart": {"start": {}, "contentBlockIndex": 1}},
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]

    stream = strands.event_loop.streaming.process_stream(agenerator(response))

    last_event = cast(ModelStopReason, (await alist(stream))[-1])

    message = _get_message_from_event(last_event)

    tru_content = message["content"]
    exp_content = [{"reasoningContent": {"reasoningText": {"text": "thinking", "signature": "SIG1"}}}]
    assert tru_content == exp_content


@pytest.mark.asyncio
async def test_stream_messages(agenerator, alist):
    mock_model = unittest.mock.MagicMock()
    mock_model.stream.return_value = agenerator(
        [
            {"contentBlockDelta": {"delta": {"text": "test"}}},
            {"contentBlockStop": {}},
        ]
    )

    stream = strands.event_loop.streaming.stream_messages(
        mock_model,
        system_prompt_content=[{"text": "test prompt"}],
        messages=[{"role": "assistant", "content": [{"text": "a"}, {"text": " \n"}]}],
        tool_specs=None,
        system_prompt="test prompt",
    )

    tru_events = await alist(stream)
    exp_events = [
        {
            "event": {
                "contentBlockDelta": {
                    "delta": {
                        "text": "test",
                    },
                },
            },
        },
        {
            "data": "test",
            "delta": {
                "text": "test",
            },
        },
        {
            "event": {
                "contentBlockStop": {},
            },
        },
        {
            "stop": (
                "end_turn",
                {"role": "assistant", "content": [{"text": "test"}]},
                {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                {"latencyMs": 0, "timeToFirstByteMs": 0},
            )
        },
    ]
    assert tru_events == exp_events

    mock_model.stream.assert_called_with(
        [{"role": "assistant", "content": [{"text": "a"}, {"text": "[blank text]"}]}],
        None,
        "test prompt",
        tool_choice=None,
        system_prompt_content=[{"text": "test prompt"}],
        invocation_state=None,
        model_state=None,
    )


@pytest.mark.asyncio
async def test_stream_messages_strips_id_before_model_call(agenerator, alist):
    """The durable message id must never leak to the model provider."""
    mock_model = unittest.mock.MagicMock()
    mock_model.stream.return_value = agenerator(
        [
            {"contentBlockDelta": {"delta": {"text": "test"}}},
            {"contentBlockStop": {}},
        ]
    )

    stream = strands.event_loop.streaming.stream_messages(
        mock_model,
        system_prompt_content=None,
        messages=[
            {
                "role": "user",
                "content": [{"text": "hi"}],
                "tracking_id": "durable-1",
                "metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}},
            }
        ],
        tool_specs=None,
        system_prompt=None,
    )

    await alist(stream)

    sent_messages = mock_model.stream.call_args[0][0]
    assert sent_messages == [{"role": "user", "content": [{"text": "hi"}]}]
    assert "tracking_id" not in sent_messages[0]
    assert "metadata" not in sent_messages[0]


@pytest.mark.asyncio
async def test_stream_messages_with_system_prompt_content(agenerator, alist):
    """Test stream_messages with SystemContentBlock input."""
    mock_model = unittest.mock.MagicMock()
    mock_model.stream.return_value = agenerator(
        [
            {"contentBlockDelta": {"delta": {"text": "test"}}},
            {"contentBlockStop": {}},
        ]
    )

    system_prompt_content = [{"text": "You are a helpful assistant."}, {"cachePoint": {"type": "default"}}]

    stream = strands.event_loop.streaming.stream_messages(
        mock_model,
        system_prompt_content=system_prompt_content,
        messages=[{"role": "user", "content": [{"text": "Hello"}]}],
        tool_specs=[],
        system_prompt=None,
    )

    await alist(stream)

    # Verify model.stream was called with both parameters
    mock_model.stream.assert_called_with(
        [{"role": "user", "content": [{"text": "Hello"}]}],
        None,
        None,
        tool_choice=None,
        system_prompt_content=system_prompt_content,
        invocation_state=None,
        model_state=None,
    )


@pytest.mark.asyncio
async def test_stream_messages_single_text_block_backwards_compatibility(agenerator, alist):
    """Test that single text block extracts system_prompt for backwards compatibility."""
    mock_model = unittest.mock.MagicMock()
    mock_model.stream.return_value = agenerator(
        [
            {"contentBlockDelta": {"delta": {"text": "test"}}},
            {"contentBlockStop": {}},
        ]
    )

    system_prompt_content = [{"text": "You are a helpful assistant."}]

    stream = strands.event_loop.streaming.stream_messages(
        mock_model,
        system_prompt_content=system_prompt_content,
        messages=[{"role": "user", "content": [{"text": "Hello"}]}],
        tool_specs=[],
        system_prompt="You are a helpful assistant.",
    )

    await alist(stream)

    # Verify model.stream was called with extracted system_prompt for backwards compatibility
    mock_model.stream.assert_called_with(
        [{"role": "user", "content": [{"text": "Hello"}]}],
        None,
        "You are a helpful assistant.",
        tool_choice=None,
        system_prompt_content=system_prompt_content,
        invocation_state=None,
        model_state=None,
    )


@pytest.mark.asyncio
async def test_stream_messages_empty_system_prompt_content(agenerator, alist):
    """Test stream_messages with empty system_prompt_content."""
    mock_model = unittest.mock.MagicMock()
    mock_model.stream.return_value = agenerator(
        [
            {"contentBlockDelta": {"delta": {"text": "test"}}},
            {"contentBlockStop": {}},
        ]
    )

    stream = strands.event_loop.streaming.stream_messages(
        mock_model,
        messages=[{"role": "user", "content": [{"text": "Hello"}]}],
        tool_specs=[],
        system_prompt=None,
        system_prompt_content=[],
    )

    await alist(stream)

    # Verify model.stream was called with None system_prompt
    mock_model.stream.assert_called_with(
        [{"role": "user", "content": [{"text": "Hello"}]}],
        None,
        None,
        tool_choice=None,
        system_prompt_content=[],
        invocation_state=None,
        model_state=None,
    )


@pytest.mark.asyncio
async def test_stream_messages_none_system_prompt_content(agenerator, alist):
    """Test stream_messages with None system_prompt_content."""
    mock_model = unittest.mock.MagicMock()
    mock_model.stream.return_value = agenerator(
        [
            {"contentBlockDelta": {"delta": {"text": "test"}}},
            {"contentBlockStop": {}},
        ]
    )

    stream = strands.event_loop.streaming.stream_messages(
        mock_model,
        system_prompt_content=None,
        messages=[{"role": "user", "content": [{"text": "Hello"}]}],
        tool_specs=None,
        system_prompt=None,
    )

    tru_events = await alist(stream)

    # Verify model.stream was called with None system_prompt and empty lists
    mock_model.stream.assert_called_with(
        [{"role": "user", "content": [{"text": "Hello"}]}],
        None,
        None,
        tool_choice=None,
        system_prompt_content=None,
        invocation_state=None,
        model_state=None,
    )

    # Ensure that we're getting typed events coming out of process_stream
    non_typed_events = [event for event in tru_events if not isinstance(event, TypedEvent)]
    assert non_typed_events == []


@pytest.mark.asyncio
async def test_stream_messages_normalizes_messages(agenerator, alist):
    mock_model = unittest.mock.MagicMock()
    mock_model.stream.return_value = agenerator(
        [
            {"contentBlockDelta": {"delta": {"text": "test"}}},
            {"contentBlockStop": {}},
        ]
    )

    messages: Messages = [
        # blank text
        {"role": "assistant", "content": [{"text": "a"}, {"text": " \n"}, {"toolUse": {"name": "a_name"}}]},
        {"role": "assistant", "content": [{"text": ""}, {"toolUse": {"name": "a_name"}}]},
        {"role": "assistant", "content": [{"text": "a"}, {"text": " \n"}]},
        # Invalid names
        {"role": "assistant", "content": [{"toolUse": {"name": "invalid name"}}]},
        {"role": "assistant", "content": [{"toolUse": {}}]},
    ]

    await alist(
        strands.event_loop.streaming.stream_messages(
            mock_model,
            system_prompt_content=[{"text": "test prompt"}],
            messages=messages,
            tool_specs=None,
            system_prompt="test prompt",
        )
    )

    assert mock_model.stream.call_args[0][0] == [
        # blank text
        {"content": [{"text": "a"}, {"toolUse": {"name": "a_name"}}], "role": "assistant"},
        {"content": [{"toolUse": {"name": "a_name"}}], "role": "assistant"},
        {"content": [{"text": "a"}, {"text": "[blank text]"}], "role": "assistant"},
        # Invalid names
        {"content": [{"toolUse": {"name": "INVALID_TOOL_NAME"}}], "role": "assistant"},
        {"content": [{"toolUse": {"name": "INVALID_TOOL_NAME"}}], "role": "assistant"},
    ]


@pytest.mark.asyncio
async def test_process_stream_overrides_end_turn_when_tool_use_present(agenerator, alist):
    response = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"contentBlockIndex": 0, "start": {"toolUse": {"toolUseId": "t1", "name": "myTool"}}}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"key": "val"}'}}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {
            "metadata": {
                "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
                "metrics": {"latencyMs": 100},
            }
        },
    ]

    stream = strands.event_loop.streaming.process_stream(agenerator(response))
    last_event = cast(ModelStopReason, (await alist(stream))[-1])

    assert last_event["stop"][0] == "tool_use"


@pytest.mark.asyncio
async def test_process_stream_keeps_end_turn_when_no_tool_use(agenerator, alist):
    response = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Hello!"}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {
            "metadata": {
                "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
                "metrics": {"latencyMs": 100},
            }
        },
    ]

    stream = strands.event_loop.streaming.process_stream(agenerator(response))
    last_event = cast(ModelStopReason, (await alist(stream))[-1])

    assert last_event["stop"][0] == "end_turn"


@pytest.mark.asyncio
async def test_process_stream_keeps_tool_use_stop_reason_unchanged(agenerator, alist):
    response = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"contentBlockIndex": 0, "start": {"toolUse": {"toolUseId": "t1", "name": "myTool"}}}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": "{}"}}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "tool_use"}},
        {
            "metadata": {
                "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
                "metrics": {"latencyMs": 100},
            }
        },
    ]

    stream = strands.event_loop.streaming.process_stream(agenerator(response))
    last_event = cast(ModelStopReason, (await alist(stream))[-1])

    assert last_event["stop"][0] == "tool_use"


def test_handle_content_block_delta_captures_tool_use_id_and_name_from_delta():
    """Delta events that include toolUseId and name should populate current_tool_use."""
    event = {"delta": {"toolUse": {"input": '{"x": 1}', "toolUseId": "abc123", "name": "output_slide"}}}
    state = {"current_tool_use": {}}

    updated_state, _ = strands.event_loop.streaming.handle_content_block_delta(event, state)

    assert updated_state["current_tool_use"] == {
        "toolUseId": "abc123",
        "name": "output_slide",
        "input": '{"x": 1}',
    }


def test_handle_content_block_delta_does_not_override_existing_tool_use_id_and_name():
    """toolUseId and name from contentBlockStart should not be overridden by a later delta."""
    event = {"delta": {"toolUse": {"input": '{"x": 1}', "toolUseId": "from_delta", "name": "from_delta"}}}
    state = {"current_tool_use": {"toolUseId": "from_start", "name": "from_start", "input": ""}}

    updated_state, _ = strands.event_loop.streaming.handle_content_block_delta(event, state)

    assert updated_state["current_tool_use"] == {
        "toolUseId": "from_start",
        "name": "from_start",
        "input": '{"x": 1}',
    }


def test_handle_content_block_delta_tool_use_without_input_key():
    """A toolUse delta missing the input key should not raise KeyError."""
    event = {"delta": {"toolUse": {}}}
    state = {"current_tool_use": {"toolUseId": "t1", "name": "tool"}}

    updated_state, _ = strands.event_loop.streaming.handle_content_block_delta(event, state)

    assert updated_state["current_tool_use"]["input"] == ""


def test_handle_content_block_stop_skips_incomplete_tool_use_missing_id(caplog):
    """A tool use block missing toolUseId is skipped with a warning."""
    import logging

    state = {
        "content": [],
        "current_tool_use": {"name": "output_slide", "input": '{"x": 1}'},
        "text": "",
        "reasoningText": "",
        "citationsContent": [],
    }

    with caplog.at_level(logging.WARNING, logger="strands.event_loop.streaming"):
        updated_state = strands.event_loop.streaming.handle_content_block_stop(state)

    assert updated_state["content"] == []
    assert updated_state["current_tool_use"] == {}
    assert "incomplete tool use block" in caplog.text


def test_handle_content_block_stop_skips_incomplete_tool_use_missing_name(caplog):
    """A tool use block missing name is skipped with a warning."""
    import logging

    state = {
        "content": [],
        "current_tool_use": {"toolUseId": "abc123", "input": '{"x": 1}'},
        "text": "",
        "reasoningText": "",
        "citationsContent": [],
    }

    with caplog.at_level(logging.WARNING, logger="strands.event_loop.streaming"):
        updated_state = strands.event_loop.streaming.handle_content_block_stop(state)

    assert updated_state["content"] == []
    assert updated_state["current_tool_use"] == {}
    assert "incomplete tool use block" in caplog.text


@pytest.mark.asyncio
async def test_process_stream_tool_use_info_in_delta(agenerator, alist):
    """Models that provide toolUseId and name in contentBlockDelta (not contentBlockStart) work correctly."""
    response = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {
            "contentBlockDelta": {
                "delta": {"toolUse": {"input": '{"title": "Test"}', "toolUseId": "xyz789", "name": "output_slide"}}
            }
        },
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "tool_use"}},
        {
            "metadata": {
                "usage": {"inputTokens": 5, "outputTokens": 10, "totalTokens": 15},
                "metrics": {"latencyMs": 50},
            }
        },
    ]

    stream = strands.event_loop.streaming.process_stream(agenerator(response))
    events = await alist(stream)
    last_event = cast(ModelStopReason, events[-1])

    stop_reason, message, _, _ = last_event["stop"]
    assert stop_reason == "tool_use"
    assert len(message["content"]) == 1
    tool_use = message["content"][0]["toolUse"]
    assert tool_use == {"toolUseId": "xyz789", "name": "output_slide", "input": {"title": "Test"}}


@pytest.mark.parametrize(
    "usage",
    [
        {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
        # Cache counters that account for the total, as every built-in provider reports them.
        {"inputTokens": 10, "outputTokens": 4, "totalTokens": 5862, "cacheReadInputTokens": 5848},
        {"inputTokens": 2, "outputTokens": 5, "totalTokens": 6457, "cacheWriteInputTokens": 6450},
    ],
)
def test_extract_usage_metrics_accepts_conforming_usage_without_warning(usage, caplog):
    caplog.set_level(logging.WARNING, logger="strands.event_loop.streaming")

    strands.event_loop.streaming.extract_usage_metrics({"usage": usage, "metrics": {"latencyMs": 1}})

    assert caplog.text == ""


def test_extract_usage_metrics_warns_when_counters_do_not_account_for_the_total(caplog):
    """A custom model reporting cache tokens inside inputTokens would inflate cost silently."""
    caplog.set_level(logging.WARNING, logger="strands.event_loop.streaming")
    usage = {"inputTokens": 6452, "outputTokens": 5, "totalTokens": 6457, "cacheReadInputTokens": 6450}

    strands.event_loop.streaming.extract_usage_metrics({"usage": usage, "metrics": {"latencyMs": 1}})

    assert "counted_tokens=<12907>, total_tokens=<6457>" in caplog.text


@pytest.mark.parametrize(
    "usage",
    [
        # A gateway reporting counts as strings, a provider omitting a counter, and a provider
        # sending null for one: the check reports on malformed usage rather than raising out of
        # the stream, matching how a provider's own counts are coerced.
        {"inputTokens": "10", "outputTokens": "5", "totalTokens": "15"},
        {"inputTokens": None, "outputTokens": 5, "totalTokens": 5},
        {"inputTokens": 12, "outputTokens": 4, "totalTokens": 16, "cacheReadInputTokens": None},
        {"inputTokens": 12, "outputTokens": 4},
        {"inputTokens": 12, "outputTokens": 4, "totalTokens": None},
        # A model that reports no usage at all commonly yields the field as null rather than
        # omitting it, and nothing constrains what it puts there.
        None,
        "usage",
        42,
        [1, 2, 3],
    ],
)
def test_extract_usage_metrics_returns_well_formed_counts_for_malformed_usage(usage):
    tru_usage, _ = strands.event_loop.streaming.extract_usage_metrics({"usage": usage, "metrics": {"latencyMs": 1}})

    # Every consumer does arithmetic on these counts, so a malformed one must not survive the
    # boundary — otherwise the raise merely moves to whichever consumer touches it first.
    assert all(type(count) is int for count in tru_usage.values())
    accumulate_usage({"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}, tru_usage)
    context_token_count(tru_usage)
    prompt_token_count(tru_usage)


@pytest.mark.parametrize(
    "usage, exp_warning",
    [
        # A count reported as something other than a non-negative integer is coerced away, which
        # can still satisfy the invariant, so it is reported on separately.
        ({"inputTokens": "abc", "outputTokens": 4, "totalTokens": 4}, True),
        ({"inputTokens": "10", "outputTokens": 5, "totalTokens": 15}, True),
        ({"inputTokens": -5, "outputTokens": 4, "totalTokens": 4}, True),
        ({"inputTokens": 10.7, "outputTokens": 4, "totalTokens": 14}, True),
        # A model reporting nothing, or a genuinely empty turn, is not a fault.
        ({"inputTokens": None, "outputTokens": None, "totalTokens": None}, False),
        ({"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}, False),
        ({}, False),
        # Bedrock reports cacheDetails alongside the counters; a structured field is not a count.
        (
            {
                "inputTokens": 12,
                "outputTokens": 1,
                "totalTokens": 5857,
                "cacheWriteInputTokens": 5844,
                "cacheDetails": [{"ttl": "5m", "inputTokens": 5844}],
            },
            False,
        ),
    ],
)
def test_extract_usage_metrics_reports_counts_it_could_not_use(usage, exp_warning, caplog):
    caplog.set_level(logging.WARNING, logger="strands.event_loop.streaming")

    strands.event_loop.streaming.extract_usage_metrics({"usage": usage, "metrics": {"latencyMs": 1}})

    assert ("not usable numbers" in caplog.text) is exp_warning


def test_extract_usage_metrics_does_not_run_a_reported_counts_own_equality():
    """A reported count can be an object whose __eq__ raises.

    The unusable check compares the coerced count against what was reported, so comparing by value
    would run model-supplied code and fail the invocation the counts merely describe.
    """

    class RaisingEquality:
        def __eq__(self, other):
            raise RuntimeError("comparison should never run")

        def __hash__(self):
            return 0

    class VendorUsage:
        def __init__(self):
            self.inputTokens = RaisingEquality()
            self.outputTokens = 1
            self.totalTokens = 1

    tru_usage, _ = strands.event_loop.streaming.extract_usage_metrics(
        {"usage": VendorUsage(), "metrics": {"latencyMs": 1}}
    )

    assert tru_usage == {"inputTokens": 0, "outputTokens": 1, "totalTokens": 1}


@pytest.mark.parametrize(
    "usage, exp_warning",
    [
        # JSON may carry a whole count as a float, which survives coercion and is usable.
        ({"inputTokens": 10.0, "outputTokens": 5, "totalTokens": 15.0}, False),
        ({"inputTokens": 10.7, "outputTokens": 5, "totalTokens": 15}, True),
    ],
)
def test_extract_usage_metrics_accepts_a_whole_float_count(usage, exp_warning, caplog):
    caplog.set_level(logging.WARNING, logger="strands.event_loop.streaming")

    strands.event_loop.streaming.extract_usage_metrics({"usage": usage, "metrics": {"latencyMs": 1}})

    assert ("not usable numbers" in caplog.text) is exp_warning


def test_extract_usage_metrics_reads_an_int_subclass_as_a_usable_count(caplog):
    """An ``IntEnum`` is a count a provider legitimately reports, and must not read as unusable.

    Coercion returns a plain int so no model-supplied type reaches a consumer. Reporting such a
    count as unusable would tell an operator a good count is bad, and that warning is the only
    signal the SDK gives for one it genuinely could not read.
    """
    caplog.set_level(logging.WARNING, logger="strands.event_loop.streaming")

    tru_usage, _ = strands.event_loop.streaming.extract_usage_metrics(
        {"usage": {"inputTokens": _CountEnum.TEN, "outputTokens": 5, "totalTokens": 15}, "metrics": {"latencyMs": 1}}
    )

    assert tru_usage == {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}
    assert all(type(value) is int for value in tru_usage.values())
    assert "not usable numbers" not in caplog.text
