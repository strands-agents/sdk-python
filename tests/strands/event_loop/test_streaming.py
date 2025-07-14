import unittest.mock

import pytest

import strands
import strands.event_loop
from strands.types.streaming import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    MessageStartEvent,
    MessageStopEvent,
)


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
                {"role": "assistant"},
                {"role": "user", "content": [{"text": " \n"}]},
            ],
            [
                {"role": "assistant", "content": [{"text": "a"}, {"toolUse": {}}]},
                {"role": "assistant", "content": [{"toolUse": {}}]},
                {"role": "assistant", "content": [{"text": "a"}, {"text": "[blank text]"}]},
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
def test_remove_blank_messages_content_text(messages, exp_result):
    tru_result = strands.event_loop.streaming.remove_blank_messages_content_text(messages)

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
    ],
)
def test_handle_content_block_start(chunk: ContentBlockStartEvent, exp_tool_use):
    tru_tool_use = strands.event_loop.streaming.handle_content_block_start(chunk)

    assert tru_tool_use == exp_tool_use


@pytest.mark.parametrize(
    ("event", "state", "exp_updated_state", "callback_args"),
    [
        # Tool Use - Existing input
        (
            {"delta": {"toolUse": {"input": '"value"}'}}},
            {"current_tool_use": {"input": '{"key": '}},
            {"current_tool_use": {"input": '{"key": "value"}'}},
            {"current_tool_use": {"input": '{"key": "value"}'}},
        ),
        # Tool Use - New input
        (
            {"delta": {"toolUse": {"input": '{"key": '}}},
            {"current_tool_use": {}},
            {"current_tool_use": {"input": '{"key": '}},
            {"current_tool_use": {"input": '{"key": '}},
        ),
        # Text
        (
            {"delta": {"text": " world"}},
            {"text": "hello"},
            {"text": "hello world"},
            {"data": " world"},
        ),
        # Reasoning - Text - Existing
        (
            {"delta": {"reasoningContent": {"text": "king"}}},
            {"reasoningText": "thin"},
            {"reasoningText": "thinking"},
            {"reasoningText": "king", "reasoning": True},
        ),
        # Reasoning - Text - New
        (
            {"delta": {"reasoningContent": {"text": "thin"}}},
            {},
            {"reasoningText": "thin"},
            {"reasoningText": "thin", "reasoning": True},
        ),
        # Reasoning - Signature - Existing
        (
            {"delta": {"reasoningContent": {"signature": "ue"}}},
            {"signature": "val"},
            {"signature": "value"},
            {"reasoning_signature": "ue", "reasoning": True},
        ),
        # Reasoning - Signature - New
        (
            {"delta": {"reasoningContent": {"signature": "val"}}},
            {},
            {"signature": "val"},
            {"reasoning_signature": "val", "reasoning": True},
        ),
        # Reasoning - Empty
        (
            {"delta": {"reasoningContent": {}}},
            {},
            {},
            {},
        ),
        # Empty
        (
            {"delta": {}},
            {},
            {},
            {},
        ),
    ],
)
def test_handle_content_block_delta(event: ContentBlockDeltaEvent, state, exp_updated_state, callback_args):
    exp_callback_event = {"callback": {**callback_args, "delta": event["delta"]}} if callback_args else {}

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
            },
            {
                "content": [{"toolUse": {"toolUseId": "123", "name": "test", "input": {"key": "value"}}}],
                "current_tool_use": {},
                "text": "",
                "reasoningText": "",
            },
        ),
        # Tool Use - Missing input
        (
            {
                "content": [],
                "current_tool_use": {"toolUseId": "123", "name": "test"},
                "text": "",
                "reasoningText": "",
            },
            {
                "content": [{"toolUse": {"toolUseId": "123", "name": "test", "input": {}}}],
                "current_tool_use": {},
                "text": "",
                "reasoningText": "",
            },
        ),
        # Text
        (
            {
                "content": [],
                "current_tool_use": {},
                "text": "test",
                "reasoningText": "",
            },
            {
                "content": [{"text": "test"}],
                "current_tool_use": {},
                "text": "",
                "reasoningText": "",
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
            },
            {
                "content": [{"reasoningContent": {"reasoningText": {"text": "test", "signature": "123"}}}],
                "current_tool_use": {},
                "text": "",
                "reasoningText": "",
                "signature": "123",
            },
        ),
        # Empty
        (
            {
                "content": [],
                "current_tool_use": {},
                "text": "",
                "reasoningText": "",
            },
            {
                "content": [],
                "current_tool_use": {},
                "text": "",
                "reasoningText": "",
            },
        ),
    ],
)
def test_handle_content_block_stop(state, exp_updated_state):
    tru_updated_state = strands.event_loop.streaming.handle_content_block_stop(state)

    assert tru_updated_state == exp_updated_state


def test_handle_message_stop():
    event: MessageStopEvent = {"stopReason": "end_turn"}

    tru_reason = strands.event_loop.streaming.handle_message_stop(event)
    exp_reason = "end_turn"

    assert tru_reason == exp_reason


def test_extract_usage_metrics():
    event = {
        "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
        "metrics": {"latencyMs": 0},
    }

    tru_usage, tru_metrics = strands.event_loop.streaming.extract_usage_metrics(event)
    exp_usage, exp_metrics = event["usage"], event["metrics"]

    assert tru_usage == exp_usage and tru_metrics == exp_metrics


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
                    "callback": {
                        "event": {
                            "messageStart": {
                                "role": "assistant",
                            },
                        },
                    },
                },
                {
                    "callback": {
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
                },
                {
                    "callback": {
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
                },
                {
                    "callback": {
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
                    },
                },
                {
                    "callback": {
                        "event": {
                            "contentBlockStop": {},
                        },
                    },
                },
                {
                    "callback": {
                        "event": {
                            "messageStop": {
                                "stopReason": "tool_use",
                            },
                        },
                    },
                },
                {
                    "callback": {
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
        # Empty Message
        (
            [{}],
            [
                {
                    "callback": {
                        "event": {},
                    },
                },
                {
                    "stop": (
                        "end_turn",
                        {
                            "role": "assistant",
                            "content": [],
                        },
                        {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                        {"latencyMs": 0},
                    ),
                },
            ],
        ),
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
                        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 1},
                        "metrics": {"latencyMs": 1},
                    }
                },
            ],
            [
                {
                    "callback": {
                        "event": {
                            "messageStart": {
                                "role": "assistant",
                            },
                        },
                    },
                },
                {
                    "callback": {
                        "event": {
                            "contentBlockStart": {
                                "start": {},
                            },
                        },
                    },
                },
                {
                    "callback": {
                        "event": {
                            "contentBlockDelta": {
                                "delta": {
                                    "text": "Hello!",
                                },
                            },
                        },
                    },
                },
                {
                    "callback": {
                        "data": "Hello!",
                        "delta": {
                            "text": "Hello!",
                        },
                    },
                },
                {
                    "callback": {
                        "event": {
                            "contentBlockStop": {},
                        },
                    },
                },
                {
                    "callback": {
                        "event": {
                            "messageStop": {
                                "stopReason": "guardrail_intervened",
                            },
                        },
                    },
                },
                {
                    "callback": {
                        "event": {
                            "redactContent": {
                                "redactAssistantContentMessage": "REDACTED.",
                                "redactUserContentMessage": "REDACTED",
                            },
                        },
                    },
                },
                {
                    "callback": {
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
                },
                {
                    "stop": (
                        "guardrail_intervened",
                        {
                            "role": "assistant",
                            "content": [{"text": "REDACTED."}],
                        },
                        {"inputTokens": 1, "outputTokens": 1, "totalTokens": 1},
                        {"latencyMs": 1},
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
        system_prompt="test prompt",
        messages=[{"role": "assistant", "content": [{"text": "a"}, {"text": " \n"}]}],
        tool_specs=None,
    )

    tru_events = await alist(stream)
    exp_events = [
        {
            "callback": {
                "event": {
                    "contentBlockDelta": {
                        "delta": {
                            "text": "test",
                        },
                    },
                },
            },
        },
        {
            "callback": {
                "data": "test",
                "delta": {
                    "text": "test",
                },
            },
        },
        {
            "callback": {
                "event": {
                    "contentBlockStop": {},
                },
            },
        },
        {
            "stop": (
                "end_turn",
                {"role": "assistant", "content": [{"text": "test"}]},
                {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                {"latencyMs": 0},
            )
        },
    ]
    assert tru_events == exp_events

    mock_model.stream.assert_called_with(
        [{"role": "assistant", "content": [{"text": "a"}, {"text": "[blank text]"}]}],
        None,
        "test prompt",
    )
