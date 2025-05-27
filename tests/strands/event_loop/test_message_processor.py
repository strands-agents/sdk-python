import copy

import pytest

from strands.event_loop import message_processor


@pytest.mark.parametrize(
    "messages,expected,expected_messages",
    [
        # Orphaned toolUse with empty input, no toolResult
        (
            [
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "1", "input": {}, "name": "foo"}}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": "2"}}]},
            ],
            True,
            [
                {"role": "assistant", "content": [{"text": "[Attempted to use foo, but operation was canceled]"}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": "2"}}]},
            ],
        ),
        # toolUse with input, has matching toolResult
        (
            [
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "1", "input": {"a": 1}, "name": "foo"}}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": "1"}}]},
            ],
            False,
            [
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "1", "input": {"a": 1}, "name": "foo"}}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": "1"}}]},
            ],
        ),
        # No messages
        (
            [],
            False,
            [],
        ),
    ],
)
def test_clean_orphaned_empty_tool_uses(messages, expected, expected_messages):
    test_messages = copy.deepcopy(messages)
    result = message_processor.clean_orphaned_empty_tool_uses(test_messages)
    assert result == expected
    assert test_messages == expected_messages


@pytest.mark.parametrize(
    "messages,expected_idx",
    [
        (
            [
                {"role": "user", "content": [{"text": "hi"}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": "1"}}]},
                {"role": "assistant", "content": [{"text": "ok"}]},
            ],
            1,
        ),
        (
            [
                {"role": "user", "content": [{"text": "hi"}]},
                {"role": "assistant", "content": [{"text": "ok"}]},
            ],
            None,
        ),
        (
            [],
            None,
        ),
    ],
)
def test_find_last_message_with_tool_results(messages, expected_idx):
    idx = message_processor.find_last_message_with_tool_results(messages)
    assert idx == expected_idx


@pytest.mark.parametrize(
    "messages,msg_idx,expected_changed,expected_content",
    [
        (
            [
                {
                    "role": "user",
                    "content": [{"toolResult": {"toolUseId": "1", "status": "ok", "content": [{"text": "big"}]}}],
                }
            ],
            0,
            True,
            [
                {
                    "toolResult": {
                        "toolUseId": "1",
                        "status": "error",
                        "content": [{"text": "The tool result was too large!"}],
                    }
                }
            ],
        ),
        (
            [{"role": "user", "content": [{"text": "no tool result"}]}],
            0,
            False,
            [{"text": "no tool result"}],
        ),
        (
            [],
            0,
            False,
            [],
        ),
        (
            [{"role": "user", "content": [{"toolResult": {"toolUseId": "1"}}]}],
            2,
            False,
            [{"toolResult": {"toolUseId": "1"}}],
        ),
    ],
)
def test_truncate_tool_results(messages, msg_idx, expected_changed, expected_content):
    test_messages = copy.deepcopy(messages)
    changed = message_processor.truncate_tool_results(test_messages, msg_idx)
    assert changed == expected_changed
    if 0 <= msg_idx < len(test_messages):
        assert test_messages[msg_idx]["content"] == expected_content
    else:
        assert test_messages == messages
