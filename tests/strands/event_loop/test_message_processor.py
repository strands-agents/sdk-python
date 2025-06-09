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
