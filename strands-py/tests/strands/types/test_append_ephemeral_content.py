"""Tests for _append_ephemeral_content, the shared cache-boundary primitive.

Both the injection engine and the agentic token-usage middleware append per-call content through
this function, so its contract is pinned here rather than only transitively through those callers.
"""

from strands.types.content import Message, _append_ephemeral_content

_BOUNDARY = {"cachePoint": {"type": "default"}}


def test_appends_behind_a_boundary():
    message: Message = {"role": "user", "content": [{"text": "durable ask"}]}

    appended = _append_ephemeral_content(message, [{"text": "EPHEMERAL"}])

    assert appended["content"] == [{"text": "durable ask"}, _BOUNDARY, {"text": "EPHEMERAL"}]


def test_returns_input_unchanged_when_no_blocks():
    message: Message = {"role": "user", "content": [{"text": "ask"}]}

    assert _append_ephemeral_content(message, []) is message


def test_emits_no_boundary_when_content_is_empty():
    # Nothing precedes the appended content, so there is no prefix worth caching.
    message: Message = {"role": "user", "content": []}

    appended = _append_ephemeral_content(message, [{"text": "EPHEMERAL"}])

    assert appended["content"] == [{"text": "EPHEMERAL"}]


def test_reuses_a_boundary_another_caller_already_opened():
    # Providers cap how many cache points a request may carry, so appending must not stack them.
    message: Message = {"role": "user", "content": [{"text": "ask"}]}

    first = _append_ephemeral_content(message, [{"text": "ONE"}])
    second = _append_ephemeral_content(first, [{"text": "TWO"}])

    assert second["content"] == [{"text": "ask"}, _BOUNDARY, {"text": "ONE"}, {"text": "TWO"}]


def test_does_not_mutate_the_input_message():
    original_content = [{"text": "ask"}]
    message: Message = {"role": "user", "content": original_content}

    appended = _append_ephemeral_content(message, [{"text": "EPHEMERAL"}])

    assert message["content"] == [{"text": "ask"}]
    assert appended["content"] is not original_content


def test_preserves_identity_fields():
    message: Message = {
        "role": "user",
        "content": [{"text": "ask"}],
        "tracking_id": "durable-1",
        "metadata": {"custom": {"keep": "me"}},
    }

    appended = _append_ephemeral_content(message, [{"text": "EPHEMERAL"}])

    assert appended["tracking_id"] == "durable-1"
    assert appended["metadata"] == {"custom": {"keep": "me"}}
