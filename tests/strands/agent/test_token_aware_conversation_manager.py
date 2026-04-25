"""Tests for TokenAwareConversationManager."""

from typing import TYPE_CHECKING, cast
from unittest.mock import Mock, patch

import pytest

if TYPE_CHECKING:
    from strands.agent.agent import Agent

from strands.agent.conversation_manager.token_aware_conversation_manager import (
    SUMMARIZATION_PROMPT,
    TokenAwareConversationManager,
    _sanitize_text,
)
from strands.hooks.events import BeforeModelCallEvent
from strands.hooks.registry import HookRegistry
from strands.types.content import Message, Messages
from strands.types.exceptions import ContextWindowOverflowException

# ---------------------------------------------------------------------------
# Async mock helpers (same pattern as test_summarizing_conversation_manager)
# ---------------------------------------------------------------------------


async def _mock_model_stream(response_text: str):
    """Create an async generator that yields stream events for a text response."""
    yield {"messageStart": {"role": "assistant"}}
    yield {"contentBlockStart": {"start": {}}}
    yield {"contentBlockDelta": {"delta": {"text": response_text}}}
    yield {"contentBlockStop": {}}
    yield {"messageStop": {"stopReason": "end_turn"}}


async def _mock_model_stream_error(error: Exception):
    """Async generator that raises an exception, simulating a model failure."""
    raise error
    yield  # pragma: no cover – makes this a generator


class MockAgent:
    """Mock agent for testing token-aware conversation manager."""

    def __init__(self, summary_response: str = "Summary of conversation."):
        self.summary_response = summary_response
        self.system_prompt = "You are helpful."
        self.messages: Messages = []
        self.model = Mock()
        self.model.stream = Mock(side_effect=lambda *a, **kw: _mock_model_stream(self.summary_response))
        self.tool_registry = Mock()
        self.tool_names: list[str] = []
        self.event_loop_metrics = Mock()
        invocation = Mock()
        cycle = Mock()
        cycle.usage = {"inputTokens": 200_000}
        invocation.cycles = [cycle]
        self.event_loop_metrics.latest_agent_invocation = invocation


def _create_mock_agent(summary_response: str = "Summary of conversation.") -> "MockAgent":
    return MockAgent(summary_response)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager():
    """Token-aware manager with low threshold for testing."""
    return TokenAwareConversationManager(compact_threshold=100, preserve_recent=2)


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


def test_init_defaults():
    """Test initialization with default values."""
    m = TokenAwareConversationManager()
    assert m.compact_threshold == 150_000
    assert m.preserve_recent == 6
    assert m.should_truncate_results is True
    assert m._last_input_tokens == 0
    assert m._model_call_count == 0
    assert m._summary_message is None


def test_init_custom():
    """Test initialization with custom values."""
    m = TokenAwareConversationManager(compact_threshold=50_000, preserve_recent=4, should_truncate_results=False)
    assert m.compact_threshold == 50_000
    assert m.preserve_recent == 4
    assert m.should_truncate_results is False


# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------


def test_register_hooks():
    """Test that hooks are registered with the registry."""
    m = TokenAwareConversationManager()
    registry = HookRegistry()
    m.register_hooks(registry)
    assert registry.has_callbacks()


# ---------------------------------------------------------------------------
# ANSI sanitization
# ---------------------------------------------------------------------------


def test_sanitize_strips_ansi_from_tool_results(manager):
    """Test ANSI escape codes are stripped from tool result content."""
    messages: Messages = [
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "1",
                        "content": [{"text": "\x1b[31mred\x1b[0m normal"}],
                        "status": "success",
                    }
                }
            ],
        }
    ]
    manager._sanitize_all_tool_results(messages)
    assert messages[0]["content"][0]["toolResult"]["content"][0]["text"] == "red normal"


def test_sanitize_collapses_repeated_lines(manager):
    """Test that repeated consecutive lines are collapsed when ANSI triggers sanitization."""
    repeated = "\x1b[0mline\nline\nline\nline\nother"
    messages: Messages = [
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "1",
                        "content": [{"text": repeated}],
                        "status": "success",
                    }
                }
            ],
        }
    ]
    manager._sanitize_all_tool_results(messages)
    result = messages[0]["content"][0]["toolResult"]["content"][0]["text"]
    assert "[repeated" in result
    assert result.count("line") < 4


def test_sanitize_skips_clean_text(manager):
    """Test that messages without ANSI or carriage returns are not modified."""
    messages: Messages = [
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "1",
                        "content": [{"text": "clean text"}],
                        "status": "success",
                    }
                }
            ],
        }
    ]
    manager._sanitize_all_tool_results(messages)
    assert messages[0]["content"][0]["toolResult"]["content"][0]["text"] == "clean text"


def test_sanitize_text_function():
    """Test the module-level _sanitize_text helper."""
    text = "\x1b[31mhello\x1b[0m\ndup\ndup\ndup\nend"
    result = _sanitize_text(text)
    assert "\x1b" not in result
    assert "[repeated 2 more time(s)]" in result


# ---------------------------------------------------------------------------
# Tool result truncation
# ---------------------------------------------------------------------------


def test_truncate_tool_results_replaces_content(manager):
    """Test that tool result content is replaced with placeholder."""
    messages: Messages = [
        {"role": "user", "content": [{"text": "keep me"}]},
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "1",
                        "content": [{"text": "big result data"}],
                        "status": "success",
                    }
                }
            ],
        },
        {"role": "user", "content": [{"text": "recent"}]},
    ]
    changed = manager._truncate_tool_results_in_message(messages, 1)
    assert changed
    assert messages[1]["content"][0]["toolResult"]["content"][0]["text"] == "The tool result was too large!"
    assert messages[1]["content"][0]["toolResult"]["status"] == "error"


def test_truncate_skips_already_truncated(manager):
    """Test that already-truncated tool results are not modified again."""
    messages: Messages = [
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "1",
                        "content": [{"text": "The tool result was too large!"}],
                        "status": "error",
                    }
                }
            ],
        },
    ]
    changed = manager._truncate_tool_results_in_message(messages, 0)
    assert not changed


def test_truncate_skips_non_tool_messages(manager):
    """Test that non-tool messages are not modified."""
    messages: Messages = [{"role": "user", "content": [{"text": "hello"}]}]
    changed = manager._truncate_tool_results_in_message(messages, 0)
    assert not changed


# ---------------------------------------------------------------------------
# Tool pair adjustment
# ---------------------------------------------------------------------------


def test_adjust_split_skips_tool_result(manager):
    """Test that split point moves past orphaned toolResult."""
    messages: Messages = [
        {"role": "user", "content": [{"text": "msg"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "1", "name": "t", "input": {}}}]},
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "1", "content": [{"text": "r"}], "status": "success"}}],
        },
        {"role": "assistant", "content": [{"text": "done"}]},
    ]
    assert manager._adjust_split_for_tool_pairs(messages, 2) == 3


def test_adjust_split_valid_position_unchanged(manager):
    """Test that a valid split point is returned unchanged."""
    messages: Messages = [
        {"role": "user", "content": [{"text": "msg"}]},
        {"role": "assistant", "content": [{"text": "response"}]},
        {"role": "user", "content": [{"text": "msg2"}]},
    ]
    assert manager._adjust_split_for_tool_pairs(messages, 1) == 1


def test_adjust_split_raises_on_all_tool_pairs():
    """Test that exception is raised when no valid split point exists."""
    m = TokenAwareConversationManager()
    # toolResult without preceding toolUse at every position — no valid split
    messages: Messages = [
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "1", "content": [{"text": "r"}], "status": "success"}}],
        },
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "2", "content": [{"text": "r"}], "status": "success"}}],
        },
    ]
    with pytest.raises(ContextWindowOverflowException):
        m._adjust_split_for_tool_pairs(messages, 0)


def test_adjust_split_tooluse_with_following_result(manager):
    """Test that toolUse followed by toolResult is a valid split point."""
    messages: Messages = [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "1", "name": "t", "input": {}}}]},
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "1", "content": [{"text": "r"}], "status": "success"}}],
        },
        {"role": "assistant", "content": [{"text": "done"}]},
    ]
    # Split at 0: toolUse with toolResult at 1 is valid
    assert manager._adjust_split_for_tool_pairs(messages, 0) == 0


# ---------------------------------------------------------------------------
# Compact — too few messages
# ---------------------------------------------------------------------------


def test_compact_raises_when_too_few_messages(manager):
    """Test that compaction raises when message count <= preserve_recent."""
    agent = cast("Agent", _create_mock_agent())
    agent.messages = [{"role": "user", "content": [{"text": "only one"}]}]
    with pytest.raises(ContextWindowOverflowException, match="Cannot reduce"):
        manager._compact(agent)


# ---------------------------------------------------------------------------
# Compact — truncation pass
# ---------------------------------------------------------------------------


def test_compact_truncates_before_summarizing(manager):
    """Test that truncation pass runs before summarization."""
    agent = cast("Agent", _create_mock_agent())
    agent.messages = [
        {"role": "user", "content": [{"text": "task"}]},
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "1",
                        "content": [{"text": "big result"}],
                        "status": "success",
                    }
                }
            ],
        },
        {"role": "assistant", "content": [{"text": "R1"}]},
        {"role": "user", "content": [{"text": "recent1"}]},
        {"role": "assistant", "content": [{"text": "recent2"}]},
    ]
    manager._compact(agent)
    # Should have truncated tool result and returned early
    assert agent.messages[1]["content"][0]["toolResult"]["content"][0]["text"] == "The tool result was too large!"


# ---------------------------------------------------------------------------
# Compact — summarization pass
# ---------------------------------------------------------------------------


def test_compact_summarizes_old_messages(manager):
    """Test that older messages are summarized preserving first message and proper role alternation."""
    agent = cast("Agent", _create_mock_agent())
    agent.messages = [
        {"role": "user", "content": [{"text": "Original task"}]},
        {"role": "assistant", "content": [{"text": "R1"}]},
        {"role": "user", "content": [{"text": "M2"}]},
        {"role": "assistant", "content": [{"text": "R2"}]},
        {"role": "user", "content": [{"text": "M3"}]},
        {"role": "assistant", "content": [{"text": "R3"}]},
    ]
    manager.should_truncate_results = False
    manager._compact(agent)
    # First message preserved as original user prompt
    assert agent.messages[0]["content"][0]["text"] == "Original task"
    assert agent.messages[0]["role"] == "user"
    # Summary is assistant role (natural alternation: user → assistant → user)
    assert agent.messages[1]["role"] == "assistant"
    assert "Summary" in agent.messages[1]["content"][0]["text"]
    # Recent messages kept
    assert len(agent.messages) <= 5


def test_compact_falls_back_to_trim_on_summarization_failure(manager):
    """Test that compaction falls back to trimming when summarization fails."""
    agent = cast("Agent", _create_mock_agent())
    agent.model.stream = Mock(side_effect=lambda *a, **kw: _mock_model_stream_error(Exception("fail")))
    agent.messages = [
        {"role": "user", "content": [{"text": "task"}]},
        {"role": "assistant", "content": [{"text": "R1"}]},
        {"role": "user", "content": [{"text": "M2"}]},
        {"role": "assistant", "content": [{"text": "R2"}]},
        {"role": "user", "content": [{"text": "M3"}]},
        {"role": "assistant", "content": [{"text": "R3"}]},
    ]
    manager.should_truncate_results = False
    manager._compact(agent)
    # First message preserved (trim doesn't merge — no summary generated)
    assert agent.messages[0]["content"][0]["text"] == "task"
    assert len(agent.messages) < 6


def test_compact_tracks_removed_message_count(manager):
    """Test that removed_message_count is properly tracked across summarizations."""
    agent = cast("Agent", _create_mock_agent())
    agent.messages = [
        {"role": "user", "content": [{"text": "task"}]},
        {"role": "assistant", "content": [{"text": "R1"}]},
        {"role": "user", "content": [{"text": "M2"}]},
        {"role": "assistant", "content": [{"text": "R2"}]},
        {"role": "user", "content": [{"text": "M3"}]},
        {"role": "assistant", "content": [{"text": "R3"}]},
    ]
    manager.should_truncate_results = False
    manager._compact(agent)
    assert manager.removed_message_count > 0


# ---------------------------------------------------------------------------
# LLM summarization
# ---------------------------------------------------------------------------


def test_generate_summary_calls_model_stream(manager):
    """Test that _generate_summary calls model.stream() and returns assistant role."""
    agent = cast("Agent", _create_mock_agent())
    messages: Messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi"}]},
    ]
    summary = manager._generate_summary(messages, agent)
    assert summary["role"] == "assistant"
    assert "Summary" in summary["content"][0]["text"]
    agent.model.stream.assert_called_once()


def test_generate_summary_uses_summarization_prompt(manager):
    """Test that model.stream() is called with the summarization system prompt."""
    agent = cast("Agent", _create_mock_agent())
    messages: Messages = [{"role": "user", "content": [{"text": "test"}]}]
    manager._generate_summary(messages, agent)
    call_kwargs = agent.model.stream.call_args
    assert call_kwargs.kwargs["system_prompt"] == SUMMARIZATION_PROMPT


def test_generate_summary_does_not_modify_agent_state(manager):
    """Test that agent state is untouched after summarization."""
    agent = _create_mock_agent()
    original_prompt = agent.system_prompt
    original_messages = agent.messages.copy()

    messages: Messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi"}]},
    ]
    manager._generate_summary(messages, cast("Agent", agent))

    assert agent.system_prompt == original_prompt
    assert agent.messages == original_messages


def test_generate_summary_raises_on_model_failure(manager):
    """Test that _generate_summary raises when model.stream() fails."""
    agent = cast("Agent", _create_mock_agent())
    agent.model.stream = Mock(side_effect=lambda *a, **kw: _mock_model_stream_error(Exception("Model failed")))
    messages: Messages = [{"role": "user", "content": [{"text": "test"}]}]

    with pytest.raises(Exception, match="Model failed"):
        manager._generate_summary(messages, agent)


# ---------------------------------------------------------------------------
# Hook callbacks
# ---------------------------------------------------------------------------


def test_before_model_call_captures_tokens_from_previous_cycle(manager):
    """Test that _on_before_model_call reads tokens from cycles[-2] (the completed cycle)."""
    agent = Mock()
    # Simulate 2 cycles: one completed (with tokens) and one just started (empty)
    completed_cycle = Mock()
    completed_cycle.usage = {"inputTokens": 200_000}
    current_cycle = Mock()
    current_cycle.usage = {"inputTokens": 0}
    agent.event_loop_metrics.latest_agent_invocation.cycles = [completed_cycle, current_cycle]
    event = BeforeModelCallEvent(agent=agent)
    with patch.object(manager, "apply_management"):
        manager._on_before_model_call(event)
    assert manager._model_call_count == 1
    assert manager._last_input_tokens == 200_000


def test_before_model_call_skips_when_only_one_cycle(manager):
    """Test that tokens are not read on the first model call (only 1 cycle, no previous)."""
    agent = Mock()
    current_cycle = Mock()
    current_cycle.usage = {"inputTokens": 0}
    agent.event_loop_metrics.latest_agent_invocation.cycles = [current_cycle]
    event = BeforeModelCallEvent(agent=agent)
    manager._on_before_model_call(event)
    assert manager._last_input_tokens == 0
    assert manager._model_call_count == 1


def test_before_model_call_skips_management_when_no_invocation(manager):
    """Test that management is skipped when no invocation exists."""
    agent = Mock()
    agent.event_loop_metrics.latest_agent_invocation = None
    event = BeforeModelCallEvent(agent=agent)
    with patch.object(manager, "apply_management") as mock_apply:
        manager._on_before_model_call(event)
        mock_apply.assert_not_called()


def test_before_model_call_triggers_management_above_threshold(manager):
    """Test that management is triggered when previous cycle tokens exceed threshold."""
    agent = Mock()
    completed_cycle = Mock()
    completed_cycle.usage = {"inputTokens": 200_000}
    current_cycle = Mock()
    current_cycle.usage = {"inputTokens": 0}
    agent.event_loop_metrics.latest_agent_invocation.cycles = [completed_cycle, current_cycle]
    event = BeforeModelCallEvent(agent=agent)
    with patch.object(manager, "apply_management") as mock_apply:
        manager._on_before_model_call(event)
        mock_apply.assert_called_once_with(agent)


# ---------------------------------------------------------------------------
# apply_management / reduce_context
# ---------------------------------------------------------------------------


def test_apply_management_skips_below_threshold(manager):
    """Test that apply_management does nothing when below threshold."""
    manager._last_input_tokens = 50  # below 100 threshold
    agent = Mock()
    agent.messages = []
    with patch.object(manager, "_compact") as mock_compact:
        manager.apply_management(agent)
        mock_compact.assert_not_called()


def test_apply_management_triggers_above_threshold(manager):
    """Test that apply_management triggers compaction when above threshold."""
    manager._last_input_tokens = 200
    agent = Mock()
    agent.messages = [{"role": "user", "content": [{"text": f"msg{i}"}]} for i in range(10)]
    with patch.object(manager, "_compact") as mock_compact:
        manager.apply_management(agent)
        mock_compact.assert_called_once_with(agent)


def test_reduce_context_calls_compact(manager):
    """Test that reduce_context delegates to _compact."""
    agent = Mock()
    with patch.object(manager, "_compact") as mock_compact:
        manager.reduce_context(agent)
        mock_compact.assert_called_once_with(agent)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


def test_get_state(manager):
    """Test that get_state returns complete manager state."""
    manager._last_input_tokens = 1000
    manager._model_call_count = 5
    state = manager.get_state()
    assert state["last_input_tokens"] == 1000
    assert state["model_call_count"] == 5
    assert state["__name__"] == "TokenAwareConversationManager"
    assert state["removed_message_count"] == 0
    assert state["summary_message"] is None


def test_restore_from_session_with_summary(manager):
    """Test that restore_from_session restores all state including summary."""
    summary: Message = {"role": "user", "content": [{"text": "prev summary"}]}
    state = {
        "__name__": "TokenAwareConversationManager",
        "removed_message_count": 3,
        "last_input_tokens": 500,
        "model_call_count": 10,
        "summary_message": summary,
    }
    result = manager.restore_from_session(state)
    assert manager._last_input_tokens == 500
    assert manager._model_call_count == 10
    assert manager._summary_message == summary
    assert manager.removed_message_count == 3
    assert result == [summary]


def test_restore_from_session_without_summary(manager):
    """Test that restore_from_session returns None when no summary exists."""
    state = {
        "__name__": "TokenAwareConversationManager",
        "removed_message_count": 0,
        "last_input_tokens": 0,
        "model_call_count": 0,
        "summary_message": None,
    }
    result = manager.restore_from_session(state)
    assert result is None


def test_restore_from_session_wrong_name_raises(manager):
    """Test that restore raises with mismatched manager name."""
    state = {
        "__name__": "SlidingWindowConversationManager",
        "removed_message_count": 0,
    }
    with pytest.raises(ValueError, match="Invalid conversation manager state"):
        manager.restore_from_session(state)


# ---------------------------------------------------------------------------
# Second summarization properly accounts for previous summary
# ---------------------------------------------------------------------------


def test_second_compact_does_not_double_count_summary(manager):
    """Test that removed_message_count subtracts previous summary on re-summarization."""
    agent = cast("Agent", _create_mock_agent())
    agent.messages = [
        {"role": "user", "content": [{"text": "task"}]},
        {"role": "assistant", "content": [{"text": "R1"}]},
        {"role": "user", "content": [{"text": "M2"}]},
        {"role": "assistant", "content": [{"text": "R2"}]},
        {"role": "user", "content": [{"text": "M3"}]},
        {"role": "assistant", "content": [{"text": "R3"}]},
    ]
    manager.should_truncate_results = False

    # First compact
    manager._compact(agent)
    first_removed = manager.removed_message_count

    # Add more messages
    agent.messages.extend(
        [
            {"role": "user", "content": [{"text": "M4"}]},
            {"role": "assistant", "content": [{"text": "R4"}]},
            {"role": "user", "content": [{"text": "M5"}]},
            {"role": "assistant", "content": [{"text": "R5"}]},
        ]
    )

    # Second compact
    manager._compact(agent)
    # Should have subtracted 1 for the previous summary message
    assert manager.removed_message_count >= first_removed
