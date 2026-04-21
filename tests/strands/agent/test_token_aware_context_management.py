"""Tests for token-aware context management features.

Covers:
- _estimate_tokens utility
- SlidingWindowConversationManager: max_context_tokens, token_counter, micro-compaction
- SummarizingConversationManager: proactive token-budget summarization
"""

from typing import cast
from unittest.mock import MagicMock, Mock, patch

import pytest

from strands.agent.agent import Agent
from strands.agent.conversation_manager._token_utils import IMAGE_CHAR_ESTIMATE, estimate_tokens
from strands.agent.conversation_manager.sliding_window_conversation_manager import SlidingWindowConversationManager
from strands.agent.conversation_manager.summarizing_conversation_manager import SummarizingConversationManager
from strands.hooks.events import BeforeModelCallEvent
from strands.hooks.registry import HookRegistry
from strands.types.content import Messages

# ==============================================================================
# estimate_tokens utility tests
# ==============================================================================


class TestEstimateTokens:
    def test_empty_messages(self):
        assert estimate_tokens([]) == 0

    def test_text_messages(self):
        messages: Messages = [
            {"role": "user", "content": [{"text": "Hello world"}]},
            {"role": "assistant", "content": [{"text": "Hi there, how can I help?"}]},
        ]
        result = estimate_tokens(messages)
        total_chars = len("Hello world") + len("Hi there, how can I help?")
        assert result == total_chars // 4

    def test_tool_result_messages(self):
        messages: Messages = [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "123",
                            "content": [{"text": "A" * 1000}],
                            "status": "success",
                        }
                    }
                ],
            }
        ]
        result = estimate_tokens(messages)
        assert result == 1000 // 4

    def test_tool_use_messages(self):
        messages: Messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "123",
                            "name": "read_file",
                            "input": {"path": "/foo/bar.py"},
                        }
                    }
                ],
            }
        ]
        result = estimate_tokens(messages)
        assert result > 0

    def test_image_in_tool_result(self):
        messages: Messages = [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "123",
                            "content": [{"image": {"format": "png", "source": {"bytes": b"data"}}}],
                            "status": "success",
                        }
                    }
                ],
            }
        ]
        result = estimate_tokens(messages)
        assert result == IMAGE_CHAR_ESTIMATE // 4

    def test_standalone_image_block(self):
        messages: Messages = [
            {
                "role": "user",
                "content": [{"image": {"format": "png", "source": {"bytes": b"data"}}}],
            }
        ]
        result = estimate_tokens(messages)
        assert result == IMAGE_CHAR_ESTIMATE // 4

    def test_document_block(self):
        messages: Messages = [
            {
                "role": "user",
                "content": [
                    {
                        "document": {
                            "format": "pdf",
                            "name": "test.pdf",
                            "source": {"bytes": b"x" * 8000},
                        }
                    }
                ],
            }
        ]
        result = estimate_tokens(messages)
        assert result == 8000 // 4

    def test_video_block(self):
        messages: Messages = [
            {
                "role": "user",
                "content": [{"video": {"format": "mp4", "source": {"bytes": b"v"}}}],
            }
        ]
        result = estimate_tokens(messages)
        assert result == (IMAGE_CHAR_ESTIMATE * 10) // 4

    def test_cache_point_block_zero_tokens(self):
        messages: Messages = [
            {"role": "user", "content": [{"cachePoint": {"type": "default"}}]},
        ]
        assert estimate_tokens(messages) == 0

    def test_guard_content_block_zero_tokens(self):
        messages: Messages = [
            {"role": "user", "content": [{"guardContent": {"text": {"text": "check"}}}]},
        ]
        assert estimate_tokens(messages) == 0

    def test_tool_use_input_uses_json_serialization(self):
        messages: Messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "1",
                            "name": "tool",
                            "input": {"key": "value"},
                        }
                    }
                ],
            }
        ]
        result = estimate_tokens(messages)
        # json.dumps produces '{"key": "value"}' (18 chars) + "tool" (4 chars) = 22 chars
        # str() would produce "{'key': 'value'}" (16 chars) — different
        expected_chars = len("tool") + len('{"key": "value"}')
        assert result == expected_chars // 4

    def test_mixed_content(self):
        messages: Messages = [
            {
                "role": "user",
                "content": [
                    {"text": "A" * 400},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": "1", "name": "tool", "input": {}}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "1", "content": [{"text": "B" * 800}], "status": "success"}},
                ],
            },
        ]
        result = estimate_tokens(messages)
        assert result > 0

    def test_empty_content_blocks(self):
        messages: Messages = [{"role": "user", "content": []}]
        assert estimate_tokens(messages) == 0


# ==============================================================================
# SlidingWindowConversationManager — max_context_tokens tests
# ==============================================================================


class TestSlidingWindowTokenBudget:
    def test_default_no_token_budget(self):
        manager = SlidingWindowConversationManager()
        assert manager.max_context_tokens is None

    def test_apply_management_skips_when_under_both_limits(self):
        manager = SlidingWindowConversationManager(
            window_size=100,
            max_context_tokens=10000,
            should_truncate_results=False,
        )
        messages: Messages = [
            {"role": "user", "content": [{"text": "short"}]},
            {"role": "assistant", "content": [{"text": "also short"}]},
        ]
        agent = Agent(messages=messages)
        original = messages.copy()
        manager.apply_management(agent)
        assert messages == original

    def test_apply_management_triggers_on_token_budget_exceeded(self):
        manager = SlidingWindowConversationManager(
            window_size=100,
            max_context_tokens=10,
            should_truncate_results=False,
        )
        messages: Messages = [
            {"role": "user", "content": [{"text": "A" * 200}]},
            {"role": "assistant", "content": [{"text": "B" * 200}]},
            {"role": "user", "content": [{"text": "C" * 200}]},
        ]
        agent = Agent(messages=messages)
        manager.apply_management(agent)
        assert len(messages) < 3

    def test_apply_management_triggers_on_message_limit_even_without_token_budget(self):
        manager = SlidingWindowConversationManager(
            window_size=2,
            should_truncate_results=False,
        )
        messages: Messages = [
            {"role": "user", "content": [{"text": "First"}]},
            {"role": "assistant", "content": [{"text": "Response"}]},
            {"role": "user", "content": [{"text": "Second"}]},
        ]
        agent = Agent(messages=messages)
        manager.apply_management(agent)
        assert len(messages) <= 2

    def test_custom_token_counter(self):
        call_count = 0

        def my_counter(msgs):
            nonlocal call_count
            call_count += 1
            return 99999

        manager = SlidingWindowConversationManager(
            window_size=100,
            max_context_tokens=100,
            token_counter=my_counter,
            should_truncate_results=False,
        )
        messages: Messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi"}]},
            {"role": "user", "content": [{"text": "More"}]},
        ]
        agent = Agent(messages=messages)
        manager.apply_management(agent)
        assert call_count > 0
        assert len(messages) < 3

    def test_token_budget_always_uses_heuristic(self):
        manager = SlidingWindowConversationManager(
            window_size=100,
            max_context_tokens=5000,
        )
        mock_agent = MagicMock()
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "A" * 400}]},
        ]
        # Even when model reports tokens, heuristic is used to avoid staleness
        mock_agent.event_loop_metrics.latest_context_size = 6000

        current = manager._get_current_token_count(mock_agent)
        assert current == 100  # 400 chars / 4, NOT 6000

    def test_before_model_call_enforces_token_budget(self):
        manager = SlidingWindowConversationManager(
            window_size=100,
            max_context_tokens=50,
            should_truncate_results=False,
        )
        registry = HookRegistry()
        manager.register_hooks(registry)

        mock_agent = MagicMock()
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "A" * 400}]},
            {"role": "assistant", "content": [{"text": "B" * 400}]},
            {"role": "user", "content": [{"text": "C" * 400}]},
        ]
        mock_agent.event_loop_metrics.latest_context_size = None
        event = BeforeModelCallEvent(agent=mock_agent, invocation_state={})

        with patch.object(manager, "apply_management") as mock_apply:
            registry.invoke_callbacks(event)
            mock_apply.assert_called_once_with(mock_agent)

    def test_before_model_call_skips_when_under_budget(self):
        manager = SlidingWindowConversationManager(
            window_size=100,
            max_context_tokens=100000,
        )
        registry = HookRegistry()
        manager.register_hooks(registry)

        mock_agent = MagicMock()
        mock_agent.messages = [{"role": "user", "content": [{"text": "short"}]}]
        mock_agent.event_loop_metrics.latest_context_size = None
        event = BeforeModelCallEvent(agent=mock_agent, invocation_state={})

        with patch.object(manager, "apply_management") as mock_apply:
            registry.invoke_callbacks(event)
            mock_apply.assert_not_called()

    def test_backward_compatibility_no_token_params(self):
        manager = SlidingWindowConversationManager(window_size=40)
        assert manager.max_context_tokens is None
        assert manager.compactable_after_messages is None
        messages: Messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi"}]},
        ]
        agent = Agent(messages=messages)
        manager.apply_management(agent)
        assert len(messages) == 2


# ==============================================================================
# SlidingWindowConversationManager — micro-compaction tests
# ==============================================================================


class TestMicroCompaction:
    def test_compactable_after_messages_validation(self):
        with pytest.raises(ValueError, match="compactable_after_messages"):
            SlidingWindowConversationManager(compactable_after_messages=0)
        with pytest.raises(ValueError, match="compactable_after_messages"):
            SlidingWindowConversationManager(compactable_after_messages=-3)

    def test_micro_compact_replaces_old_tool_results(self):
        manager = SlidingWindowConversationManager(
            window_size=100,
            compactable_after_messages=2,
            should_truncate_results=False,
        )
        messages: Messages = [
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": "1", "name": "read", "input": {}}}]},
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "1", "content": [{"text": "A" * 5000}], "status": "success"}}
                ],
            },
            {"role": "user", "content": [{"text": "Recent message 1"}]},
            {"role": "assistant", "content": [{"text": "Recent response"}]},
        ]
        reclaimed = manager._micro_compact(messages)
        assert reclaimed > 0
        assert messages[1]["content"][0]["toolResult"]["content"][0]["text"] == manager._COMPACT_STUB

    def test_micro_compact_preserves_recent_results(self):
        manager = SlidingWindowConversationManager(
            window_size=100,
            compactable_after_messages=3,
            should_truncate_results=False,
        )
        original_text = "B" * 5000
        messages: Messages = [
            {"role": "user", "content": [{"text": "Old message"}]},
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": "2", "name": "read", "input": {}}}]},
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "2", "content": [{"text": original_text}], "status": "success"}}
                ],
            },
        ]
        manager._micro_compact(messages)
        # All 3 messages are within the compactable_after_messages=3 window
        assert messages[2]["content"][0]["toolResult"]["content"][0]["text"] == original_text

    def test_micro_compact_does_not_double_compact(self):
        manager = SlidingWindowConversationManager(
            window_size=100,
            compactable_after_messages=1,
            should_truncate_results=False,
        )
        messages: Messages = [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "1",
                            "content": [{"text": manager._COMPACT_STUB}],
                            "status": "success",
                        }
                    }
                ],
            },
            {"role": "user", "content": [{"text": "Recent"}]},
        ]
        reclaimed = manager._micro_compact(messages)
        assert reclaimed == 0

    def test_micro_compact_preserves_tool_pair_structure(self):
        manager = SlidingWindowConversationManager(
            window_size=100,
            compactable_after_messages=1,
            should_truncate_results=False,
        )
        messages: Messages = [
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": "1", "name": "tool", "input": {}}}]},
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "1", "content": [{"text": "big result"}], "status": "success"}}
                ],
            },
            {"role": "user", "content": [{"text": "Latest"}]},
        ]
        manager._micro_compact(messages)
        assert "toolResult" in messages[1]["content"][0]
        assert messages[1]["content"][0]["toolResult"]["toolUseId"] == "1"
        assert messages[1]["content"][0]["toolResult"]["status"] == "success"

    def test_micro_compact_handles_empty_messages(self):
        manager = SlidingWindowConversationManager(
            window_size=100,
            compactable_after_messages=5,
            should_truncate_results=False,
        )
        reclaimed = manager._micro_compact([])
        assert reclaimed == 0

    def test_micro_compact_runs_in_apply_management(self):
        manager = SlidingWindowConversationManager(
            window_size=100,
            compactable_after_messages=2,
            should_truncate_results=False,
        )
        messages: Messages = [
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": "1", "name": "tool", "input": {}}}]},
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "1", "content": [{"text": "X" * 10000}], "status": "success"}}
                ],
            },
            {"role": "user", "content": [{"text": "Recent 1"}]},
            {"role": "assistant", "content": [{"text": "Recent 2"}]},
        ]
        agent = Agent(messages=messages)
        manager.apply_management(agent)
        assert messages[1]["content"][0]["toolResult"]["content"][0]["text"] == manager._COMPACT_STUB

    def test_micro_compact_skips_non_tool_result_blocks(self):
        manager = SlidingWindowConversationManager(
            window_size=100,
            compactable_after_messages=1,
            should_truncate_results=False,
        )
        messages: Messages = [
            {"role": "user", "content": [{"text": "plain text should not be compacted"}]},
            {"role": "user", "content": [{"text": "Recent"}]},
        ]
        original_text = messages[0]["content"][0]["text"]
        manager._micro_compact(messages)
        assert messages[0]["content"][0]["text"] == original_text

    def test_micro_compact_replaces_image_blocks(self):
        manager = SlidingWindowConversationManager(
            window_size=100,
            compactable_after_messages=1,
            should_truncate_results=False,
        )
        messages: Messages = [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "1",
                            "content": [{"image": {"format": "png", "source": {"bytes": b"bigdata"}}}],
                            "status": "success",
                        }
                    }
                ],
            },
            {"role": "user", "content": [{"text": "Recent"}]},
        ]
        reclaimed = manager._micro_compact(messages)
        assert reclaimed == IMAGE_CHAR_ESTIMATE // 4
        assert messages[0]["content"][0]["toolResult"]["content"][0]["text"] == manager._COMPACT_STUB

    def test_micro_compact_reclaimed_subtracts_stub_length(self):
        manager = SlidingWindowConversationManager(
            window_size=100,
            compactable_after_messages=1,
            should_truncate_results=False,
        )
        original_text = "A" * 200
        messages: Messages = [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "1",
                            "content": [{"text": original_text}],
                            "status": "success",
                        }
                    }
                ],
            },
            {"role": "user", "content": [{"text": "Recent"}]},
        ]
        reclaimed = manager._micro_compact(messages)
        stub_len = len(manager._COMPACT_STUB)
        assert reclaimed == (len(original_text) - stub_len) // 4

    def test_micro_compact_skips_already_processed_messages(self):
        """Issue #9: _last_compacted_index prevents re-scanning already compacted messages."""
        manager = SlidingWindowConversationManager(
            window_size=100,
            compactable_after_messages=1,
            should_truncate_results=False,
        )
        messages: Messages = [
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "1", "content": [{"text": "X" * 5000}], "status": "success"}}
                ],
            },
            {"role": "user", "content": [{"text": "msg2"}]},
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "2", "content": [{"text": "Y" * 5000}], "status": "success"}}
                ],
            },
            {"role": "user", "content": [{"text": "msg4"}]},
        ]
        # First call compacts messages 0-2 (cutoff = 4-1=3)
        reclaimed1 = manager._micro_compact(messages)
        assert reclaimed1 > 0
        assert manager._last_compacted_index == 3

        # Add more messages and compact again — should only process new range
        messages.append({"role": "user", "content": [{"text": "msg5"}]})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "3", "content": [{"text": "Z" * 5000}], "status": "success"}}
                ],
            }
        )
        messages.append({"role": "user", "content": [{"text": "msg7"}]})
        reclaimed2 = manager._micro_compact(messages)
        # Should compact messages at indices 3-5 (cutoff=7-1=6), but _last_compacted_index=3
        # so it starts from 3
        assert reclaimed2 > 0

    def test_micro_compact_then_truncation_interaction(self):
        """Issue #11: micro-compaction + truncation work together without conflict."""
        manager = SlidingWindowConversationManager(
            window_size=100,
            compactable_after_messages=2,
            should_truncate_results=True,
            max_context_tokens=10,
        )
        messages: Messages = [
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": "1", "name": "tool", "input": {}}}]},
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "1", "content": [{"text": "X" * 10000}], "status": "success"}}
                ],
            },
            {"role": "user", "content": [{"text": "Recent 1"}]},
            {"role": "assistant", "content": [{"text": "Recent 2"}]},
            {"role": "user", "content": [{"text": "Recent 3"}]},
        ]
        agent = Agent(messages=messages)

        # First micro-compaction should replace old tool result
        manager._micro_compact(messages)
        assert messages[1]["content"][0]["toolResult"]["content"][0]["text"] == manager._COMPACT_STUB

        # Then apply_management which checks token budget — should still work
        manager.apply_management(agent)
        # Messages should still be valid (no crash)
        assert len(messages) > 0


# ==============================================================================
# Parameter validation tests
# ==============================================================================


class TestParameterValidation:
    def test_max_context_tokens_zero_raises_sliding_window(self):
        with pytest.raises(ValueError, match="max_context_tokens"):
            SlidingWindowConversationManager(max_context_tokens=0)

    def test_max_context_tokens_negative_raises_sliding_window(self):
        with pytest.raises(ValueError, match="max_context_tokens"):
            SlidingWindowConversationManager(max_context_tokens=-100)

    def test_max_context_tokens_zero_raises_summarizing(self):
        with pytest.raises(ValueError, match="max_context_tokens"):
            SummarizingConversationManager(max_context_tokens=0)

    def test_max_context_tokens_negative_raises_summarizing(self):
        with pytest.raises(ValueError, match="max_context_tokens"):
            SummarizingConversationManager(max_context_tokens=-50)

    def test_max_context_tokens_positive_accepted(self):
        sw = SlidingWindowConversationManager(max_context_tokens=1000)
        assert sw.max_context_tokens == 1000
        sm = SummarizingConversationManager(max_context_tokens=5000)
        assert sm.max_context_tokens == 5000


# ==============================================================================
# SummarizingConversationManager — proactive token-budget tests
# ==============================================================================


async def _mock_model_stream(response_text):
    yield {"messageStart": {"role": "assistant"}}
    yield {"contentBlockStart": {"start": {}}}
    yield {"contentBlockDelta": {"delta": {"text": response_text}}}
    yield {"contentBlockStop": {}}
    yield {"messageStop": {"stopReason": "end_turn"}}


class MockSummarizationAgent:
    def __init__(self, summary_response="Summary of conversation."):
        self.summary_response = summary_response
        self.system_prompt = None
        self.messages = []
        self.model = Mock()
        self.model.stream = Mock(side_effect=lambda *a, **kw: _mock_model_stream(self.summary_response))
        self.call_tracker = Mock()
        self.tool_registry = Mock()
        self.tool_names = []
        self._default_structured_output_model = None


def create_mock_agent(summary_response="Summary of conversation.") -> "Agent":
    return cast("Agent", MockSummarizationAgent(summary_response))


class TestSummarizingTokenBudget:
    def test_default_no_token_budget(self):
        manager = SummarizingConversationManager()
        assert manager.max_context_tokens is None
        assert manager.proactive_threshold == 0.8

    def test_proactive_threshold_clamped(self):
        manager = SummarizingConversationManager(max_context_tokens=1000, proactive_threshold=0.05)
        assert manager.proactive_threshold == 0.1

        manager = SummarizingConversationManager(max_context_tokens=1000, proactive_threshold=1.5)
        assert manager.proactive_threshold == 1.0

    def test_apply_management_triggers_when_over_budget(self):
        """apply_management checks token budget and triggers summarization when exceeded."""
        manager = SummarizingConversationManager(
            max_context_tokens=100,
            proactive_threshold=0.5,
            preserve_recent_messages=1,
        )
        mock_agent = create_mock_agent()
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "A" * 10000}]},
            {"role": "assistant", "content": [{"text": "B" * 10000}]},
        ]
        mock_agent.event_loop_metrics = MagicMock()
        mock_agent.event_loop_metrics.latest_context_size = None

        with patch.object(manager, "reduce_context") as mock_reduce:
            manager.apply_management(mock_agent)
            mock_reduce.assert_called_once()

    def test_apply_management_noop_without_token_budget(self):
        manager = SummarizingConversationManager()
        mock_agent = create_mock_agent()
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "A" * 10000}]},
            {"role": "assistant", "content": [{"text": "B" * 10000}]},
        ]
        original = mock_agent.messages.copy()
        manager.apply_management(mock_agent)
        assert mock_agent.messages == original

    def test_before_model_call_proactive_summarization(self):
        manager = SummarizingConversationManager(
            max_context_tokens=100,
            proactive_threshold=0.5,
            preserve_recent_messages=1,
        )
        registry = HookRegistry()
        manager.register_hooks(registry)

        mock_agent = MagicMock()
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "A" * 1000}]},
        ]
        mock_agent.event_loop_metrics.latest_context_size = None
        event = BeforeModelCallEvent(agent=mock_agent, invocation_state={})

        with patch.object(manager, "reduce_context") as mock_reduce:
            registry.invoke_callbacks(event)
            mock_reduce.assert_called_once()

    def test_before_model_call_skips_without_token_budget(self):
        """Issue #16: hook is not registered when max_context_tokens is None."""
        manager = SummarizingConversationManager()
        registry = HookRegistry()
        manager.register_hooks(registry)

        mock_agent = MagicMock()
        mock_agent.messages = [{"role": "user", "content": [{"text": "A" * 10000}]}]
        mock_agent.event_loop_metrics.latest_context_size = None
        event = BeforeModelCallEvent(agent=mock_agent, invocation_state={})

        with patch.object(manager, "reduce_context") as mock_reduce:
            registry.invoke_callbacks(event)
            mock_reduce.assert_not_called()

    def test_token_count_always_uses_heuristic(self):
        """Issue #3: _get_current_token_count always uses heuristic, never stale model-reported value."""
        manager = SummarizingConversationManager(
            max_context_tokens=5000,
            proactive_threshold=0.8,
        )
        mock_agent = MagicMock()
        mock_agent.messages = [{"role": "user", "content": [{"text": "A" * 400}]}]
        mock_agent.event_loop_metrics.latest_context_size = 4500

        current = manager._get_current_token_count(mock_agent)
        assert current == 100  # 400 chars / 4, NOT 4500

    def test_custom_token_counter(self):
        def always_big(msgs):
            return 999999

        manager = SummarizingConversationManager(
            max_context_tokens=100,
            token_counter=always_big,
            preserve_recent_messages=1,
        )
        registry = HookRegistry()
        manager.register_hooks(registry)

        mock_agent = MagicMock()
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "A"}]},
            {"role": "assistant", "content": [{"text": "B"}]},
        ]
        event = BeforeModelCallEvent(agent=mock_agent, invocation_state={})

        with patch.object(manager, "reduce_context") as mock_reduce:
            registry.invoke_callbacks(event)
            mock_reduce.assert_called_once()

    def test_backward_compatibility(self):
        manager = SummarizingConversationManager(
            summary_ratio=0.5,
            preserve_recent_messages=2,
        )
        assert manager.max_context_tokens is None

        mock_agent = create_mock_agent()
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "A" * 10000}]},
            {"role": "assistant", "content": [{"text": "B" * 10000}]},
        ]
        original = mock_agent.messages.copy()
        manager.apply_management(mock_agent)
        assert mock_agent.messages == original

    def test_proactive_summarization_catches_all_exceptions(self):
        """Issue #4: hook catches Exception, not just ContextWindowOverflowException."""
        manager = SummarizingConversationManager(
            max_context_tokens=10,
            proactive_threshold=0.5,
            preserve_recent_messages=1,
        )
        registry = HookRegistry()
        manager.register_hooks(registry)

        mock_agent = MagicMock()
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "A" * 400}]},
        ]
        event = BeforeModelCallEvent(agent=mock_agent, invocation_state={})

        with patch.object(manager, "reduce_context", side_effect=RuntimeError("model timeout")):
            # Should not raise — gracefully logs warning
            registry.invoke_callbacks(event)

    def test_reduce_context_preserves_cause_chain_with_exception(self):
        """Issue #13: raise from e only when e is not None."""
        manager = SummarizingConversationManager(
            preserve_recent_messages=100,
        )
        mock_agent = create_mock_agent()
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "short"}]},
        ]
        mock_agent.event_loop_metrics = MagicMock()
        mock_agent.event_loop_metrics.latest_context_size = None

        # When e=None, should raise without "from None" (preserves natural __cause__)
        with pytest.raises(Exception) as exc_info:
            manager.reduce_context(mock_agent, e=None)
        assert exc_info.value.__cause__ is None


# ==============================================================================
# _model_call_count semantics regression tests
# ==============================================================================


class TestModelCallCountSemantics:
    """Issue #18: _model_call_count should only increment when per_turn is enabled."""

    def test_model_call_count_not_incremented_when_per_turn_false(self):
        manager = SlidingWindowConversationManager(per_turn=False)
        registry = HookRegistry()
        manager.register_hooks(registry)

        mock_agent = MagicMock()
        mock_agent.messages = [{"role": "user", "content": [{"text": "hi"}]}]
        mock_agent.event_loop_metrics.latest_context_size = None
        event = BeforeModelCallEvent(agent=mock_agent, invocation_state={})

        registry.invoke_callbacks(event)
        registry.invoke_callbacks(event)
        registry.invoke_callbacks(event)
        assert manager._model_call_count == 0

    def test_model_call_count_incremented_when_per_turn_true(self):
        manager = SlidingWindowConversationManager(per_turn=True)
        registry = HookRegistry()
        manager.register_hooks(registry)

        mock_agent = MagicMock()
        mock_agent.messages = [{"role": "user", "content": [{"text": "hi"}]}]
        mock_agent.event_loop_metrics.latest_context_size = None
        event = BeforeModelCallEvent(agent=mock_agent, invocation_state={})

        with patch.object(manager, "apply_management"):
            registry.invoke_callbacks(event)
            registry.invoke_callbacks(event)
        assert manager._model_call_count == 2

    def test_model_call_count_incremented_when_per_turn_int(self):
        manager = SlidingWindowConversationManager(per_turn=3)
        registry = HookRegistry()
        manager.register_hooks(registry)

        mock_agent = MagicMock()
        mock_agent.messages = [{"role": "user", "content": [{"text": "hi"}]}]
        mock_agent.event_loop_metrics.latest_context_size = None
        event = BeforeModelCallEvent(agent=mock_agent, invocation_state={})

        with patch.object(manager, "apply_management"):
            for _ in range(5):
                registry.invoke_callbacks(event)
        assert manager._model_call_count == 5

    def test_per_turn_int_modulo_applies_correctly(self):
        applied_count = 0

        manager = SlidingWindowConversationManager(per_turn=3)

        def counting_apply(agent, **kwargs):
            nonlocal applied_count
            applied_count += 1

        registry = HookRegistry()
        manager.register_hooks(registry)

        mock_agent = MagicMock()
        mock_agent.messages = [{"role": "user", "content": [{"text": "hi"}]}]
        mock_agent.event_loop_metrics.latest_context_size = None
        event = BeforeModelCallEvent(agent=mock_agent, invocation_state={})

        with patch.object(manager, "apply_management", side_effect=counting_apply):
            for _ in range(9):
                registry.invoke_callbacks(event)
        # Should apply at calls 3, 6, 9
        assert applied_count == 3


# ==============================================================================
# Integration: hook -> apply_management -> reduce_context flow
# ==============================================================================


class TestIntegrationHookFlow:
    """Issue #10: Integration test for the full hook -> apply_management -> reduce_context flow."""

    def test_sliding_window_hook_triggers_full_management_pipeline(self):
        manager = SlidingWindowConversationManager(
            window_size=100,
            max_context_tokens=50,
            compactable_after_messages=2,
            should_truncate_results=False,
        )
        registry = HookRegistry()
        manager.register_hooks(registry)

        messages: Messages = [
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": "1", "name": "tool", "input": {}}}]},
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "1", "content": [{"text": "X" * 5000}], "status": "success"}}
                ],
            },
            {"role": "user", "content": [{"text": "msg3"}]},
            {"role": "assistant", "content": [{"text": "msg4"}]},
            {"role": "user", "content": [{"text": "msg5"}]},
        ]
        agent = Agent(messages=messages)
        event = BeforeModelCallEvent(agent=agent, invocation_state={})

        registry.invoke_callbacks(event)

        # Micro-compaction should have run on old tool result
        # The exact state depends on whether reduce_context also trimmed
        # but the key invariant is: no crash and messages are valid
        assert len(messages) > 0
        for msg in messages:
            assert msg["role"] in ("user", "assistant")

    def test_last_compacted_index_adjusted_after_trim(self):
        """_last_compacted_index is adjusted when messages are trimmed by reduce_context."""
        manager = SlidingWindowConversationManager(
            window_size=2,
            compactable_after_messages=1,
            should_truncate_results=False,
        )
        initial_index = 5
        manager._last_compacted_index = initial_index

        messages: Messages = [
            {"role": "user", "content": [{"text": "a"}]},
            {"role": "assistant", "content": [{"text": "b"}]},
            {"role": "user", "content": [{"text": "c"}]},
            {"role": "assistant", "content": [{"text": "d"}]},
            {"role": "user", "content": [{"text": "e"}]},
        ]
        original_len = len(messages)
        agent = Agent(messages=messages)

        manager.reduce_context(agent)
        trimmed_count = original_len - len(messages)
        assert trimmed_count > 0
        assert manager._last_compacted_index == max(0, initial_index - trimmed_count)


# ==============================================================================
# Token budget convergence tests
# ==============================================================================


class TestTokenBudgetConvergence:
    """Bug #1: apply_management must loop reduce_context until under token budget."""

    def test_converges_when_under_window_size_but_over_token_budget(self):
        """Messages under window_size but over max_context_tokens — must reduce repeatedly.

        5 messages x 400 chars = 2000 chars / 4 = 500 tokens, budget = 100.
        Each reduce_context trims 2 messages (default when under window_size).
        After 1st: 3 msgs, 300 tokens. After 2nd: 1 msg, 100 tokens. Converges.
        """
        manager = SlidingWindowConversationManager(
            window_size=100,
            max_context_tokens=100,
            should_truncate_results=False,
        )
        messages: Messages = [
            {"role": "user", "content": [{"text": "A" * 400}]},
            {"role": "assistant", "content": [{"text": "B" * 400}]},
            {"role": "user", "content": [{"text": "C" * 400}]},
            {"role": "assistant", "content": [{"text": "D" * 400}]},
            {"role": "user", "content": [{"text": "E" * 400}]},
        ]
        agent = Agent(messages=messages)
        manager.apply_management(agent)
        current_tokens = manager.token_counter(agent.messages)
        assert current_tokens <= 100
        assert len(messages) < 5

    def test_stops_when_no_progress(self):
        """If reduce_context can't shrink further, apply_management should not loop forever."""
        manager = SlidingWindowConversationManager(
            window_size=100,
            max_context_tokens=1,
            should_truncate_results=False,
        )
        messages: Messages = [
            {"role": "user", "content": [{"text": "A" * 400}]},
        ]
        agent = Agent(messages=messages)
        manager.apply_management(agent)
        assert len(messages) >= 1

    def test_loop_terminates_when_reduce_makes_no_progress(self):
        """apply_management stops looping when reduce_context can't shrink further.

        Patches reduce_context to never actually remove messages, simulating a stuck state.
        The loop must detect no-progress and break rather than spinning.
        """
        reduce_call_count = 0

        def noop_reduce(agent, **kwargs):
            nonlocal reduce_call_count
            reduce_call_count += 1

        manager = SlidingWindowConversationManager(
            window_size=100,
            max_context_tokens=1,
            should_truncate_results=False,
        )
        messages: Messages = [
            {"role": "user", "content": [{"text": "A" * 400}]},
            {"role": "assistant", "content": [{"text": "B" * 400}]},
            {"role": "user", "content": [{"text": "C" * 400}]},
        ]
        agent = Agent(messages=messages)

        with patch.object(manager, "reduce_context", side_effect=noop_reduce):
            manager.apply_management(agent)

        assert reduce_call_count == 1


# ==============================================================================
# State round-trip tests
# ==============================================================================


class TestStateRoundTrip:
    """Test gap #13: get_state / restore_from_session round-trip for new fields."""

    def test_sliding_window_state_round_trip(self):
        manager = SlidingWindowConversationManager(
            window_size=40,
            compactable_after_messages=5,
            per_turn=3,
        )
        manager._model_call_count = 7
        manager._last_compacted_index = 12
        manager.removed_message_count = 3

        state = manager.get_state()
        assert state["model_call_count"] == 7
        assert state["last_compacted_index"] == 12
        assert state["removed_message_count"] == 3

        new_manager = SlidingWindowConversationManager(
            window_size=40,
            compactable_after_messages=5,
            per_turn=3,
        )
        new_manager.restore_from_session(state)

        assert new_manager._model_call_count == 7
        assert new_manager._last_compacted_index == 12
        assert new_manager.removed_message_count == 3

    def test_sliding_window_state_defaults_for_missing_keys(self):
        """Backward compat: old session state without new keys should use defaults."""
        manager = SlidingWindowConversationManager(window_size=40)
        state = {
            "__name__": "SlidingWindowConversationManager",
            "removed_message_count": 5,
        }
        manager.restore_from_session(state)
        assert manager._model_call_count == 0
        assert manager._last_compacted_index == 0
        assert manager.removed_message_count == 5


# ==============================================================================
# SummarizingConversationManager — apply_management contract tests
# ==============================================================================


class TestSummarizingApplyManagement:
    """Design #5: apply_management should honor the token budget contract."""

    def test_apply_management_triggers_summarization_when_over_budget(self):
        manager = SummarizingConversationManager(
            max_context_tokens=100,
            proactive_threshold=0.5,
            preserve_recent_messages=1,
        )
        mock_agent = create_mock_agent()
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "A" * 10000}]},
            {"role": "assistant", "content": [{"text": "B" * 10000}]},
        ]

        with patch.object(manager, "reduce_context") as mock_reduce:
            manager.apply_management(mock_agent)
            mock_reduce.assert_called_once()

    def test_apply_management_noop_when_under_budget(self):
        manager = SummarizingConversationManager(
            max_context_tokens=100000,
            proactive_threshold=0.8,
        )
        mock_agent = create_mock_agent()
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "short"}]},
        ]

        with patch.object(manager, "reduce_context") as mock_reduce:
            manager.apply_management(mock_agent)
            mock_reduce.assert_not_called()

    def test_apply_management_noop_without_max_context_tokens(self):
        manager = SummarizingConversationManager()
        mock_agent = create_mock_agent()
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "A" * 10000}]},
        ]

        with patch.object(manager, "reduce_context") as mock_reduce:
            manager.apply_management(mock_agent)
            mock_reduce.assert_not_called()

    def test_apply_management_catches_exceptions(self):
        manager = SummarizingConversationManager(
            max_context_tokens=10,
            proactive_threshold=0.5,
            preserve_recent_messages=1,
        )
        mock_agent = create_mock_agent()
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "A" * 400}]},
        ]

        with patch.object(manager, "reduce_context", side_effect=RuntimeError("model timeout")):
            manager.apply_management(mock_agent)

    def test_no_double_summarization_in_same_cycle(self):
        """Hook and apply_management in same cycle should not both call reduce_context."""
        reduce_count = 0

        def counting_reduce(agent, **kwargs):
            nonlocal reduce_count
            reduce_count += 1
            # Simulate successful summarization by shrinking messages
            agent.messages[:] = agent.messages[-1:]

        manager = SummarizingConversationManager(
            max_context_tokens=10,
            proactive_threshold=0.5,
            preserve_recent_messages=1,
        )
        registry = HookRegistry()
        manager.register_hooks(registry)

        mock_agent = MagicMock()
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "A" * 400}]},
            {"role": "assistant", "content": [{"text": "B" * 400}]},
        ]
        event = BeforeModelCallEvent(agent=mock_agent, invocation_state={})

        with patch.object(manager, "reduce_context", side_effect=counting_reduce):
            # Hook fires — triggers first summarization
            registry.invoke_callbacks(event)
            assert reduce_count == 1

            # apply_management fires (as agent's finally block would) — should skip
            manager.apply_management(mock_agent)
            assert reduce_count == 1

    def test_summarization_runs_again_after_new_messages(self):
        """After new messages arrive, summarization should fire again."""
        reduce_count = 0

        def counting_reduce(agent, **kwargs):
            nonlocal reduce_count
            reduce_count += 1
            agent.messages[:] = agent.messages[-1:]

        manager = SummarizingConversationManager(
            max_context_tokens=10,
            proactive_threshold=0.5,
            preserve_recent_messages=1,
        )

        mock_agent = MagicMock()
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "A" * 400}]},
            {"role": "assistant", "content": [{"text": "B" * 400}]},
        ]

        with patch.object(manager, "reduce_context", side_effect=counting_reduce):
            manager.apply_management(mock_agent)
            assert reduce_count == 1

            # Simulate new messages arriving
            mock_agent.messages.append({"role": "user", "content": [{"text": "C" * 400}]})
            mock_agent.messages.append({"role": "assistant", "content": [{"text": "D" * 400}]})

            manager.apply_management(mock_agent)
            assert reduce_count == 2
