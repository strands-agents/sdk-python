"""Tests for the injection delivery primitives."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from strands._middleware.stages import InvokeModelContext
from strands.injection._message_injection import (
    _create_injection_middleware,
    _fold_into_last_user_message,
    _fold_into_system_prompt,
    _is_user_turn,
    _resolve_trigger,
)


def user(text: str) -> dict:
    return {"role": "user", "content": [{"text": text}]}


def assistant(text: str) -> dict:
    return {"role": "assistant", "content": [{"text": text}]}


def tool_result() -> dict:
    return {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": "t1", "status": "success", "content": [{"text": "done"}]}}],
    }


def injection_ctx(messages: list[dict]) -> Any:
    # _resolve_trigger predicates only read `messages`; a minimal stub suffices.
    ctx = MagicMock()
    ctx.messages = messages
    return ctx


def make_agent(state: Any = None) -> Any:
    agent = MagicMock()
    agent.state = state if state is not None else MagicMock()
    return agent


def invoke_ctx(messages: list[dict], agent: Any = None, system_prompt: Any = None) -> InvokeModelContext:
    """Build an InvokeModelContext; the handler reads messages and agent.state/agent."""
    return InvokeModelContext(
        agent=agent or make_agent(),
        messages=messages,
        system_prompt=system_prompt,
        tool_specs=[],
        tool_choice=None,
        invocation_state={},
    )


class TestFoldIntoLastUserMessage:
    def test_prepends_text_ahead_of_user_content(self):
        messages = [user("original task"), assistant("prior step"), user("next ask")]
        result = _fold_into_last_user_message(messages, "INJECTED")

        assert result == [
            {"role": "user", "content": [{"text": "original task"}]},
            {"role": "assistant", "content": [{"text": "prior step"}]},
            {"role": "user", "content": [{"text": "INJECTED"}, {"text": "next ask"}]},
        ]

    def test_returns_new_list_and_does_not_mutate_input(self):
        original = user("ask")
        messages = [assistant("prior"), original]
        result = _fold_into_last_user_message(messages, "INJECTED")

        assert result is not messages
        assert messages[1] is original
        assert len(original["content"]) == 1  # untouched
        assert result[1] is not original

    def test_appends_after_tool_result_block(self):
        tr = tool_result()
        result = _fold_into_last_user_message([user("task"), assistant("thinking"), tr], "INJECTED")

        # Providers require the tool result to be the first block, so the text is appended.
        assert result == [
            {"role": "user", "content": [{"text": "task"}]},
            {"role": "assistant", "content": [{"text": "thinking"}]},
            {"role": "user", "content": [tr["content"][0], {"text": "INJECTED"}]},
        ]

    def test_targets_most_recent_user_message(self):
        messages = [user("first"), assistant("a"), user("second")]
        result = _fold_into_last_user_message(messages, "INJECTED")

        assert result == [
            {"role": "user", "content": [{"text": "first"}]},
            {"role": "assistant", "content": [{"text": "a"}]},
            {"role": "user", "content": [{"text": "INJECTED"}, {"text": "second"}]},
        ]

    def test_preserves_message_metadata(self):
        tagged = {"role": "user", "content": [{"text": "ask"}], "metadata": {"custom": {"keep": "me"}}}
        result = _fold_into_last_user_message([tagged], "INJECTED")
        assert result[0]["metadata"] == {"custom": {"keep": "me"}}

    def test_returns_input_unchanged_when_no_user_message(self):
        messages = [assistant("only assistant")]
        result = _fold_into_last_user_message(messages, "INJECTED")
        assert result is messages


class TestIsUserTurn:
    def test_true_on_plain_user_ask(self):
        assert _is_user_turn([assistant("prior"), user("ask")]) is True

    def test_false_on_user_tool_result_turn(self):
        assert _is_user_turn([user("task"), assistant("a"), tool_result()]) is False

    def test_false_on_assistant_message(self):
        assert _is_user_turn([user("ask"), assistant("reply")]) is False

    def test_false_on_empty_conversation(self):
        assert _is_user_turn([]) is False


class TestResolveTrigger:
    def test_default_uses_user_turn(self):
        trigger = _resolve_trigger(None)
        assert trigger(injection_ctx([user("ask")])) is True
        assert trigger(injection_ctx([tool_result()])) is False

    def test_user_turn_uses_is_user_turn(self):
        trigger = _resolve_trigger("userTurn")
        assert trigger(injection_ctx([user("ask")])) is True
        assert trigger(injection_ctx([tool_result()])) is False

    def test_every_turn_always_fires(self):
        trigger = _resolve_trigger("everyTurn")
        assert trigger(injection_ctx([])) is True
        assert trigger(injection_ctx([tool_result()])) is True

    def test_custom_predicate_over_context(self):
        trigger = _resolve_trigger(lambda context: len(context.messages) >= 2)
        assert trigger(injection_ctx([user("a")])) is False
        assert trigger(injection_ctx([user("a"), assistant("b")])) is True

    def test_fails_open_when_custom_predicate_raises(self, caplog):
        def boom(context):
            raise ValueError("boom")

        trigger = _resolve_trigger(boom)
        assert trigger(injection_ctx([user("ask")])) is False
        assert "skipping injection" in caplog.text


@pytest.mark.asyncio
class TestCreateInjectionMiddleware:
    async def test_folds_text_into_latest_user_message(self):
        handler = _create_injection_middleware(lambda context: "INJECTED")
        result = await handler(invoke_ctx([assistant("prior"), user("ask")]))

        assert result.messages == [
            {"role": "assistant", "content": [{"text": "prior"}]},
            {"role": "user", "content": [{"text": "INJECTED"}, {"text": "ask"}]},
        ]

    async def test_passes_conversation_to_render_content(self):
        seen = []

        def render(context):
            seen.extend(message["role"] for message in context.messages)
            return "x"

        handler = _create_injection_middleware(render)
        await handler(invoke_ctx([assistant("prior"), user("ask")]))

        assert seen == ["assistant", "user"]

    async def test_exposes_state_and_agent_on_context(self):
        agent = make_agent(state="stashed")
        received = {}

        def render(context):
            received["state"] = context.state
            received["agent"] = context.agent
            return None

        handler = _create_injection_middleware(render)
        await handler(invoke_ctx([user("ask")], agent=agent))

        assert received == {"state": "stashed", "agent": agent}

    async def test_supports_async_render_content(self):
        async def render(context):
            return "INJECTED"

        handler = _create_injection_middleware(render)
        result = await handler(invoke_ctx([user("ask")]))

        assert result.messages == [{"role": "user", "content": [{"text": "INJECTED"}, {"text": "ask"}]}]

    async def test_returns_context_unchanged_when_trigger_does_not_fire(self):
        render = MagicMock(return_value="x")
        handler = _create_injection_middleware(render)  # default 'userTurn'
        ctx = invoke_ctx([user("task"), assistant("a"), tool_result()])
        result = await handler(ctx)

        assert result is ctx
        render.assert_not_called()

    async def test_every_turn_injects_on_tool_result_turn_keeping_tool_result_first(self):
        handler = _create_injection_middleware(lambda context: "INJECTED", trigger="everyTurn")
        tr = tool_result()
        result = await handler(invoke_ctx([user("task"), assistant("a"), tr]))

        assert result.messages == [
            {"role": "user", "content": [{"text": "task"}]},
            {"role": "assistant", "content": [{"text": "a"}]},
            {"role": "user", "content": [tr["content"][0], {"text": "INJECTED"}]},
        ]

    async def test_returns_context_unchanged_when_render_yields_empty(self):
        handler = _create_injection_middleware(lambda context: "   ")
        ctx = invoke_ctx([assistant("prior"), user("ask")])
        result = await handler(ctx)

        assert result is ctx

    async def test_fails_open_when_render_content_raises(self, caplog):
        def render(context):
            raise ValueError("boom")

        handler = _create_injection_middleware(render)
        ctx = invoke_ctx([assistant("prior"), user("ask")])
        result = await handler(ctx)

        assert result is ctx
        assert "skipping injection" in caplog.text

    async def test_does_not_mutate_original_context_messages(self):
        handler = _create_injection_middleware(lambda context: "INJECTED")
        ctx = invoke_ctx([assistant("prior"), user("ask")])
        before = ctx.messages[1]
        await handler(ctx)

        assert len(before["content"]) == 1  # original user message untouched


class TestFoldIntoSystemPrompt:
    def test_none_prompt_becomes_the_text(self):
        assert _fold_into_system_prompt(None, "INJECTED") == "INJECTED"

    def test_string_prompt_appends_with_blank_line(self):
        assert _fold_into_system_prompt("Base prompt.", "INJECTED") == "Base prompt.\n\nINJECTED"

    def test_block_prompt_appends_trailing_text_block(self):
        blocks = [{"text": "Base prompt."}, {"cachePoint": {"type": "default"}}]
        result = _fold_into_system_prompt(blocks, "INJECTED")
        assert result == [
            {"text": "Base prompt."},
            {"cachePoint": {"type": "default"}},
            {"text": "INJECTED"},
        ]

    def test_returns_new_list_and_does_not_mutate_input(self):
        blocks = [{"text": "Base prompt."}]
        result = _fold_into_system_prompt(blocks, "INJECTED")
        assert result is not blocks
        assert blocks == [{"text": "Base prompt."}]


@pytest.mark.asyncio
class TestSystemPromptLocation:
    async def test_appends_to_per_call_system_prompt_and_leaves_messages_untouched(self):
        handler = _create_injection_middleware(lambda context: "INJECTED", location="systemPrompt")
        ctx = invoke_ctx([user("ask")], system_prompt="Base prompt.")

        result = await handler(ctx)

        assert result.system_prompt == "Base prompt.\n\nINJECTED"
        assert result.messages == [user("ask")]

    async def test_none_prompt_becomes_injected_text(self):
        handler = _create_injection_middleware(lambda context: "INJECTED", location="systemPrompt")

        result = await handler(invoke_ctx([user("ask")], system_prompt=None))

        assert result.system_prompt == "INJECTED"

    async def test_respects_trigger_gate(self):
        handler = _create_injection_middleware(lambda context: "INJECTED", location="systemPrompt")
        ctx = invoke_ctx([assistant("reply")], system_prompt="Base prompt.")

        result = await handler(ctx)

        assert result is ctx

    async def test_every_turn_injects_on_tool_result_turn(self):
        handler = _create_injection_middleware(lambda context: "INJECTED", trigger="everyTurn", location="systemPrompt")
        ctx = invoke_ctx([user("ask"), assistant("use tool"), tool_result()], system_prompt="Base prompt.")

        result = await handler(ctx)

        assert result.system_prompt == "Base prompt.\n\nINJECTED"
        # The tool-result turn is untouched: injection landed in the system prompt instead.
        assert result.messages == ctx.messages

    async def test_default_location_still_folds_into_messages(self):
        handler = _create_injection_middleware(lambda context: "INJECTED")
        ctx = invoke_ctx([user("ask")], system_prompt="Base prompt.")

        result = await handler(ctx)

        assert result.system_prompt == "Base prompt."
        assert result.messages == [{"role": "user", "content": [{"text": "INJECTED"}, {"text": "ask"}]}]


@pytest.mark.asyncio
class TestInjectedTokenAccounting:
    """The handler folds the injected text's estimated tokens into ``projected_input_tokens``.

    The event-loop projection is computed before input middleware runs, so it cannot see
    injected text; the handler corrects the projection on the returned context so downstream
    consumers see the true size of the request.
    """

    def agent_with_counter(self, injected_tokens: int = 100, limit: int | None = None) -> Any:
        agent = make_agent()
        agent.model.count_tokens = AsyncMock(return_value=injected_tokens)
        agent.model.context_window_limit = limit
        return agent

    def ctx_with_projection(
        self, agent: Any, projected: int | None, system_prompt: Any = "Base."
    ) -> InvokeModelContext:
        return InvokeModelContext(
            agent=agent,
            messages=[user("ask")],
            system_prompt=system_prompt,
            tool_specs=[],
            tool_choice=None,
            invocation_state={},
            projected_input_tokens=projected,
        )

    async def test_system_prompt_injection_bumps_projection(self):
        agent = self.agent_with_counter(injected_tokens=100)
        handler = _create_injection_middleware(lambda context: "INJECTED", location="systemPrompt")

        result = await handler(self.ctx_with_projection(agent, projected=448))

        assert result.projected_input_tokens == 548
        assert result.system_prompt == "Base.\n\nINJECTED"

    async def test_last_user_message_injection_bumps_projection(self):
        agent = self.agent_with_counter(injected_tokens=100)
        handler = _create_injection_middleware(lambda context: "INJECTED")

        result = await handler(self.ctx_with_projection(agent, projected=2))

        assert result.projected_input_tokens == 102

    async def test_counts_the_injected_text(self):
        agent = self.agent_with_counter()
        handler = _create_injection_middleware(lambda context: "INJECTED", location="systemPrompt")

        await handler(self.ctx_with_projection(agent, projected=1))

        agent.model.count_tokens.assert_awaited_once_with([{"role": "user", "content": [{"text": "INJECTED"}]}])

    async def test_no_projection_skips_counting(self):
        agent = self.agent_with_counter()
        handler = _create_injection_middleware(lambda context: "INJECTED", location="systemPrompt")

        result = await handler(self.ctx_with_projection(agent, projected=None))

        assert result.projected_input_tokens is None
        agent.model.count_tokens.assert_not_awaited()
        assert result.system_prompt == "Base.\n\nINJECTED"  # injection still applied

    async def test_counting_failure_fails_open_and_keeps_injection(self, caplog):
        agent = make_agent()
        agent.model.count_tokens = AsyncMock(side_effect=RuntimeError("boom"))
        handler = _create_injection_middleware(lambda context: "INJECTED", location="systemPrompt")

        result = await handler(self.ctx_with_projection(agent, projected=448))

        assert result.projected_input_tokens == 448  # projection unchanged
        assert result.system_prompt == "Base.\n\nINJECTED"  # injection stands
        assert "projection left unchanged" in caplog.text

    async def test_no_injection_leaves_projection_untouched(self):
        agent = self.agent_with_counter()
        handler = _create_injection_middleware(lambda context: None, location="systemPrompt")

        result = await handler(self.ctx_with_projection(agent, projected=448))

        assert result is not None
        assert result.projected_input_tokens == 448
        agent.model.count_tokens.assert_not_awaited()

    async def test_overflow_logs_deterministic_warning(self, caplog):
        agent = self.agent_with_counter(injected_tokens=2000, limit=1000)
        handler = _create_injection_middleware(lambda context: "INJECTED", location="systemPrompt")

        with caplog.at_level("WARNING"):
            result = await handler(self.ctx_with_projection(agent, projected=500))

        assert result.projected_input_tokens == 2500
        assert "past the model's context window" in caplog.text

    async def test_within_limit_logs_no_warning(self, caplog):
        agent = self.agent_with_counter(injected_tokens=100, limit=10000)
        handler = _create_injection_middleware(lambda context: "INJECTED", location="systemPrompt")

        with caplog.at_level("WARNING"):
            result = await handler(self.ctx_with_projection(agent, projected=500))

        assert result.projected_input_tokens == 600
        assert "past the model's context window" not in caplog.text
