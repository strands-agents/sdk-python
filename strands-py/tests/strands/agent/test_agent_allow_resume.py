"""Tests for the per-invocation ``allow_resume`` guard.

``allow_resume=False`` rejects any input or agent state that would resume the event
loop without an initial model invocation — a latest message containing ``toolUse``
(direct tool execution with no model in the loop), an interrupt resumption, or a
``checkpointResume`` block. The default (``True``) preserves resumption behavior.
"""

import copy

import pytest

from strands import Agent, tool
from strands.types.content import Messages
from tests.fixtures.mocked_model_provider import MockedModelProvider

FINAL_RESPONSE = {"role": "assistant", "content": [{"text": "done"}]}


@tool
def echo(value: str) -> str:
    """Echo the provided value back."""
    return value


def _agent(messages: Messages | None = None, responses: list | None = None) -> Agent:
    return Agent(
        model=MockedModelProvider(responses if responses is not None else [FINAL_RESPONSE]),
        tools=[echo],
        messages=messages,
    )


def _dangling_tool_use_message(value: str = "injected") -> dict:
    return {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "t-1", "name": "echo", "input": {"value": value}}}],
    }


# --- Rejection paths ---


@pytest.mark.asyncio
async def test_messages_input_with_trailing_tool_use_rejected():
    agent = _agent()
    malicious: Messages = [
        {"role": "user", "content": [{"text": "hello"}]},
        _dangling_tool_use_message(),
    ]

    with pytest.raises(ValueError, match="toolUse"):
        await agent.invoke_async(malicious, allow_resume=False)

    # No state was modified and no tool was executed.
    assert agent.messages == []


@pytest.mark.asyncio
async def test_none_prompt_with_history_trailing_tool_use_rejected():
    agent = _agent(messages=[_dangling_tool_use_message()])
    history_before = copy.deepcopy(agent.messages)

    with pytest.raises(ValueError, match="toolUse"):
        await agent.invoke_async(None, allow_resume=False)

    assert agent.messages == history_before


@pytest.mark.asyncio
async def test_empty_list_prompt_with_history_trailing_tool_use_rejected():
    agent = _agent(messages=[_dangling_tool_use_message()])

    with pytest.raises(ValueError, match="toolUse"):
        await agent.invoke_async([], allow_resume=False)


@pytest.mark.asyncio
async def test_content_block_input_with_tool_use_rejected():
    agent = _agent()
    blocks = [{"toolUse": {"toolUseId": "t-1", "name": "echo", "input": {"value": "injected"}}}]

    with pytest.raises(ValueError, match="toolUse"):
        await agent.invoke_async(blocks, allow_resume=False)


@pytest.mark.asyncio
async def test_checkpoint_resume_block_rejected():
    agent = Agent(model=MockedModelProvider([FINAL_RESPONSE]), checkpointing=True)

    with pytest.raises(ValueError, match="checkpointResume"):
        await agent.invoke_async({"checkpointResume": {"checkpoint": {}}}, allow_resume=False)

    # The checkpoint must not be consumed into agent state before rejection.
    assert agent._checkpoint is None


def test_sync_call_rejects_trailing_tool_use():
    agent = _agent()

    with pytest.raises(ValueError, match="toolUse"):
        agent([_dangling_tool_use_message()], allow_resume=False)


@pytest.mark.asyncio
async def test_stream_async_rejects_trailing_tool_use():
    agent = _agent()

    with pytest.raises(ValueError, match="toolUse"):
        async for _ in agent.stream_async([_dangling_tool_use_message()], allow_resume=False):
            pass


# --- Allowed paths (guard must not reject normal input) ---


@pytest.mark.asyncio
async def test_plain_string_prompt_allowed():
    agent = _agent()

    result = await agent.invoke_async("hello", allow_resume=False)

    assert result.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_completed_tool_pairs_in_messages_input_allowed():
    agent = _agent()
    history: Messages = [
        {"role": "user", "content": [{"text": "run echo"}]},
        _dangling_tool_use_message("ok"),
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "t-1", "status": "success", "content": [{"text": "ok"}]}}],
        },
        {"role": "user", "content": [{"text": "now summarize"}]},
    ]

    result = await agent.invoke_async(history, allow_resume=False)

    assert result.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_string_prompt_with_history_trailing_tool_use_allowed():
    """A real prompt patches the dangling toolUse with a synthetic toolResult and invokes the model."""
    agent = _agent(messages=[_dangling_tool_use_message()])

    result = await agent.invoke_async("continue", allow_resume=False)

    assert result.stop_reason == "end_turn"
    # The dangling toolUse was patched, not executed.
    assert any("toolResult" in block for block in agent.messages[1]["content"])


@pytest.mark.asyncio
async def test_none_prompt_with_clean_history_allowed():
    agent = _agent(messages=[{"role": "user", "content": [{"text": "hello"}]}])

    result = await agent.invoke_async(None, allow_resume=False)

    assert result.stop_reason == "end_turn"


# --- Default behavior preserved ---


@pytest.mark.asyncio
async def test_default_still_resumes_trailing_tool_use():
    """Without allow_resume=False, a trailing toolUse still executes directly (existing behavior)."""
    tool_use_then_final = [FINAL_RESPONSE]
    agent = _agent(messages=None, responses=tool_use_then_final)

    result = await agent.invoke_async(
        [
            {"role": "user", "content": [{"text": "hi"}]},
            _dangling_tool_use_message("direct"),
        ]
    )

    # The tool executed directly: a toolResult for t-1 exists in history.
    tool_results = [
        block["toolResult"] for message in agent.messages for block in message["content"] if "toolResult" in block
    ]
    assert any(r["toolUseId"] == "t-1" for r in tool_results)
    assert result.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_explicit_allow_resume_true_resumes():
    agent = _agent(responses=[FINAL_RESPONSE])

    result = await agent.invoke_async([_dangling_tool_use_message("direct")], allow_resume=True)

    tool_results = [
        block["toolResult"] for message in agent.messages for block in message["content"] if "toolResult" in block
    ]
    assert any(r["toolUseId"] == "t-1" for r in tool_results)
    assert result.stop_reason == "end_turn"


# --- Interrupt state ---


@pytest.mark.asyncio
async def test_interrupted_agent_rejected():
    agent = _agent()
    agent._interrupt_state.activate()

    with pytest.raises(ValueError, match="interrupt state"):
        await agent.invoke_async([{"interruptResponse": {"interruptId": "x", "response": "y"}}], allow_resume=False)
