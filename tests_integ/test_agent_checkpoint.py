"""Integration tests for agent checkpointing with Amazon Bedrock.

These tests exercise the end-to-end durability contract: an agent with
``checkpointing=True`` pauses at ReAct cycle boundaries, returns an
``AgentResult`` with ``stop_reason="checkpoint"`` and a populated
``checkpoint`` field, and a fresh ``Agent`` instance resumes from the
persisted checkpoint through a ``checkpointResume`` content block.

Requires valid AWS credentials and may incur API costs.

To run:
    hatch run test-integ tests_integ/test_agent_checkpoint.py
"""

import json
import os

import pytest

from strands import Agent, tool
from strands.experimental.checkpoint import Checkpoint
from strands.models import BedrockModel

# Skip all tests if no AWS region is configured (boto3 accepts either env var)
pytestmark = [
    pytest.mark.skipif(
        not (os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")),
        reason="AWS credentials not available",
    ),
    pytest.mark.asyncio,
]


MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"


def _build_agent(tools: list) -> Agent:
    """Build a checkpointing agent with a deterministic tool-using system prompt."""
    return Agent(
        model=BedrockModel(model_id=MODEL_ID),
        tools=tools,
        system_prompt=(
            "You are a helpful assistant. When a user asks a factual question, "
            "you MUST call the provided tools to answer. Do not answer from memory."
        ),
        checkpointing=True,
    )


async def _drive_to_completion(
    tools: list,
    first_prompt: str,
    max_resumes: int = 10,
) -> tuple[Agent, list[Checkpoint]]:
    """Drive a checkpointing agent to end_turn across fresh Agent instances.

    Each time the agent pauses on a checkpoint, we serialize the checkpoint,
    discard the Agent, build a fresh one, and resume. Returns the final Agent
    (so callers can inspect ``messages``) and the ordered list of checkpoints
    that were observed along the way.
    """
    agent = _build_agent(tools)
    result = await agent.invoke_async(first_prompt)

    checkpoints: list[Checkpoint] = []
    resumes = 0
    while result.stop_reason == "checkpoint":
        assert result.checkpoint is not None, "checkpoint field must be populated on pause"
        checkpoints.append(result.checkpoint)

        # Serialize through JSON to prove the checkpoint is durable across a
        # process boundary (simulated here by round-tripping through bytes).
        persisted = json.loads(json.dumps(result.checkpoint.to_dict()))

        # Discard the Agent entirely. A fresh instance resumes from scratch,
        # holding no in-memory state from the previous invocation.
        del agent
        agent = _build_agent(tools)

        result = await agent.invoke_async([{"checkpointResume": {"checkpoint": persisted}}])

        resumes += 1
        if resumes > max_resumes:
            raise AssertionError(f"exceeded max_resumes={max_resumes} without reaching end_turn")

    assert result.stop_reason == "end_turn", f"unexpected terminal stop_reason: {result.stop_reason}"
    return agent, checkpoints


async def test_checkpoint_roundtrip_completes_through_fresh_agent():
    """Pause at a cycle boundary, resume on a fresh Agent, reach end_turn.

    Uses a simple single-tool prompt so the agent is forced through at least
    one after_model + after_tools pair before the final end_turn cycle.
    """

    @tool
    def get_color_of_sky() -> str:
        """Return the color of the sky."""
        return "blue"

    final_agent, checkpoints = await _drive_to_completion(
        tools=[get_color_of_sky],
        first_prompt="What color is the sky? Use the get_color_of_sky tool.",
    )

    # At least one checkpoint was emitted on the way to completion.
    assert len(checkpoints) >= 1

    # All checkpoints are at one of the two defined boundaries.
    assert all(cp.position in ("after_model", "after_tools") for cp in checkpoints)

    # Cycle indices are non-decreasing across the run.
    cycle_indices = [cp.cycle_index for cp in checkpoints]
    assert cycle_indices == sorted(cycle_indices), f"cycle indices not monotonic: {cycle_indices}"

    # The final agent's message history contains the tool result.
    tool_result_texts = [
        block["toolResult"]["content"][0]["text"]
        for message in final_agent.messages
        for block in message["content"]
        if "toolResult" in block
    ]
    assert "blue" in tool_result_texts

    # The assistant's final message references the tool output.
    final_message_text = json.dumps(final_agent.messages[-1]).lower()
    assert "blue" in final_message_text


async def test_checkpoint_survives_process_boundary_no_tool_rerun():
    """The durability invariant: completed tool calls are not re-run on resume.

    Uses a module-level counter that each tool increments on every call. After
    driving the agent through multiple resume cycles, each tool must have been
    called exactly once — proof that resuming from ``after_tools`` skips the
    tools that already ran rather than re-executing them.
    """
    call_counts = {"time": 0, "day": 0, "weather": 0}

    @tool
    def get_time() -> str:
        """Return the current time."""
        call_counts["time"] += 1
        return "12:01"

    @tool
    def get_day() -> str:
        """Return the current day of the week."""
        call_counts["day"] += 1
        return "monday"

    @tool
    def get_weather() -> str:
        """Return the current weather."""
        call_counts["weather"] += 1
        return "sunny"

    final_agent, checkpoints = await _drive_to_completion(
        tools=[get_time, get_day, get_weather],
        first_prompt=("What is the time, the day, and the weather? Use the get_time, get_day, and get_weather tools."),
    )

    # Each tool ran exactly once across the entire durable run. Resuming from
    # a checkpoint must not re-execute tools that already completed.
    assert call_counts == {"time": 1, "day": 1, "weather": 1}, (
        f"tools were re-executed on resume — counts: {call_counts}"
    )

    # At least one after_tools checkpoint was observed (the scenario the
    # durability invariant protects).
    assert any(cp.position == "after_tools" for cp in checkpoints), (
        f"no after_tools checkpoint observed: {[cp.position for cp in checkpoints]}"
    )

    # Final message references all three tool outputs.
    final_message_text = json.dumps(final_agent.messages[-1]).lower()
    assert all(s in final_message_text for s in ["12:01", "monday", "sunny"])


async def test_checkpoint_resume_preserves_conversation_history():
    """After resume, agent.messages contains the full pre-crash conversation.

    The snapshot-based state transfer must restore not only the pending tool
    results but the entire message history (user prompt, assistant tool_use,
    tool results). Otherwise the resumed model call would be missing context.
    """

    @tool
    def get_favorite_number() -> int:
        """Return the user's favorite number."""
        return 42

    final_agent, _ = await _drive_to_completion(
        tools=[get_favorite_number],
        first_prompt="What is my favorite number? Use the get_favorite_number tool.",
    )

    # The user's original prompt survived the full checkpoint/resume cycle.
    user_messages = [m for m in final_agent.messages if m["role"] == "user"]
    first_user_message_text = json.dumps(user_messages[0]).lower()
    assert "favorite number" in first_user_message_text

    # The assistant reached a terminal response.
    assert final_agent.messages[-1]["role"] == "assistant"
    assert "42" in json.dumps(final_agent.messages[-1])
