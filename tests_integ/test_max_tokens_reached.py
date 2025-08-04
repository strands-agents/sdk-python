import logging

import pytest

from strands import Agent, tool
from strands.experimental.hooks.providers.correct_tool_use_hook_provider import CorrectToolUseHookProvider
from strands.models.bedrock import BedrockModel
from strands.types.exceptions import MaxTokensReachedException

logger = logging.getLogger(__name__)


@tool
def story_tool(story: str) -> str:
    """
    Tool that writes a story that is minimum 50,000 lines long.
    """
    return story


def test_context_window_overflow():
    model = BedrockModel(max_tokens=100)
    agent = Agent(model=model, tools=[story_tool])

    with pytest.raises(MaxTokensReachedException):
        agent("Tell me a story!")

    assert len(agent.messages) == 1


def test_max_tokens_reached_with_hook_provider():
    """Test that MaxTokensReachedException can be handled by a hook provider."""
    model = BedrockModel(max_tokens=100)
    hook_provider = CorrectToolUseHookProvider()
    agent = Agent(model=model, tools=[story_tool], hooks=[hook_provider])

    # This should NOT raise an exception because the hook handles it
    agent("Tell me a story!")

    # Validate that at least one message contains the incomplete tool use error message
    expected_text = "tool use was incomplete due to maximum token limits being reached"
    all_text_content = [
        content_block["text"]
        for message in agent.messages
        for content_block in message.get("content", [])
        if "text" in content_block
    ]

    assert any(expected_text in text for text in all_text_content), (
        f"Expected to find message containing '{expected_text}' in agent messages"
    )
