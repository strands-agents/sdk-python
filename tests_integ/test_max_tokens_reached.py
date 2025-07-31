import pytest

from strands import Agent, tool
from strands.models.bedrock import BedrockModel
from strands.types.exceptions import MaxTokensReachedException


@tool
def story_tool(story: str) -> str:
    return story


def test_context_window_overflow():
    model = BedrockModel(max_tokens=1)
    agent = Agent(model=model, tools=[story_tool])

    with pytest.raises(MaxTokensReachedException):
        agent("Tell me a story!")
