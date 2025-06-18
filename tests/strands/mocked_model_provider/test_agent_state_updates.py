from strands.agent.agent import Agent
from strands.tools.decorator import tool
from strands.types.content import Messages

from .mocked_model_provider import MockedModelProvider


@tool
def update_state(agent: Agent):
    agent.state.set("hello", "world")


def test_agent_state_update_from_tool():
    agent_messages: Messages = [
        {
            "role": "assistant",
            "content": [{"toolUse": {"name": "update_state", "toolUseId": "123", "input": {}}}],
        },
        {"role": "assistant", "content": [{"text": "I invoked a tool!"}]},
    ]
    mocked_model_provider = MockedModelProvider(agent_messages)

    agent = Agent(model=mocked_model_provider, tools=[update_state])

    assert agent.state.get("hello") is None

    agent("Invoke Mocked!")

    assert agent.state.get("hello") == "world"
