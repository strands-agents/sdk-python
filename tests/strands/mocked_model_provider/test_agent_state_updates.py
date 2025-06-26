from strands.agent.agent import Agent
from strands.tools.decorator import tool
from strands.types.content import Messages

from .mocked_model_provider import MockedModelProvider


@tool
def update_state(agent: Agent):
    agent.state.set("hello", "world")
    agent.state.set("foo", "baz")


def test_agent_state_update_from_tool():
    agent_messages: Messages = [
        {
            "role": "assistant",
            "content": [{"toolUse": {"name": "update_state", "toolUseId": "123", "input": {}}}],
        },
        {"role": "assistant", "content": [{"text": "I invoked a tool!"}]},
    ]
    mocked_model_provider = MockedModelProvider(agent_messages)

    agent = Agent(
        model=mocked_model_provider,
        tools=[update_state],
        state={"foo": "bar"},
    )

    assert agent.state.get("hello") is None
    assert agent.state.get("foo") == "bar"

    agent("Invoke Mocked!")

    assert agent.state.get("hello") == "world"
    assert agent.state.get("foo") == "baz"
