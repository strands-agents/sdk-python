import unittest.mock
from unittest.mock import call

import pytest
from pydantic import BaseModel

import strands
from strands import Agent
from strands.experimental.hooks import AgentInitializedEvent, EndRequestEvent, StartRequestEvent
from strands.types.content import Messages
from tests.fixtures.mock_hook_provider import MockHookProvider
from tests.fixtures.mocked_model_provider import MockedModelProvider


@pytest.fixture
def hook_provider():
    return MockHookProvider([AgentInitializedEvent, StartRequestEvent, EndRequestEvent])


@pytest.fixture
def agent_tool():
    @strands.tools.tool(name="tool_decorated")
    def reverse(random_string: str) -> str:
        return random_string[::-1]

    return reverse


@pytest.fixture
def tool_use(agent_tool):
    return {"name": agent_tool.tool_name, "toolUseId": "123", "input": {"random_string": "I invoked a tool!"}}


@pytest.fixture
def mock_model(tool_use):
    agent_messages: Messages = [
        {
            "role": "assistant",
            "content": [{"toolUse": tool_use}],
        },
        {"role": "assistant", "content": [{"text": "I invoked a tool!"}]},
    ]
    return MockedModelProvider(agent_messages)


@pytest.fixture
def agent(
    mock_model,
    hook_provider,
    agent_tool,
):
    agent = Agent(
        model=mock_model,
        system_prompt="You are a helpful assistant.",
        callback_handler=None,
        tools=[agent_tool],
    )

    # for now, hooks are private
    agent._hooks.add_hook(hook_provider)

    return agent


@pytest.fixture
def user():
    class User(BaseModel):
        name: str
        age: int

    return User(name="Jane Doe", age=30)


@unittest.mock.patch("strands.experimental.hooks.registry.HookRegistry.invoke_callbacks")
def test_agent__init__hooks(mock_invoke_callbacks):
    """Verify that the AgentInitializedEvent is emitted on Agent construction."""
    agent = Agent()

    # Verify AgentInitialized event was invoked
    mock_invoke_callbacks.assert_called_once()
    assert mock_invoke_callbacks.call_args == call(AgentInitializedEvent(agent=agent))


def test_agent__call__hooks(agent, hook_provider, agent_tool, tool_use):
    """Verify that the correct hook events are emitted as part of __call__."""

    agent("test message")

    events = hook_provider.get_events()
    assert len(events) == 2

    assert events.popleft() == StartRequestEvent(agent=agent)
    assert events.popleft() == EndRequestEvent(agent=agent)


@pytest.mark.asyncio
async def test_agent_stream_async_hooks(agent, hook_provider, agent_tool, tool_use):
    """Verify that the correct hook events are emitted as part of stream_async."""
    iterator = agent.stream_async("test message")
    await anext(iterator)
    assert hook_provider.events_received == [StartRequestEvent(agent=agent)]

    # iterate the rest
    async for _ in iterator:
        pass

    events = hook_provider.get_events()
    assert len(events) == 2

    assert events.popleft() == StartRequestEvent(agent=agent)
    assert events.popleft() == EndRequestEvent(agent=agent)


def test_agent_structured_output_hooks(agent, hook_provider, user, agenerator):
    """Verify that the correct hook events are emitted as part of structured_output."""

    agent.model.structured_output = unittest.mock.Mock(return_value=agenerator([{"output": user}]))
    agent.structured_output(type(user), "example prompt")

    assert hook_provider.events_received == [StartRequestEvent(agent=agent), EndRequestEvent(agent=agent)]


@pytest.mark.asyncio
async def test_agent_structured_async_output_hooks(agent, hook_provider, user, agenerator):
    """Verify that the correct hook events are emitted as part of structured_output_async."""

    agent.model.structured_output = unittest.mock.Mock(return_value=agenerator([{"output": user}]))
    await agent.structured_output_async(type(user), "example prompt")

    assert hook_provider.events_received == [StartRequestEvent(agent=agent), EndRequestEvent(agent=agent)]
