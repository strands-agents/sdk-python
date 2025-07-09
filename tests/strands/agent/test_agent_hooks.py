from unittest.mock import ANY, Mock, call, patch

import pytest
from pydantic import BaseModel

import strands
from strands import Agent
from strands.experimental.hooks import (
    AfterModelInvocationEvent,
    AfterToolInvocationEvent,
    AgentInitializedEvent,
    BeforeModelInvocationEvent,
    BeforeToolInvocationEvent,
    EndRequestEvent,
    StartRequestEvent,
)
from strands.types.content import Messages
from tests.fixtures.mock_hook_provider import MockHookProvider
from tests.fixtures.mocked_model_provider import MockedModelProvider


@pytest.fixture
def hook_provider():
    return MockHookProvider(
        [
            AgentInitializedEvent,
            StartRequestEvent,
            EndRequestEvent,
            AfterToolInvocationEvent,
            BeforeToolInvocationEvent,
            BeforeModelInvocationEvent,
            AfterModelInvocationEvent,
        ]
    )


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
def tools_config(agent):
    return agent.tool_config["tools"]


@pytest.fixture
def user():
    class User(BaseModel):
        name: str
        age: int

    return User(name="Jane Doe", age=30)


@patch("strands.experimental.hooks.registry.HookRegistry.invoke_callbacks")
def test_agent__init__hooks(mock_invoke_callbacks):
    """Verify that the AgentInitializedEvent is emitted on Agent construction."""
    agent = Agent()

    # Verify AgentInitialized event was invoked
    mock_invoke_callbacks.assert_called_once()
    assert mock_invoke_callbacks.call_args == call(AgentInitializedEvent(agent=agent))


def test_agent__call__hooks(agent, hook_provider, agent_tool, mock_model, tools_config, tool_use):
    """Verify that the correct hook events are emitted as part of __call__."""

    agent("test message")

    length, events = hook_provider.get_events()

    assert next(events) == StartRequestEvent(agent=agent)
    assert next(events) == BeforeModelInvocationEvent(agent=agent, tool_configs=tools_config)
    assert next(events) == AfterModelInvocationEvent(
        agent=agent,
        tool_configs=tools_config,
        stop_data=AfterModelInvocationEvent.ModelStopResponse(
            message={
                "content": [{"toolUse": tool_use}],
                "role": "assistant",
            },
            stop_reason="tool_use",
        ),
        exception=None,
    )
    assert next(events) == BeforeToolInvocationEvent(
        agent=agent, selected_tool=agent_tool, tool_use=tool_use, kwargs=ANY
    )
    assert next(events) == AfterToolInvocationEvent(
        agent=agent,
        selected_tool=agent_tool,
        tool_use=tool_use,
        kwargs=ANY,
        result={"content": [{"text": "!loot a dekovni I"}], "status": "success", "toolUseId": "123"},
    )
    assert next(events) == BeforeModelInvocationEvent(agent=agent, tool_configs=tools_config)
    assert next(events) == AfterModelInvocationEvent(
        agent=agent,
        tool_configs=tools_config,
        stop_data=AfterModelInvocationEvent.ModelStopResponse(
            message=mock_model.agent_responses[1],
            stop_reason="end_turn",
        ),
        exception=None,
    )
    assert next(events) == EndRequestEvent(agent=agent)


@pytest.mark.asyncio
async def test_agent_stream_async_hooks(agent, hook_provider, agent_tool, mock_model, tools_config, tool_use):
    """Verify that the correct hook events are emitted as part of stream_async."""
    iterator = agent.stream_async("test message")
    await anext(iterator)
    assert hook_provider.events_received == [StartRequestEvent(agent=agent)]

    # iterate the rest
    async for _ in iterator:
        pass

    length, events = hook_provider.get_events()

    assert length == 8

    assert next(events) == StartRequestEvent(agent=agent)
    assert next(events) == BeforeModelInvocationEvent(agent=agent, tool_configs=tools_config)
    assert next(events) == AfterModelInvocationEvent(
        agent=agent,
        tool_configs=tools_config,
        stop_data=AfterModelInvocationEvent.ModelStopResponse(
            message={
                "content": [{"toolUse": tool_use}],
                "role": "assistant",
            },
            stop_reason="tool_use",
        ),
        exception=None,
    )
    assert next(events) == BeforeToolInvocationEvent(
        agent=agent, selected_tool=agent_tool, tool_use=tool_use, kwargs=ANY
    )
    assert next(events) == AfterToolInvocationEvent(
        agent=agent,
        selected_tool=agent_tool,
        tool_use=tool_use,
        kwargs=ANY,
        result={"content": [{"text": "!loot a dekovni I"}], "status": "success", "toolUseId": "123"},
    )
    assert next(events) == BeforeModelInvocationEvent(agent=agent, tool_configs=tools_config)
    assert next(events) == AfterModelInvocationEvent(
        agent=agent,
        tool_configs=tools_config,
        stop_data=AfterModelInvocationEvent.ModelStopResponse(
            message=mock_model.agent_responses[1],
            stop_reason="end_turn",
        ),
        exception=None,
    )
    assert next(events) == EndRequestEvent(agent=agent)


def test_agent_structured_output_hooks(agent, hook_provider, user, agenerator):
    """Verify that the correct hook events are emitted as part of structured_output."""

    agent.model.structured_output = Mock(return_value=agenerator([{"output": user}]))
    agent.structured_output(type(user), "example prompt")

    assert hook_provider.events_received == [StartRequestEvent(agent=agent), EndRequestEvent(agent=agent)]


@pytest.mark.asyncio
async def test_agent_structured_async_output_hooks(agent, hook_provider, user, agenerator):
    """Verify that the correct hook events are emitted as part of structured_output_async."""

    agent.model.structured_output = Mock(return_value=agenerator([{"output": user}]))
    await agent.structured_output_async(type(user), "example prompt")

    assert hook_provider.events_received == [StartRequestEvent(agent=agent), EndRequestEvent(agent=agent)]
