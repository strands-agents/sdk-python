from unittest.mock import ANY, Mock, call, patch

import pytest
from pydantic import BaseModel

import strands
from strands import Agent
from strands.experimental.hooks import (
    AfterInvocationEvent,
    AfterModelInvocationEvent,
    AfterToolInvocationEvent,
    AgentInitializedEvent,
    BeforeInvocationEvent,
    BeforeModelInvocationEvent,
    BeforeToolInvocationEvent,
    MessageAddedEvent,
    get_registry,
)
from strands.types.content import Messages
from strands.types.tools import ToolResult, ToolUse
from tests.fixtures.mock_hook_provider import MockHookProvider
from tests.fixtures.mocked_model_provider import MockedModelProvider


@pytest.fixture
def hook_provider():
    return MockHookProvider(
        [
            AgentInitializedEvent,
            BeforeInvocationEvent,
            AfterInvocationEvent,
            AfterToolInvocationEvent,
            BeforeToolInvocationEvent,
            BeforeModelInvocationEvent,
            AfterModelInvocationEvent,
            MessageAddedEvent,
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

    hooks = get_registry(agent)
    hooks.add_hook(hook_provider)

    def assert_message_is_last_message_added(event: MessageAddedEvent):
        assert event.agent.messages[-1] == event.message

    hooks.add_callback(MessageAddedEvent, assert_message_is_last_message_added)

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


def test_agent_tool_call(agent, hook_provider, agent_tool):
    agent.tool.tool_decorated(random_string="a string")

    length, events = hook_provider.get_events()

    tool_use: ToolUse = {"input": {"random_string": "a string"}, "name": "tool_decorated", "toolUseId": ANY}
    result: ToolResult = {"content": [{"text": "gnirts a"}], "status": "success", "toolUseId": ANY}

    assert length == 6

    assert next(events) == BeforeToolInvocationEvent(
        agent=agent, selected_tool=agent_tool, tool_use=tool_use, kwargs=ANY
    )
    assert next(events) == AfterToolInvocationEvent(
        agent=agent,
        selected_tool=agent_tool,
        tool_use=tool_use,
        kwargs=ANY,
        result=result,
    )
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[0])
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[1])
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[2])
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[3])

    assert len(agent.messages) == 4


def test_agent__call__hooks(agent, hook_provider, agent_tool, mock_model, tool_use):
    """Verify that the correct hook events are emitted as part of __call__."""

    agent("test message")

    length, events = hook_provider.get_events()

    assert length == 12

    assert next(events) == BeforeInvocationEvent(agent=agent)
    assert next(events) == MessageAddedEvent(
        agent=agent,
        message=agent.messages[0],
    )
    assert next(events) == BeforeModelInvocationEvent(agent=agent)
    assert next(events) == AfterModelInvocationEvent(
        agent=agent,
        stop_response=AfterModelInvocationEvent.ModelStopResponse(
            message={
                "content": [{"toolUse": tool_use}],
                "role": "assistant",
            },
            stop_reason="tool_use",
        ),
        exception=None,
    )

    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[1])
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
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[2])
    assert next(events) == BeforeModelInvocationEvent(agent=agent)
    assert next(events) == AfterModelInvocationEvent(
        agent=agent,
        stop_response=AfterModelInvocationEvent.ModelStopResponse(
            message=mock_model.agent_responses[1],
            stop_reason="end_turn",
        ),
        exception=None,
    )
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[3])

    assert next(events) == AfterInvocationEvent(agent=agent)

    assert len(agent.messages) == 4


@pytest.mark.asyncio
async def test_agent_stream_async_hooks(agent, hook_provider, agent_tool, mock_model, tool_use, agenerator):
    """Verify that the correct hook events are emitted as part of stream_async."""
    iterator = agent.stream_async("test message")
    await anext(iterator)
    assert hook_provider.events_received == [BeforeInvocationEvent(agent=agent)]

    # iterate the rest
    async for _ in iterator:
        pass

    length, events = hook_provider.get_events()

    assert length == 12

    assert next(events) == BeforeInvocationEvent(agent=agent)
    assert next(events) == MessageAddedEvent(
        agent=agent,
        message=agent.messages[0],
    )
    assert next(events) == BeforeModelInvocationEvent(agent=agent)
    assert next(events) == AfterModelInvocationEvent(
        agent=agent,
        stop_response=AfterModelInvocationEvent.ModelStopResponse(
            message={
                "content": [{"toolUse": tool_use}],
                "role": "assistant",
            },
            stop_reason="tool_use",
        ),
        exception=None,
    )

    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[1])
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
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[2])
    assert next(events) == BeforeModelInvocationEvent(agent=agent)
    assert next(events) == AfterModelInvocationEvent(
        agent=agent,
        stop_response=AfterModelInvocationEvent.ModelStopResponse(
            message=mock_model.agent_responses[1],
            stop_reason="end_turn",
        ),
        exception=None,
    )
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[3])

    assert next(events) == AfterInvocationEvent(agent=agent)

    assert len(agent.messages) == 4


def test_agent_structured_output_hooks(agent, hook_provider, user, agenerator):
    """Verify that the correct hook events are emitted as part of structured_output."""

    agent.model.structured_output = Mock(return_value=agenerator([{"output": user}]))
    agent.structured_output(type(user), "example prompt")

    length, events = hook_provider.get_events()

    assert length == 3

    assert next(events) == BeforeInvocationEvent(agent=agent)
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[0])
    assert next(events) == AfterInvocationEvent(agent=agent)

    assert len(agent.messages) == 1


@pytest.mark.asyncio
async def test_agent_structured_async_output_hooks(agent, hook_provider, user, agenerator):
    """Verify that the correct hook events are emitted as part of structured_output_async."""

    agent.model.structured_output = Mock(return_value=agenerator([{"output": user}]))
    await agent.structured_output_async(type(user), "example prompt")

    length, events = hook_provider.get_events()

    assert length == 3

    assert next(events) == BeforeInvocationEvent(agent=agent)
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[0])
    assert next(events) == AfterInvocationEvent(agent=agent)

    assert len(agent.messages) == 1
