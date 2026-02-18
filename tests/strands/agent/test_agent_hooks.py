from unittest.mock import ANY, AsyncMock, MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

import strands
from strands import Agent
from strands.hooks import (
    AfterInvocationEvent,
    AfterModelCallEvent,
    AfterToolCallEvent,
    AgentInitializedEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    MessageAddedEvent,
)
from strands.types.content import Messages
from strands.types.exceptions import ModelThrottledException
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
            AfterToolCallEvent,
            BeforeToolCallEvent,
            BeforeModelCallEvent,
            AfterModelCallEvent,
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

    hooks = agent.hooks
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


@pytest.fixture
def mock_sleep():
    with patch.object(strands.event_loop._retry.asyncio, "sleep", new_callable=AsyncMock) as mock:
        yield mock


def test_agent__init__hooks():
    """Verify that the AgentInitializedEvent is emitted on Agent construction."""
    hook_provider = MockHookProvider(event_types=[AgentInitializedEvent])
    agent = Agent(hooks=[hook_provider])

    length, events = hook_provider.get_events()

    assert length == 1

    assert next(events) == AgentInitializedEvent(agent=agent)


def test_agent_tool_call(agent, hook_provider, agent_tool):
    agent.tool.tool_decorated(random_string="a string")

    length, events = hook_provider.get_events()

    tool_use: ToolUse = {"input": {"random_string": "a string"}, "name": "tool_decorated", "toolUseId": ANY}
    result: ToolResult = {"content": [{"text": "gnirts a"}], "status": "success", "toolUseId": ANY}

    assert length == 6

    assert next(events) == BeforeToolCallEvent(
        agent=agent,
        selected_tool=agent_tool,
        tool_use=tool_use,
        invocation_state=ANY,
    )
    assert next(events) == AfterToolCallEvent(
        agent=agent,
        selected_tool=agent_tool,
        tool_use=tool_use,
        invocation_state=ANY,
        result=result,
    )
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[0])
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[1])
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[2])
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[3])

    assert len(agent.messages) == 4


def test_agent__call__hooks(agent, hook_provider, agent_tool, mock_model, tool_use):
    """Verify that the correct hook events are emitted as part of __call__."""

    result = agent("test message")

    length, events = hook_provider.get_events()

    assert length == 12

    assert next(events) == BeforeInvocationEvent(agent=agent, invocation_state=ANY, messages=agent.messages[0:1])
    assert next(events) == MessageAddedEvent(
        agent=agent,
        message=agent.messages[0],
    )
    assert next(events) == BeforeModelCallEvent(agent=agent, invocation_state=ANY)
    assert next(events) == AfterModelCallEvent(
        agent=agent,
        invocation_state=ANY,
        stop_response=AfterModelCallEvent.ModelStopResponse(
            message={
                "content": [{"toolUse": tool_use}],
                "role": "assistant",
            },
            stop_reason="tool_use",
        ),
        exception=None,
    )

    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[1])
    assert next(events) == BeforeToolCallEvent(
        agent=agent,
        selected_tool=agent_tool,
        tool_use=tool_use,
        invocation_state=ANY,
    )
    assert next(events) == AfterToolCallEvent(
        agent=agent,
        selected_tool=agent_tool,
        tool_use=tool_use,
        invocation_state=ANY,
        result={"content": [{"text": "!loot a dekovni I"}], "status": "success", "toolUseId": "123"},
    )
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[2])
    assert next(events) == BeforeModelCallEvent(agent=agent, invocation_state=ANY)
    assert next(events) == AfterModelCallEvent(
        agent=agent,
        invocation_state=ANY,
        stop_response=AfterModelCallEvent.ModelStopResponse(
            message=mock_model.agent_responses[1],
            stop_reason="end_turn",
        ),
        exception=None,
    )
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[3])

    assert next(events) == AfterInvocationEvent(agent=agent, invocation_state=ANY, result=result)

    assert len(agent.messages) == 4


@pytest.mark.asyncio
async def test_agent_stream_async_hooks(agent, hook_provider, agent_tool, mock_model, tool_use, agenerator):
    """Verify that the correct hook events are emitted as part of stream_async."""
    iterator = agent.stream_async("test message")
    await anext(iterator)

    # Verify first event is BeforeInvocationEvent with invocation_state and messages
    assert len(hook_provider.events_received) == 1
    assert hook_provider.events_received[0].invocation_state is not None
    assert hook_provider.events_received[0].messages is not None
    assert hook_provider.events_received[0].messages[0]["role"] == "user"

    # iterate the rest
    result = None
    async for item in iterator:
        if "result" in item:
            result = item["result"]

    length, events = hook_provider.get_events()

    assert length == 12

    assert next(events) == BeforeInvocationEvent(agent=agent, invocation_state=ANY, messages=agent.messages[0:1])
    assert next(events) == MessageAddedEvent(
        agent=agent,
        message=agent.messages[0],
    )
    assert next(events) == BeforeModelCallEvent(agent=agent, invocation_state=ANY)
    assert next(events) == AfterModelCallEvent(
        agent=agent,
        invocation_state=ANY,
        stop_response=AfterModelCallEvent.ModelStopResponse(
            message={
                "content": [{"toolUse": tool_use}],
                "role": "assistant",
            },
            stop_reason="tool_use",
        ),
        exception=None,
    )

    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[1])
    assert next(events) == BeforeToolCallEvent(
        agent=agent,
        selected_tool=agent_tool,
        tool_use=tool_use,
        invocation_state=ANY,
    )
    assert next(events) == AfterToolCallEvent(
        agent=agent,
        selected_tool=agent_tool,
        tool_use=tool_use,
        invocation_state=ANY,
        result={"content": [{"text": "!loot a dekovni I"}], "status": "success", "toolUseId": "123"},
    )
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[2])
    assert next(events) == BeforeModelCallEvent(agent=agent, invocation_state=ANY)
    assert next(events) == AfterModelCallEvent(
        agent=agent,
        invocation_state=ANY,
        stop_response=AfterModelCallEvent.ModelStopResponse(
            message=mock_model.agent_responses[1],
            stop_reason="end_turn",
        ),
        exception=None,
    )
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[3])

    assert next(events) == AfterInvocationEvent(agent=agent, invocation_state=ANY, result=result)

    assert len(agent.messages) == 4


@pytest.mark.filterwarnings("ignore:Agent.structured_output method is deprecated:DeprecationWarning")
def test_agent_structured_output_hooks(agent, hook_provider, user, agenerator):
    """Verify that the correct hook events are emitted as part of structured_output."""

    agent.model.structured_output = Mock(return_value=agenerator([{"output": user}]))
    agent.structured_output(type(user), "example prompt")

    length, events = hook_provider.get_events()

    assert length == 2

    assert next(events) == BeforeInvocationEvent(agent=agent, invocation_state=ANY)
    assert next(events) == AfterInvocationEvent(agent=agent, invocation_state=ANY)

    assert len(agent.messages) == 0  # no new messages added


@pytest.mark.filterwarnings("ignore:Agent.structured_output_async method is deprecated:DeprecationWarning")
@pytest.mark.asyncio
async def test_agent_structured_async_output_hooks(agent, hook_provider, user, agenerator):
    """Verify that the correct hook events are emitted as part of structured_output_async."""

    agent.model.structured_output = Mock(return_value=agenerator([{"output": user}]))
    await agent.structured_output_async(type(user), "example prompt")

    length, events = hook_provider.get_events()

    assert length == 2

    assert next(events) == BeforeInvocationEvent(agent=agent, invocation_state=ANY)
    assert next(events) == AfterInvocationEvent(agent=agent, invocation_state=ANY)

    assert len(agent.messages) == 0  # no new messages added


@pytest.mark.asyncio
async def test_hook_retry_on_successful_call():
    """Test that hooks can retry even on successful model calls based on response content."""

    mock_provider = MockedModelProvider(
        [
            {
                "role": "assistant",
                "content": [{"text": "Short"}],
            },
            {
                "role": "assistant",
                "content": [{"text": "This is a much longer and more detailed response"}],
            },
        ]
    )

    # Hook that retries if response is too short
    class MinLengthRetryHook:
        def __init__(self, min_length=10):
            self.min_length = min_length
            self.call_count = 0

        def register_hooks(self, registry):
            registry.add_callback(strands.hooks.AfterModelCallEvent, self.handle_after_model_call)

        async def handle_after_model_call(self, event):
            self.call_count += 1

            # Check successful responses for minimum length
            if event.stop_response:
                message = event.stop_response.message
                text_content = "".join(block.get("text", "") for block in message.get("content", []))

                if len(text_content) < self.min_length:
                    event.retry = True

    retry_hook = MinLengthRetryHook(min_length=10)
    agent = Agent(model=mock_provider, hooks=[retry_hook])

    result = agent("Generate a response")

    # Verify hook was called twice (once for short response, once for long)
    assert retry_hook.call_count == 2

    # Verify final result is the longer response
    assert result.message["content"][0]["text"] == "This is a much longer and more detailed response"


@pytest.mark.asyncio
async def test_hook_retry_on_exception_basic(alist, mock_sleep):
    """Test that hooks can retry model calls on exceptions."""

    class CustomException(Exception):
        pass

    model = MagicMock()
    model.stream.side_effect = [
        CustomException("First attempt fails"),
        MockedModelProvider(
            [
                {
                    "role": "assistant",
                    "content": [{"text": "Success after retry"}],
                },
            ]
        ).stream([]),
    ]

    # Hook that enables retry on CustomException
    class RetryHook:
        def __init__(self):
            self.after_model_call_count = 0

        def register_hooks(self, registry):
            registry.add_callback(strands.hooks.AfterModelCallEvent, self.handle_after_model_call)

        async def handle_after_model_call(self, event):
            self.after_model_call_count += 1
            if event.exception and isinstance(event.exception, CustomException):
                event.retry = True

    retry_hook = RetryHook()
    agent = Agent(model=model, hooks=[retry_hook])

    result = agent("Test retry")

    # Verify the hook was called twice (once for failure, once for success)
    assert retry_hook.after_model_call_count == 2
    assert result.stop_reason == "end_turn"
    assert result.message["content"][0]["text"] == "Success after retry"


@pytest.mark.asyncio
async def test_hook_retry_not_set_on_success(alist):
    """Test that model is not retried when hook doesn't set retry_model on success."""
    mock_provider = MockedModelProvider(
        [
            {
                "role": "assistant",
                "content": [{"text": "First successful response"}],
            },
        ]
    )

    # Hook that tries to set retry_model=True even on success
    class NoRetryHook:
        def __init__(self):
            self.call_count = 0

        def register_hooks(self, registry):
            registry.add_callback(strands.hooks.AfterModelCallEvent, self.handle_after_model_call)

        async def handle_after_model_call(self, event):
            self.call_count += 1
            # Try to set retry even on success
            # Don't set retry_model (leave it as False)

    retry_hook = NoRetryHook()
    agent = Agent(model=mock_provider, hooks=[retry_hook])

    result = agent("Test no retry when not set")

    # Should only be called once since retry_model was not set
    assert retry_hook.call_count == 1
    assert result.message["content"][0]["text"] == "First successful response"


@pytest.mark.asyncio
async def test_hook_retry_with_limit(alist, mock_sleep):
    """Test that hooks can control retry limits."""

    class CustomException(Exception):
        pass

    model = MagicMock()
    model.stream.side_effect = [
        CustomException("Attempt 1 fails"),
        CustomException("Attempt 2 fails"),
        CustomException("Attempt 3 fails"),
    ]

    # Hook that allows max 2 retries
    class LimitedRetryHook:
        def __init__(self, max_retries=2):
            self.max_retries = max_retries
            self.retry_count = 0
            self.call_count = 0

        def register_hooks(self, registry):
            registry.add_callback(strands.hooks.AfterModelCallEvent, self.handle_after_model_call)

        async def handle_after_model_call(self, event):
            self.call_count += 1
            if event.exception and isinstance(event.exception, CustomException):
                if self.retry_count < self.max_retries:
                    self.retry_count += 1
                    event.retry = True
                # else: let exception propagate

    retry_hook = LimitedRetryHook(max_retries=2)
    agent = Agent(model=model, hooks=[retry_hook])

    with pytest.raises(CustomException, match="Attempt 3 fails"):
        await agent("Test limited retries")

    # Should be called 3 times: initial + 2 retries
    assert retry_hook.call_count == 3
    assert retry_hook.retry_count == 2


@pytest.mark.asyncio
async def test_hook_retry_multiple_hooks(alist, mock_sleep):
    """Test that multiple hooks can modify retry_model and last one wins."""

    class CustomException(Exception):
        pass

    model = MagicMock()
    model.stream.side_effect = [
        CustomException("First attempt fails"),
        MockedModelProvider(
            [
                {
                    "role": "assistant",
                    "content": [{"text": "Success"}],
                },
            ]
        ).stream([]),
    ]

    async def retry_enabler(event: AfterModelCallEvent):
        if event.exception:
            event.retry = True

    async def another_retry_enabler(event: AfterModelCallEvent):
        if event.exception:
            event.retry = True

    agent = Agent(model=model)
    agent.hooks.add_callback(AfterModelCallEvent, retry_enabler)
    agent.hooks.add_callback(AfterModelCallEvent, another_retry_enabler)

    result = agent("Test multiple hooks")

    assert result.stop_reason == "end_turn"
    assert result.message["content"][0]["text"] == "Success"


@pytest.mark.asyncio
async def test_hook_retry_last_hook_wins(alist, mock_sleep):
    """Test that when multiple hooks set retry_model, the last-called hook wins.

    Note: AfterModelCallEvent callbacks are invoked in reverse order, so the first
    registered hook is called last.
    """

    class CustomException(Exception):
        pass

    call_count = [0]

    def mock_stream(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise CustomException("First attempt fails")
        else:
            raise CustomException(f"Should not be called (call {call_count[0]})")

    model = MagicMock()
    model.stream = mock_stream

    async def retry_enabler(event: AfterModelCallEvent):
        """Called first due to reverse order."""
        if event.exception:
            event.retry = True

    async def retry_disabler(event: AfterModelCallEvent):
        """Called last, so it wins."""
        if event.exception:
            event.retry = False

    agent = Agent(model=model)
    agent.hooks.add_callback(AfterModelCallEvent, retry_disabler)  # Registered first, called last
    agent.hooks.add_callback(AfterModelCallEvent, retry_enabler)  # Registered second, called first

    # Should raise exception since last-called hook disabled retry
    with pytest.raises(CustomException, match="First attempt fails"):
        agent("Test last hook wins")

    # Verify stream was only called once
    assert call_count[0] == 1


@pytest.mark.asyncio
async def test_hook_retry_with_throttle_exception(alist, mock_sleep):
    """Test that hook retry works alongside existing throttle retry."""

    class CustomException(Exception):
        pass

    model = MagicMock()
    model.stream.side_effect = [
        CustomException("Custom error"),
        ModelThrottledException("ThrottlingException"),
        ModelThrottledException("ThrottlingException"),
        MockedModelProvider(
            [
                {
                    "role": "assistant",
                    "content": [{"text": "Success after mixed retries"}],
                },
            ]
        ).stream([]),
    ]

    async def handle_after_model_call(event: AfterModelCallEvent):
        if event.exception and isinstance(event.exception, CustomException):
            event.retry = True

    agent = Agent(model=model)
    agent.hooks.add_callback(AfterModelCallEvent, handle_after_model_call)

    result = agent("Test mixed retries")

    # Should succeed after: custom retry + 2 throttle retries
    assert result.stop_reason == "end_turn"
    assert result.message["content"][0]["text"] == "Success after mixed retries"


def test_before_invocation_event_message_modification():
    """Test that hooks can modify messages in BeforeInvocationEvent for input guardrails."""
    mock_provider = MockedModelProvider(
        [
            {
                "role": "assistant",
                "content": [{"text": "I received your redacted message"}],
            },
        ]
    )

    modified_content = None

    async def input_guardrail_hook(event: BeforeInvocationEvent):
        """Simulates a guardrail that redacts sensitive content."""
        nonlocal modified_content
        if event.messages is not None:
            for message in event.messages:
                if message.get("role") == "user":
                    content = message.get("content", [])
                    for block in content:
                        if "text" in block and "SECRET" in block["text"]:
                            # Redact sensitive content in-place
                            block["text"] = block["text"].replace("SECRET", "[REDACTED]")
            modified_content = event.messages[0]["content"][0]["text"]

    agent = Agent(model=mock_provider)
    agent.hooks.add_callback(BeforeInvocationEvent, input_guardrail_hook)

    agent("My password is SECRET123")

    # Verify the message was modified before being processed
    assert modified_content == "My password is [REDACTED]123"
    # Verify the modified message was added to agent's conversation history
    assert agent.messages[0]["content"][0]["text"] == "My password is [REDACTED]123"


def test_before_invocation_event_message_overwrite():
    """Test that hooks can overwrite messages in BeforeInvocationEvent."""
    mock_provider = MockedModelProvider(
        [
            {
                "role": "assistant",
                "content": [{"text": "I received your message message"}],
            },
        ]
    )

    async def overwrite_input_hook(event: BeforeInvocationEvent):
        event.messages = [{"role": "user", "content": [{"text": "GOODBYE"}]}]

    agent = Agent(model=mock_provider)
    agent.hooks.add_callback(BeforeInvocationEvent, overwrite_input_hook)

    agent("HELLO")

    # Verify the message was overwritten to agent's conversation history
    assert agent.messages[0]["content"][0]["text"] == "GOODBYE"


@pytest.mark.filterwarnings("ignore:Agent.structured_output_async method is deprecated:DeprecationWarning")
@pytest.mark.asyncio
async def test_before_invocation_event_messages_none_in_structured_output(agenerator):
    """Test that BeforeInvocationEvent.messages is None when called from deprecated structured_output."""

    class Person(BaseModel):
        name: str
        age: int

    mock_provider = MockedModelProvider([])
    mock_provider.structured_output = Mock(return_value=agenerator([{"output": Person(name="Test", age=30)}]))

    received_messages = "not_set"

    async def capture_messages_hook(event: BeforeInvocationEvent):
        nonlocal received_messages
        received_messages = event.messages

    agent = Agent(model=mock_provider)
    agent.hooks.add_callback(BeforeInvocationEvent, capture_messages_hook)

    await agent.structured_output_async(Person, "Test prompt")

    # structured_output_async uses deprecated path that doesn't pass messages
    assert received_messages is None
