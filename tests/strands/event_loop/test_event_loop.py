import concurrent
import unittest.mock
from unittest.mock import MagicMock, call, patch

import pytest

import strands
import strands.telemetry
from strands.experimental.hooks import (
    AfterModelInvocationEvent,
    AfterToolInvocationEvent,
    BeforeModelInvocationEvent,
    BeforeToolInvocationEvent,
)
from strands.hooks import HookRegistry
from strands.telemetry.metrics import EventLoopMetrics
from strands.tools.executors import SequentialToolExecutor
from strands.tools.registry import ToolRegistry
from strands.types.exceptions import (
    ContextWindowOverflowException,
    EventLoopException,
    MaxTokensReachedException,
    ModelThrottledException,
)
from tests.fixtures.mock_hook_provider import MockHookProvider


@pytest.fixture
def mock_sleep():
    with unittest.mock.patch.object(
        strands.event_loop.event_loop.asyncio, "sleep", new_callable=unittest.mock.AsyncMock
    ) as mock:
        yield mock


@pytest.fixture
def model():
    return unittest.mock.Mock()


@pytest.fixture
def system_prompt():
    return "p1"


@pytest.fixture
def messages():
    return [{"role": "user", "content": [{"text": "Hello"}]}]


@pytest.fixture
def tool_registry():
    return ToolRegistry()


@pytest.fixture
def thread_pool():
    return concurrent.futures.ThreadPoolExecutor(max_workers=1)


@pytest.fixture
def tool(tool_registry):
    @strands.tool
    def tool_for_testing(random_string: str):
        return random_string

    tool_registry.register_tool(tool_for_testing)

    return tool_for_testing


@pytest.fixture
def tool_times_2(tool_registry):
    @strands.tools.tool
    def multiply_by_2(x: int) -> int:
        return x * 2

    tool_registry.register_tool(multiply_by_2)

    return multiply_by_2


@pytest.fixture
def tool_times_5(tool_registry):
    @strands.tools.tool
    def multiply_by_5(x: int) -> int:
        return x * 5

    tool_registry.register_tool(multiply_by_5)

    return multiply_by_5


@pytest.fixture
def tool_stream(tool):
    return [
        {
            "contentBlockStart": {
                "start": {
                    "toolUse": {
                        "toolUseId": "t1",
                        "name": tool.tool_spec["name"],
                    },
                },
            },
        },
        {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"random_string": "abcdEfghI123"}'}}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "tool_use"}},
    ]


@pytest.fixture
def hook_registry():
    return HookRegistry()


@pytest.fixture
def hook_provider(hook_registry):
    provider = MockHookProvider(
        event_types=[
            BeforeToolInvocationEvent,
            AfterToolInvocationEvent,
            BeforeModelInvocationEvent,
            AfterModelInvocationEvent,
        ]
    )
    hook_registry.add_hook(provider)
    return provider


@pytest.fixture
def tool_executor():
    return SequentialToolExecutor()


@pytest.fixture
def agent(model, system_prompt, messages, tool_registry, thread_pool, hook_registry, tool_executor):
    mock = unittest.mock.Mock(name="agent")
    mock.config.cache_points = []
    mock.model = model
    mock.system_prompt = system_prompt
    mock.messages = messages
    mock.tool_registry = tool_registry
    mock.thread_pool = thread_pool
    mock.event_loop_metrics = EventLoopMetrics()
    mock.hooks = hook_registry
    mock.tool_executor = tool_executor

    return mock


@pytest.fixture
def mock_tracer():
    tracer = MagicMock()
    tracer.start_event_loop_cycle_span.return_value = MagicMock()
    tracer.start_model_invoke_span.return_value = MagicMock()
    return tracer


@pytest.mark.asyncio
async def test_event_loop_cycle_text_response(
    agent,
    model,
    agenerator,
    alist,
):
    model.stream.return_value = agenerator(
        [
            {"contentBlockDelta": {"delta": {"text": "test text"}}},
            {"contentBlockStop": {}},
        ]
    )

    stream = strands.event_loop.event_loop.event_loop_cycle(
        agent=agent,
        invocation_state={},
    )
    events = await alist(stream)
    tru_stop_reason, tru_message, _, tru_request_state = events[-1]["stop"]

    exp_stop_reason = "end_turn"
    exp_message = {"role": "assistant", "content": [{"text": "test text"}]}
    exp_request_state = {}

    assert tru_stop_reason == exp_stop_reason and tru_message == exp_message and tru_request_state == exp_request_state


@pytest.mark.asyncio
async def test_event_loop_cycle_text_response_throttling(
    mock_sleep,
    agent,
    model,
    agenerator,
    alist,
):
    model.stream.side_effect = [
        ModelThrottledException("ThrottlingException | ConverseStream"),
        agenerator(
            [
                {"contentBlockDelta": {"delta": {"text": "test text"}}},
                {"contentBlockStop": {}},
            ]
        ),
    ]

    stream = strands.event_loop.event_loop.event_loop_cycle(
        agent=agent,
        invocation_state={},
    )
    events = await alist(stream)
    tru_stop_reason, tru_message, _, tru_request_state = events[-1]["stop"]

    exp_stop_reason = "end_turn"
    exp_message = {"role": "assistant", "content": [{"text": "test text"}]}
    exp_request_state = {}

    assert tru_stop_reason == exp_stop_reason and tru_message == exp_message and tru_request_state == exp_request_state
    # Verify that sleep was called once with the initial delay
    mock_sleep.assert_called_once()


@pytest.mark.asyncio
async def test_event_loop_cycle_exponential_backoff(
    mock_sleep,
    agent,
    model,
    agenerator,
    alist,
):
    """Test that the exponential backoff works correctly with multiple retries."""
    # Set up the model to raise throttling exceptions multiple times before succeeding
    model.stream.side_effect = [
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
        agenerator(
            [
                {"contentBlockDelta": {"delta": {"text": "test text"}}},
                {"contentBlockStop": {}},
            ]
        ),
    ]

    stream = strands.event_loop.event_loop.event_loop_cycle(
        agent=agent,
        invocation_state={},
    )
    events = await alist(stream)
    tru_stop_reason, tru_message, _, tru_request_state = events[-1]["stop"]

    # Verify the final response
    assert tru_stop_reason == "end_turn"
    assert tru_message == {"role": "assistant", "content": [{"text": "test text"}]}
    assert tru_request_state == {}

    # Verify that sleep was called with increasing delays
    # Initial delay is 4, then 8, then 16
    assert mock_sleep.call_count == 3
    assert mock_sleep.call_args_list == [call(4), call(8), call(16)]


@pytest.mark.asyncio
async def test_event_loop_cycle_text_response_throttling_exceeded(
    mock_sleep,
    agent,
    model,
    alist,
):
    model.stream.side_effect = [
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
    ]

    with pytest.raises(ModelThrottledException):
        stream = strands.event_loop.event_loop.event_loop_cycle(
            agent=agent,
            invocation_state={},
        )
        await alist(stream)

    mock_sleep.assert_has_calls(
        [
            call(4),
            call(8),
            call(16),
            call(32),
            call(64),
        ]
    )


@pytest.mark.asyncio
async def test_event_loop_cycle_text_response_error(
    agent,
    model,
    alist,
):
    model.stream.side_effect = RuntimeError("Unhandled error")

    with pytest.raises(RuntimeError):
        stream = strands.event_loop.event_loop.event_loop_cycle(
            agent=agent,
            invocation_state={},
        )
        await alist(stream)


@patch("strands.event_loop.event_loop.recover_message_on_max_tokens_reached")
@pytest.mark.asyncio
async def test_event_loop_cycle_tool_result(
    mock_recover_message,
    agent,
    model,
    system_prompt,
    messages,
    tool_stream,
    tool_registry,
    agenerator,
    alist,
):
    model.stream.side_effect = [
        agenerator(tool_stream),
        agenerator(
            [
                {"contentBlockDelta": {"delta": {"text": "test text"}}},
                {"contentBlockStop": {}},
            ]
        ),
    ]

    stream = strands.event_loop.event_loop.event_loop_cycle(
        agent=agent,
        invocation_state={},
    )
    events = await alist(stream)
    tru_stop_reason, tru_message, _, tru_request_state = events[-1]["stop"]

    exp_stop_reason = "end_turn"
    exp_message = {"role": "assistant", "content": [{"text": "test text"}]}
    exp_request_state = {}

    assert tru_stop_reason == exp_stop_reason and tru_message == exp_message and tru_request_state == exp_request_state

    # Verify that recover_message_on_max_tokens_reached was NOT called for tool_use stop reason
    mock_recover_message.assert_not_called()

    model.stream.assert_called_with(
        [
            {"role": "user", "content": [{"text": "Hello"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "t1",
                            "name": "tool_for_testing",
                            "input": {"random_string": "abcdEfghI123"},
                        }
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "t1",
                            "status": "success",
                            "content": [{"text": "abcdEfghI123"}],
                        },
                    },
                ],
            },
            {"role": "assistant", "content": [{"text": "test text"}]},
        ],
        tool_registry.get_all_tool_specs(),
        "p1",
    )


@pytest.mark.asyncio
async def test_event_loop_cycle_tool_result_error(
    agent,
    model,
    tool_stream,
    agenerator,
    alist,
):
    model.stream.side_effect = [agenerator(tool_stream)]

    with pytest.raises(EventLoopException):
        stream = strands.event_loop.event_loop.event_loop_cycle(
            agent=agent,
            invocation_state={},
        )
        await alist(stream)


@pytest.mark.asyncio
async def test_event_loop_cycle_tool_result_no_tool_handler(
    agent,
    model,
    tool_stream,
    agenerator,
    alist,
):
    model.stream.side_effect = [agenerator(tool_stream)]
    # Set tool_handler to None for this test
    agent.tool_handler = None

    with pytest.raises(EventLoopException):
        stream = strands.event_loop.event_loop.event_loop_cycle(
            agent=agent,
            invocation_state={},
        )
        await alist(stream)


@pytest.mark.asyncio
async def test_event_loop_cycle_stop(
    agent,
    model,
    tool,
    agenerator,
    alist,
):
    model.stream.side_effect = [
        agenerator(
            [
                {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "toolUseId": "t1",
                                "name": tool.tool_spec["name"],
                            },
                        },
                    },
                },
                {"contentBlockStop": {}},
                {"messageStop": {"stopReason": "tool_use"}},
            ]
        ),
    ]

    stream = strands.event_loop.event_loop.event_loop_cycle(
        agent=agent,
        invocation_state={"request_state": {"stop_event_loop": True}},
    )
    events = await alist(stream)
    tru_stop_reason, tru_message, _, tru_request_state = events[-1]["stop"]

    exp_stop_reason = "tool_use"
    exp_message = {
        "role": "assistant",
        "content": [
            {
                "toolUse": {
                    "input": {},
                    "name": "tool_for_testing",
                    "toolUseId": "t1",
                }
            }
        ],
    }
    exp_request_state = {"stop_event_loop": True}

    assert tru_stop_reason == exp_stop_reason and tru_message == exp_message and tru_request_state == exp_request_state


@pytest.mark.asyncio
async def test_cycle_exception(
    agent,
    model,
    tool_stream,
    agenerator,
):
    model.stream.side_effect = [
        agenerator(tool_stream),
        agenerator(tool_stream),
        agenerator(tool_stream),
        ValueError("Invalid error presented"),
    ]

    tru_stop_event = None
    exp_stop_event = {"force_stop": True, "force_stop_reason": "Invalid error presented"}

    with pytest.raises(EventLoopException):
        stream = strands.event_loop.event_loop.event_loop_cycle(
            agent=agent,
            invocation_state={},
        )
        async for event in stream:
            tru_stop_event = event

    assert tru_stop_event == exp_stop_event


@patch("strands.event_loop.event_loop.get_tracer")
@pytest.mark.asyncio
async def test_event_loop_cycle_creates_spans(
    mock_get_tracer,
    agent,
    model,
    mock_tracer,
    agenerator,
    alist,
):
    # Setup
    mock_get_tracer.return_value = mock_tracer
    cycle_span = MagicMock()
    mock_tracer.start_event_loop_cycle_span.return_value = cycle_span
    model_span = MagicMock()
    mock_tracer.start_model_invoke_span.return_value = model_span

    model.stream.return_value = agenerator(
        [
            {"contentBlockDelta": {"delta": {"text": "test text"}}},
            {"contentBlockStop": {}},
        ]
    )

    # Call event_loop_cycle
    stream = strands.event_loop.event_loop.event_loop_cycle(
        agent=agent,
        invocation_state={},
    )
    await alist(stream)

    # Verify tracer methods were called correctly
    mock_get_tracer.assert_called_once()
    mock_tracer.start_event_loop_cycle_span.assert_called_once()
    mock_tracer.start_model_invoke_span.assert_called_once()
    mock_tracer.end_model_invoke_span.assert_called_once()
    mock_tracer.end_event_loop_cycle_span.assert_called_once()


@patch("strands.event_loop.event_loop.get_tracer")
@pytest.mark.asyncio
async def test_event_loop_tracing_with_model_error(
    mock_get_tracer,
    agent,
    model,
    mock_tracer,
    alist,
):
    # Setup
    mock_get_tracer.return_value = mock_tracer
    cycle_span = MagicMock()
    mock_tracer.start_event_loop_cycle_span.return_value = cycle_span
    model_span = MagicMock()
    mock_tracer.start_model_invoke_span.return_value = model_span

    # Set up model to raise an exception
    model.stream.side_effect = ContextWindowOverflowException("Input too long")

    # Call event_loop_cycle, expecting it to handle the exception
    with pytest.raises(ContextWindowOverflowException):
        stream = strands.event_loop.event_loop.event_loop_cycle(
            agent=agent,
            invocation_state={},
        )
        await alist(stream)

    # Verify error handling span methods were called
    mock_tracer.end_span_with_error.assert_called_once_with(model_span, "Input too long", model.stream.side_effect)


@pytest.mark.asyncio
async def test_event_loop_cycle_max_tokens_exception(
    agent,
    model,
    agenerator,
    alist,
):
    """Test that max_tokens stop reason calls _recover_message_on_max_tokens_reached then MaxTokensReachedException."""

    model.stream.side_effect = [
        agenerator(
            [
                {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "toolUseId": "t1",
                                "name": "asdf",
                                "input": {},  # empty
                            },
                        },
                    },
                },
                {"contentBlockStop": {}},
                {"messageStop": {"stopReason": "max_tokens"}},
            ]
        ),
    ]

    # Call event_loop_cycle, expecting it to raise MaxTokensReachedException
    expected_message = (
        "Agent has reached an unrecoverable state due to max_tokens limit. "
        "For more information see: "
        "https://strandsagents.com/latest/user-guide/concepts/agents/agent-loop/#maxtokensreachedexception"
    )
    with pytest.raises(MaxTokensReachedException, match=expected_message):
        stream = strands.event_loop.event_loop.event_loop_cycle(
            agent=agent,
            invocation_state={},
        )
        await alist(stream)

    # Verify the exception message contains the expected content
    assert len(agent.messages) == 2
    assert "tool use was incomplete due" in agent.messages[1]["content"][0]["text"]


@patch("strands.event_loop.event_loop.get_tracer")
@pytest.mark.asyncio
async def test_event_loop_tracing_with_tool_execution(
    mock_get_tracer,
    agent,
    model,
    tool_stream,
    mock_tracer,
    agenerator,
    alist,
):
    # Setup
    mock_get_tracer.return_value = mock_tracer
    cycle_span = MagicMock()
    mock_tracer.start_event_loop_cycle_span.return_value = cycle_span
    model_span = MagicMock()
    mock_tracer.start_model_invoke_span.return_value = model_span

    # Set up model to return tool use and then text response
    model.stream.side_effect = [
        agenerator(tool_stream),
        agenerator(
            [
                {"contentBlockDelta": {"delta": {"text": "test text"}}},
                {"contentBlockStop": {}},
            ]
        ),
    ]

    # Call event_loop_cycle which should execute a tool
    stream = strands.event_loop.event_loop.event_loop_cycle(
        agent=agent,
        invocation_state={},
    )
    await alist(stream)

    # Verify the parent_span parameter is passed to run_tools
    # At a minimum, verify both model spans were created (one for each model invocation)
    assert mock_tracer.start_model_invoke_span.call_count == 2
    assert mock_tracer.end_model_invoke_span.call_count == 2


@patch("strands.event_loop.event_loop.get_tracer")
@pytest.mark.asyncio
async def test_event_loop_tracing_with_throttling_exception(
    mock_get_tracer,
    agent,
    model,
    mock_tracer,
    agenerator,
    alist,
):
    # Setup
    mock_get_tracer.return_value = mock_tracer
    cycle_span = MagicMock()
    mock_tracer.start_event_loop_cycle_span.return_value = cycle_span
    model_span = MagicMock()
    mock_tracer.start_model_invoke_span.return_value = model_span

    # Set up model to raise a throttling exception and then succeed
    model.stream.side_effect = [
        ModelThrottledException("Throttling Error"),
        agenerator(
            [
                {"contentBlockDelta": {"delta": {"text": "test text"}}},
                {"contentBlockStop": {}},
            ]
        ),
    ]

    # Mock the time.sleep function to speed up the test
    with patch("strands.event_loop.event_loop.asyncio.sleep", new_callable=unittest.mock.AsyncMock):
        stream = strands.event_loop.event_loop.event_loop_cycle(
            agent=agent,
            invocation_state={},
        )
        await alist(stream)

    # Verify error span was created for the throttling exception
    assert mock_tracer.end_span_with_error.call_count == 1
    # Verify span was created for the successful retry
    assert mock_tracer.start_model_invoke_span.call_count == 2
    assert mock_tracer.end_model_invoke_span.call_count == 1


@patch("strands.event_loop.event_loop.get_tracer")
@pytest.mark.asyncio
async def test_event_loop_cycle_with_parent_span(
    mock_get_tracer,
    agent,
    model,
    messages,
    mock_tracer,
    agenerator,
    alist,
):
    # Setup
    mock_get_tracer.return_value = mock_tracer
    parent_span = MagicMock()
    cycle_span = MagicMock()
    mock_tracer.start_event_loop_cycle_span.return_value = cycle_span

    model.stream.return_value = agenerator(
        [
            {"contentBlockDelta": {"delta": {"text": "test text"}}},
            {"contentBlockStop": {}},
        ]
    )

    # Set the parent span for this test
    agent.trace_span = parent_span

    # Call event_loop_cycle with a parent span
    stream = strands.event_loop.event_loop.event_loop_cycle(
        agent=agent,
        invocation_state={},
    )
    await alist(stream)

    # Verify parent_span was used when creating cycle span
    mock_tracer.start_event_loop_cycle_span.assert_called_once_with(
        invocation_state=unittest.mock.ANY, parent_span=parent_span, messages=messages
    )


@pytest.mark.asyncio
async def test_request_state_initialization(alist):
    # Create a mock agent
    mock_agent = MagicMock()
    mock_agent.event_loop_metrics.start_cycle.return_value = (0, MagicMock())

    # Call without providing request_state
    stream = strands.event_loop.event_loop.event_loop_cycle(
        agent=mock_agent,
        invocation_state={},
    )
    events = await alist(stream)
    _, _, _, tru_request_state = events[-1]["stop"]

    # Verify request_state was initialized to empty dict
    assert tru_request_state == {}

    # Call with pre-existing request_state
    initial_request_state = {"key": "value"}
    stream = strands.event_loop.event_loop.event_loop_cycle(
        agent=mock_agent,
        invocation_state={"request_state": initial_request_state},
    )
    events = await alist(stream)
    _, _, _, tru_request_state = events[-1]["stop"]

    # Verify existing request_state was preserved
    assert tru_request_state == initial_request_state


@pytest.mark.asyncio
async def test_prepare_next_cycle_in_tool_execution(agent, model, tool_stream, agenerator, alist):
    """Test that cycle ID and metrics are properly updated during tool execution."""
    model.stream.side_effect = [
        agenerator(tool_stream),
        agenerator(
            [
                {"contentBlockStop": {}},
            ]
        ),
    ]

    # Create a mock for recurse_event_loop to capture the invocation_state passed to it
    with unittest.mock.patch.object(strands.event_loop.event_loop, "recurse_event_loop") as mock_recurse:
        # Set up mock to return a valid response
        mock_recurse.return_value = agenerator(
            [
                (
                    "end_turn",
                    {"role": "assistant", "content": [{"text": "test text"}]},
                    strands.telemetry.metrics.EventLoopMetrics(),
                    {},
                ),
            ]
        )

        # Call event_loop_cycle which should execute a tool and then call recurse_event_loop
        stream = strands.event_loop.event_loop.event_loop_cycle(
            agent=agent,
            invocation_state={},
        )
        await alist(stream)

        assert mock_recurse.called

        # Verify required properties are present
        recursive_args = mock_recurse.call_args[1]
        assert "event_loop_parent_cycle_id" in recursive_args["invocation_state"]
        assert (
            recursive_args["invocation_state"]["event_loop_parent_cycle_id"]
            == recursive_args["invocation_state"]["event_loop_cycle_id"]
        )


@pytest.mark.asyncio
async def test_event_loop_cycle_exception_model_hooks(mock_sleep, agent, model, agenerator, alist, hook_provider):
    """Test that model hooks are correctly emitted even when throttled."""
    # Set up the model to raise throttling exceptions multiple times before succeeding
    exception = ModelThrottledException("ThrottlingException | ConverseStream")
    model.stream.side_effect = [
        exception,
        exception,
        exception,
        agenerator(
            [
                {"contentBlockDelta": {"delta": {"text": "test text"}}},
                {"contentBlockStop": {}},
            ]
        ),
    ]

    stream = strands.event_loop.event_loop.event_loop_cycle(
        agent=agent,
        invocation_state={},
    )
    await alist(stream)

    count, events = hook_provider.get_events()

    assert count == 8

    # 1st call - throttled
    assert next(events) == BeforeModelInvocationEvent(agent=agent)
    assert next(events) == AfterModelInvocationEvent(agent=agent, stop_response=None, exception=exception)

    # 2nd call - throttled
    assert next(events) == BeforeModelInvocationEvent(agent=agent)
    assert next(events) == AfterModelInvocationEvent(agent=agent, stop_response=None, exception=exception)

    # 3rd call - throttled
    assert next(events) == BeforeModelInvocationEvent(agent=agent)
    assert next(events) == AfterModelInvocationEvent(agent=agent, stop_response=None, exception=exception)

    # 4th call - successful
    assert next(events) == BeforeModelInvocationEvent(agent=agent)
    assert next(events) == AfterModelInvocationEvent(
        agent=agent,
        stop_response=AfterModelInvocationEvent.ModelStopResponse(
            message={"content": [{"text": "test text"}], "role": "assistant"}, stop_reason="end_turn"
        ),
        exception=None,
    )
