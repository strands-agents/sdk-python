import unittest.mock
from unittest.mock import ANY, MagicMock, call

import pytest

import strands
from strands import Agent
from strands.agent import AgentResult
from strands.types._events import TypedEvent
from strands.types.exceptions import ModelThrottledException
from tests.fixtures.mocked_model_provider import MockedModelProvider


@pytest.fixture
def mock_time():
    with unittest.mock.patch.object(strands.event_loop.event_loop, "time") as mock:
        yield mock


@pytest.mark.asyncio
async def test_stream_async_e2e(alist, mock_time):
    @strands.tool
    def fake_tool(agent: Agent):
        return "Done!"

    mock_provider = MockedModelProvider(
        [
            {"redactedUserContent": "BLOCKED!", "redactedAssistantContent": "INPUT BLOCKED!"},
            {"role": "assistant", "content": [{"text": "Okay invoking tool!"}]},
            {
                "role": "assistant",
                "content": [{"toolUse": {"name": "fake_tool", "toolUseId": "123", "input": {}}}],
            },
            {"role": "assistant", "content": [{"text": "I invoked a tool!"}]},
        ]
    )
    model = MagicMock()
    model.stream.side_effect = [
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
        mock_provider.stream([]),
    ]

    mock_callback = unittest.mock.Mock()
    agent = Agent(model=model, tools=[fake_tool], callback_handler=mock_callback)

    stream = agent.stream_async("Do the stuff", arg1=1013)

    # Base object with common properties
    throttle_props = {
        "agent": ANY,
        "event_loop_cycle_id": ANY,
        "event_loop_cycle_span": ANY,
        "event_loop_cycle_trace": ANY,
        "arg1": 1013,
        "request_state": {},
    }

    tru_events = await alist(stream)
    exp_events = [
        {"arg1": 1013, "init_event_loop": True},
        {"start": True},
        {"start_event_loop": True},
        {"event_loop_throttled_delay": 8, **throttle_props},
        {"event_loop_throttled_delay": 16, **throttle_props},
        {"event": {"messageStart": {"role": "assistant"}}},
        {"event": {"redactContent": {"redactUserContentMessage": "BLOCKED!"}}},
        {"event": {"contentBlockStart": {"start": {}}}},
        {"event": {"contentBlockDelta": {"delta": {"text": "INPUT BLOCKED!"}}}},
        {
            "agent": ANY,
            "arg1": 1013,
            "data": "INPUT BLOCKED!",
            "delta": {"text": "INPUT BLOCKED!"},
            "event_loop_cycle_id": ANY,
            "event_loop_cycle_span": ANY,
            "event_loop_cycle_trace": ANY,
            "request_state": {},
        },
        {"event": {"contentBlockStop": {}}},
        {"event": {"messageStop": {"stopReason": "guardrail_intervened"}}},
        {"message": {"content": [{"text": "INPUT BLOCKED!"}], "role": "assistant"}},
        {
            "result": AgentResult(
                stop_reason="guardrail_intervened",
                message={"content": [{"text": "INPUT BLOCKED!"}], "role": "assistant"},
                metrics=ANY,
                state={},
            ),
        },
    ]
    assert tru_events == exp_events

    exp_calls = [call(**event) for event in exp_events]
    act_calls = mock_callback.call_args_list
    assert act_calls == exp_calls

    # Ensure that all events coming out of the agent are *not* typed events
    typed_events = [event for event in tru_events if isinstance(event, TypedEvent)]
    assert typed_events == []


@pytest.mark.asyncio
async def test_event_loop_cycle_text_response_throttling_early_end(
    agenerator,
    alist,
    mock_time,
):
    model = MagicMock()
    model.stream.side_effect = [
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
    ]

    mock_callback = unittest.mock.Mock()
    with pytest.raises(ModelThrottledException):
        agent = Agent(model=model, callback_handler=mock_callback)

        # Because we're throwing an exception, we manually collect the items here
        tru_events = []
        stream = agent.stream_async("Do the stuff", arg1=1013)
        async for event in stream:
            tru_events.append(event)

    # Base object with common properties
    common_props = {
        "agent": ANY,
        "event_loop_cycle_id": ANY,
        "event_loop_cycle_span": ANY,
        "event_loop_cycle_trace": ANY,
        "arg1": 1013,
        "request_state": {},
    }

    exp_events = [
        {"init_event_loop": True, "arg1": 1013},
        {"start": True},
        {"start_event_loop": True},
        {"event_loop_throttled_delay": 8, **common_props},
        {"event_loop_throttled_delay": 16, **common_props},
        {"event_loop_throttled_delay": 32, **common_props},
        {"event_loop_throttled_delay": 64, **common_props},
        {"event_loop_throttled_delay": 128, **common_props},
        {"force_stop": True, "force_stop_reason": "ThrottlingException | ConverseStream"},
    ]

    assert tru_events == exp_events

    exp_calls = [call(**event) for event in exp_events]
    act_calls = mock_callback.call_args_list
    assert act_calls == exp_calls

    # Ensure that all events coming out of the agent are *not* typed events
    typed_events = [event for event in tru_events if isinstance(event, TypedEvent)]
    assert typed_events == []
