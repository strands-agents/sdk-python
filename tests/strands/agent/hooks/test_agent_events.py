import asyncio
import unittest.mock
from unittest.mock import ANY, MagicMock, call

import pytest

import strands
from strands import Agent
from strands.agent import AgentResult
from strands.types._events import TypedEvent
from strands.types.exceptions import ModelThrottledException
from tests.fixtures.mocked_model_provider import MockedModelProvider


@strands.tool
def normal_tool(agent: Agent):
    return f"Done with synchronous {agent.name}!"


@strands.tool
async def async_tool(agent: Agent):
    await asyncio.sleep(0.1)
    return f"Done with asynchronous {agent.name}!"


@strands.tool
async def streaming_tool():
    await asyncio.sleep(0.2)
    yield {"tool_streaming": True}
    yield "Final result"


@pytest.fixture
def mock_sleep():
    with unittest.mock.patch.object(
        strands.event_loop.event_loop.asyncio, "sleep", new_callable=unittest.mock.AsyncMock
    ) as mock:
        yield mock


any_props = {
    "agent": ANY,
    "event_loop_cycle_id": ANY,
    "event_loop_cycle_span": ANY,
    "event_loop_cycle_trace": ANY,
    "request_state": {},
}


@pytest.mark.asyncio
async def test_stream_e2e_success(alist):
    mock_provider = MockedModelProvider(
        [
            {
                "role": "assistant",
                "content": [
                    {"text": "Okay invoking normal tool"},
                    {"toolUse": {"name": "normal_tool", "toolUseId": "123", "input": {}}},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"text": "Invoking async tool"},
                    {"toolUse": {"name": "async_tool", "toolUseId": "1234", "input": {}}},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"text": "Invoking streaming tool"},
                    {"toolUse": {"name": "streaming_tool", "toolUseId": "12345", "input": {}}},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"text": "I invoked the tools!"},
                ],
            },
        ]
    )

    mock_callback = unittest.mock.Mock()
    agent = Agent(model=mock_provider, tools=[async_tool, normal_tool, streaming_tool], callback_handler=mock_callback)

    stream = agent.stream_async("Do the stuff", arg1=1013)

    tool_config = {
        "toolChoice": {"auto": {}},
        "tools": [
            {
                "toolSpec": {
                    "description": "async_tool",
                    "inputSchema": {"json": {"properties": {}, "required": [], "type": "object"}},
                    "name": "async_tool",
                }
            },
            {
                "toolSpec": {
                    "description": "normal_tool",
                    "inputSchema": {"json": {"properties": {}, "required": [], "type": "object"}},
                    "name": "normal_tool",
                }
            },
            {
                "toolSpec": {
                    "description": "streaming_tool",
                    "inputSchema": {"json": {"properties": {}, "required": [], "type": "object"}},
                    "name": "streaming_tool",
                }
            },
        ],
    }

    tru_events = await alist(stream)
    exp_events = [
        # Cycle 1: Initialize and invoke normal_tool
        {"arg1": 1013, "init_event_loop": True},
        {"start": True},
        {"start_event_loop": True},
        {"event": {"messageStart": {"role": "assistant"}}},
        {"event": {"contentBlockStart": {"start": {}}}},
        {"event": {"contentBlockDelta": {"delta": {"text": "Okay invoking normal tool"}}}},
        {
            **any_props,
            "arg1": 1013,
            "data": "Okay invoking normal tool",
            "delta": {"text": "Okay invoking normal tool"},
        },
        {"event": {"contentBlockStop": {}}},
        {"event": {"contentBlockStart": {"start": {"toolUse": {"name": "normal_tool", "toolUseId": "123"}}}}},
        {"event": {"contentBlockDelta": {"delta": {"toolUse": {"input": "{}"}}}}},
        {
            **any_props,
            "arg1": 1013,
            "current_tool_use": {"input": {}, "name": "normal_tool", "toolUseId": "123"},
            "delta": {"toolUse": {"input": "{}"}},
        },
        {"event": {"contentBlockStop": {}}},
        {"event": {"messageStop": {"stopReason": "tool_use"}}},
        {
            "message": {
                "content": [
                    {"text": "Okay invoking normal tool"},
                    {"toolUse": {"input": {}, "name": "normal_tool", "toolUseId": "123"}},
                ],
                "role": "assistant",
            }
        },
        {
            "message": {
                "content": [
                    {
                        "toolResult": {
                            "content": [{"text": "Done with synchronous Strands Agents!"}],
                            "status": "success",
                            "toolUseId": "123",
                        }
                    },
                ],
                "role": "user",
            }
        },
        # Cycle 2: Invoke async_tool
        {"start": True},
        {"start": True},
        {"start_event_loop": True},
        {"event": {"messageStart": {"role": "assistant"}}},
        {"event": {"contentBlockStart": {"start": {}}}},
        {"event": {"contentBlockDelta": {"delta": {"text": "Invoking async tool"}}}},
        {
            **any_props,
            "arg1": 1013,
            "data": "Invoking async tool",
            "delta": {"text": "Invoking async tool"},
            "event_loop_parent_cycle_id": ANY,
            "messages": ANY,
            "model": ANY,
            "system_prompt": None,
            "tool_config": tool_config,
        },
        {"event": {"contentBlockStop": {}}},
        {"event": {"contentBlockStart": {"start": {"toolUse": {"name": "async_tool", "toolUseId": "1234"}}}}},
        {"event": {"contentBlockDelta": {"delta": {"toolUse": {"input": "{}"}}}}},
        {
            **any_props,
            "arg1": 1013,
            "current_tool_use": {"input": {}, "name": "async_tool", "toolUseId": "1234"},
            "delta": {"toolUse": {"input": "{}"}},
            "event_loop_parent_cycle_id": ANY,
            "messages": ANY,
            "model": ANY,
            "system_prompt": None,
            "tool_config": tool_config,
        },
        {"event": {"contentBlockStop": {}}},
        {"event": {"messageStop": {"stopReason": "tool_use"}}},
        {
            "message": {
                "content": [
                    {"text": "Invoking async tool"},
                    {"toolUse": {"input": {}, "name": "async_tool", "toolUseId": "1234"}},
                ],
                "role": "assistant",
            }
        },
        {
            "message": {
                "content": [
                    {
                        "toolResult": {
                            "content": [{"text": "Done with asynchronous Strands Agents!"}],
                            "status": "success",
                            "toolUseId": "1234",
                        }
                    },
                ],
                "role": "user",
            }
        },
        # Cycle 3: Invoke streaming_tool
        {"start": True},
        {"start": True},
        {"start_event_loop": True},
        {"event": {"messageStart": {"role": "assistant"}}},
        {"event": {"contentBlockStart": {"start": {}}}},
        {"event": {"contentBlockDelta": {"delta": {"text": "Invoking streaming tool"}}}},
        {
            **any_props,
            "arg1": 1013,
            "data": "Invoking streaming tool",
            "delta": {"text": "Invoking streaming tool"},
            "event_loop_parent_cycle_id": ANY,
            "messages": ANY,
            "model": ANY,
            "system_prompt": None,
            "tool_config": tool_config,
        },
        {"event": {"contentBlockStop": {}}},
        {"event": {"contentBlockStart": {"start": {"toolUse": {"name": "streaming_tool", "toolUseId": "12345"}}}}},
        {"event": {"contentBlockDelta": {"delta": {"toolUse": {"input": "{}"}}}}},
        {
            **any_props,
            "arg1": 1013,
            "current_tool_use": {"input": {}, "name": "streaming_tool", "toolUseId": "12345"},
            "delta": {"toolUse": {"input": "{}"}},
            "event_loop_parent_cycle_id": ANY,
            "messages": ANY,
            "model": ANY,
            "system_prompt": None,
            "tool_config": tool_config,
        },
        {"event": {"contentBlockStop": {}}},
        {"event": {"messageStop": {"stopReason": "tool_use"}}},
        {
            "message": {
                "content": [
                    {"text": "Invoking streaming tool"},
                    {"toolUse": {"input": {}, "name": "streaming_tool", "toolUseId": "12345"}},
                ],
                "role": "assistant",
            }
        },
        {
            "tool_stream_event": {
                "data": {"tool_streaming": True},
                "tool_use": {"input": {}, "name": "streaming_tool", "toolUseId": "12345"},
            }
        },
        {
            "tool_stream_event": {
                "data": "Final result",
                "tool_use": {"input": {}, "name": "streaming_tool", "toolUseId": "12345"},
            }
        },
        {
            "message": {
                "content": [
                    {"toolResult": {"content": [{"text": "Final result"}], "status": "success", "toolUseId": "12345"}}
                ],
                "role": "user",
            }
        },
        # Cycle 4: Final response
        {"start": True},
        {"start": True},
        {"start_event_loop": True},
        {"event": {"messageStart": {"role": "assistant"}}},
        {"event": {"contentBlockStart": {"start": {}}}},
        {"event": {"contentBlockDelta": {"delta": {"text": "I invoked the tools!"}}}},
        {
            **any_props,
            "arg1": 1013,
            "data": "I invoked the tools!",
            "delta": {"text": "I invoked the tools!"},
            "event_loop_parent_cycle_id": ANY,
            "messages": ANY,
            "model": ANY,
            "system_prompt": None,
            "tool_config": tool_config,
        },
        {"event": {"contentBlockStop": {}}},
        {"event": {"messageStop": {"stopReason": "end_turn"}}},
        {"message": {"content": [{"text": "I invoked the tools!"}], "role": "assistant"}},
        {
            "result": AgentResult(
                stop_reason="end_turn",
                message={"content": [{"text": "I invoked the tools!"}], "role": "assistant"},
                metrics=ANY,
                state={},
            )
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
async def test_stream_e2e_throttle_and_redact(alist, mock_sleep):
    model = MagicMock()
    model.stream.side_effect = [
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
        MockedModelProvider(
            [
                {"redactedUserContent": "BLOCKED!", "redactedAssistantContent": "INPUT BLOCKED!"},
            ]
        ).stream([]),
    ]

    mock_callback = unittest.mock.Mock()
    agent = Agent(model=model, tools=[normal_tool], callback_handler=mock_callback)

    stream = agent.stream_async("Do the stuff", arg1=1013)

    # Base object with common properties
    throttle_props = {
        **any_props,
        "arg1": 1013,
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
            **any_props,
            "arg1": 1013,
            "data": "INPUT BLOCKED!",
            "delta": {"text": "INPUT BLOCKED!"},
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
    mock_sleep,
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
        **any_props,
        "arg1": 1013,
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
