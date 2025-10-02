import json
from unittest.mock import ANY

import pytest

from strands import Agent, tool
from strands.hooks import BeforeToolCallEvent, HookProvider, Interrupt


@pytest.fixture
def interrupt_hook():
    class Hook(HookProvider):
        def register_hooks(self, registry):
            registry.add_callback(BeforeToolCallEvent, self.interrupt)

        def interrupt(self, event):
            response = event.interrupt("need approval")
            if response != "APPROVE":
                event.cancel_tool = "tool rejected"

    return Hook()


@pytest.fixture
def time_tool():
    @tool(name="time_tool")
    def func():
        return "12:00"

    return func


@pytest.fixture
def agent(interrupt_hook, time_tool):
    return Agent(hooks=[interrupt_hook], tools=[time_tool])


@pytest.mark.asyncio
def test_agent_invoke_interrupt(agent):
    result = agent("What is the time?")

    tru_stop_reason = result.stop_reason
    exp_stop_reason = "interrupt"
    assert tru_stop_reason == exp_stop_reason

    tru_interrupts = result.interrupts
    exp_interrupts = [
        Interrupt(
            name="time_tool",
            event_name="BeforeToolCallEvent",
            reasons=["need approval"],
            activated=True,
        ),
    ]
    assert tru_interrupts == exp_interrupts

    responses = [
        {
            "interruptResponse": {
                "name": "time_tool",
                "event_name": "BeforeToolCallEvent",
                "response": "APPROVE",
            },
        },
    ]
    result = agent(responses)

    tru_stop_reason = result.stop_reason
    exp_stop_reason = "end_turn"
    assert tru_stop_reason == exp_stop_reason

    tru_result_message = json.dumps(result.message)
    exp_result_message = "12:00"
    assert exp_result_message in tru_result_message

    tru_tool_result_message = agent.messages[-2]
    exp_tool_result_message = {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": ANY,
                    "status": "success",
                    "content": [
                        {
                            "json": {
                                "interrupt": {
                                    "name": "time_tool",
                                    "event_name": "BeforeToolCallEvent",
                                    "reasons": ["need approval"],
                                },
                            },
                        },
                        {"text": "12:00"},
                    ],
                },
            },
        ],
    }
    assert tru_tool_result_message == exp_tool_result_message


@pytest.mark.asyncio
def test_agent_invoke_interrupt_reject(agent):
    result = agent("What is the time?")

    tru_stop_reason = result.stop_reason
    exp_stop_reason = "interrupt"
    assert tru_stop_reason == exp_stop_reason

    responses = [
        {
            "interruptResponse": {
                "name": "time_tool",
                "event_name": "BeforeToolCallEvent",
                "response": "REJECT",
            },
        },
    ]
    result = agent(responses)

    tru_stop_reason = result.stop_reason
    exp_stop_reason = "end_turn"
    assert tru_stop_reason == exp_stop_reason

    tru_tool_result_message = agent.messages[-2]
    exp_tool_result_message = {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": ANY,
                    "status": "error",
                    "content": [
                        {
                            "json": {
                                "interrupt": {
                                    "name": "time_tool",
                                    "event_name": "BeforeToolCallEvent",
                                    "reasons": ["need approval"],
                                },
                            },
                        },
                        {"text": "tool rejected"},
                    ],
                },
            },
        ],
    }
    assert tru_tool_result_message == exp_tool_result_message
