import json
from unittest.mock import ANY

import pytest

from strands import Agent, tool
from strands.interrupt import Interrupt
from strands.multiagent import Swarm
from strands.multiagent.base import Status
from strands.types.tools import ToolContext


@pytest.fixture
def weather_tool():
    @tool(name="weather_tool", context=True)
    def func(tool_context: ToolContext) -> str:
        response = tool_context.interrupt("test_interrupt", reason="need weather")
        return response

    return func


@pytest.fixture
def swarm(weather_tool):
    weather_agent = Agent(name="weather", tools=[weather_tool])

    return Swarm([weather_agent])


def test_swarm_interrupt_agent(swarm):
    multiagent_result = swarm("What is the weather?")

    tru_status = multiagent_result.status
    exp_status = Status.INTERRUPTED
    assert tru_status == exp_status

    tru_interrupts = multiagent_result.interrupts
    exp_interrupts = [
        Interrupt(
            id=ANY,
            name="test_interrupt",
            reason="need weather",
        ),
    ]
    assert tru_interrupts == exp_interrupts

    interrupt = multiagent_result.interrupts[0]

    responses = [
        {
            "interruptResponse": {
                "interruptId": interrupt.id,
                "response": "sunny",
            },
        },
    ]
    multiagent_result = swarm(responses)

    tru_status = multiagent_result.status
    exp_status = Status.COMPLETED
    assert tru_status == exp_status

    assert len(multiagent_result.results) == 1
    weather_result = multiagent_result.results["weather"]

    weather_message = json.dumps(weather_result.result.message).lower()
    assert "sunny" in weather_message
