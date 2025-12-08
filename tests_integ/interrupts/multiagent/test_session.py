import json
from unittest.mock import ANY

import pytest

from strands import Agent, tool
from strands.interrupt import Interrupt
from strands.multiagent import Swarm
from strands.multiagent.base import Status
from strands.session import FileSessionManager
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


def test_swarm_interrupt_session(weather_tool, tmpdir):
    weather_agent = Agent(name="weather", tools=[weather_tool])
    summarizer_agent = Agent(name="summarizer")
    session_manager = FileSessionManager(session_id="strands-interrupt-test", storage_dir=tmpdir)
    swarm = Swarm([weather_agent, summarizer_agent], session_manager=session_manager)

    multiagent_result = swarm("Can you check the weather and then summarize the results?")

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

    weather_agent = Agent(name="weather", tools=[weather_tool])
    summarizer_agent = Agent(name="summarizer")
    session_manager = FileSessionManager(session_id="strands-interrupt-test", storage_dir=tmpdir)
    swarm = Swarm([weather_agent, summarizer_agent], session_manager=session_manager)

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

    assert len(multiagent_result.results) == 2
    summarizer_result = multiagent_result.results["summarizer"]

    summarizer_message = json.dumps(summarizer_result.result.message).lower()
    assert "sunny" in summarizer_message
