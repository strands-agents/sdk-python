import json
from unittest.mock import ANY

import pytest

from strands import Agent, tool
from strands.experimental.hooks.multiagent import BeforeNodeCallEvent
from strands.hooks import HookProvider
from strands.interrupt import Interrupt
from strands.multiagent import GraphBuilder, Swarm
from strands.multiagent.base import Status
from strands.session import FileSessionManager
from strands.types.tools import ToolContext


@pytest.fixture
def interrupt_hook():
    class Hook(HookProvider):
        def register_hooks(self, registry):
            registry.add_callback(BeforeNodeCallEvent, self.interrupt)

        def interrupt(self, event):
            if event.node_id == "time":
                response = event.interrupt("test_interrupt", reason="need approval")
                if response != "APPROVE":
                    event.cancel_node = "node rejected"

    return Hook()


@pytest.fixture
def weather_tool():
    @tool(name="weather_tool", context=True)
    def func(tool_context: ToolContext) -> str:
        response = tool_context.interrupt("test_interrupt", reason="need weather")
        return response

    return func


@pytest.fixture
def time_tool():
    @tool(name="time_tool")
    def func():
        return "12:01"

    return func


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


def test_graph_interrupt_session(interrupt_hook, time_tool, tmpdir):
    time_agent = Agent(name="time", tools=[time_tool])
    summarizer_agent = Agent(name="summarizer")
    session_manager = FileSessionManager(session_id="strands-interrupt-test", storage_dir=tmpdir)

    builder = GraphBuilder()
    builder.add_node(time_agent, "time")
    builder.add_node(summarizer_agent, "summarizer")
    builder.add_edge("time", "summarizer")
    builder.set_hook_providers([interrupt_hook])
    builder.set_session_manager(session_manager)
    graph = builder.build()

    multiagent_result = graph("Can you check the time and then summarize the results?")

    tru_result_status = multiagent_result.status
    exp_result_status = Status.INTERRUPTED
    assert tru_result_status == exp_result_status

    tru_state_status = graph.state.status
    exp_state_status = Status.INTERRUPTED
    assert tru_state_status == exp_state_status

    tru_interrupts = multiagent_result.interrupts
    exp_interrupts = [
        Interrupt(
            id=ANY,
            name="test_interrupt",
            reason="need approval",
        ),
    ]
    assert tru_interrupts == exp_interrupts

    interrupt = multiagent_result.interrupts[0]

    time_agent = Agent(name="time", tools=[time_tool])
    summarizer_agent = Agent(name="summarizer")
    session_manager = FileSessionManager(session_id="strands-interrupt-test", storage_dir=tmpdir)

    builder = GraphBuilder()
    builder.add_node(time_agent, "time")
    builder.add_node(summarizer_agent, "summarizer")
    builder.add_edge("time", "summarizer")
    builder.set_hook_providers([interrupt_hook])
    builder.set_session_manager(session_manager)
    graph = builder.build()

    responses = [
        {
            "interruptResponse": {
                "interruptId": interrupt.id,
                "response": "APPROVE",
            },
        },
    ]
    multiagent_result = graph(responses)

    tru_result_status = multiagent_result.status
    exp_result_status = Status.COMPLETED
    assert tru_result_status == exp_result_status

    tru_state_status = graph.state.status
    exp_state_status = Status.COMPLETED
    assert tru_state_status == exp_state_status

    assert len(multiagent_result.results) == 2
    summarizer_message = json.dumps(multiagent_result.results["summarizer"].result.message).lower()
    assert "12:01" in summarizer_message
