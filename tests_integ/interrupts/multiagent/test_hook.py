import json
from unittest.mock import ANY

import pytest

from strands import Agent, tool
from strands.experimental.hooks.multiagent import BeforeNodeCallEvent
from strands.hooks import HookProvider
from strands.interrupt import Interrupt
from strands.multiagent import Swarm
from strands.multiagent.base import Status


@pytest.fixture
def interrupt_hook():
    class Hook(HookProvider):
        def register_hooks(self, registry):
            registry.add_callback(BeforeNodeCallEvent, self.interrupt)

        def interrupt(self, event):
            if event.node_id == "info":
                return

            response = event.interrupt("test_interrupt", reason="need approval")
            if response != "APPROVE":
                event.cancel_node = "node rejected"

    return Hook()


@pytest.fixture
def weather_tool():
    @tool(name="weather_tool")
    def func():
        return "sunny"

    return func


@pytest.fixture
def swarm(interrupt_hook, weather_tool):
    info_agent = Agent(name="info")
    weather_agent = Agent(name="weather", tools=[weather_tool])

    return Swarm([info_agent, weather_agent], hooks=[interrupt_hook])


def test_swarm_interrupt(swarm):
    multiagent_result = swarm("What is the weather?")

    tru_status = multiagent_result.status
    exp_status = Status.INTERRUPTED
    assert tru_status == exp_status

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

    responses = [
        {
            "interruptResponse": {
                "interruptId": interrupt.id,
                "response": "APPROVE",
            },
        },
    ]
    multiagent_result = swarm(responses)

    tru_status = multiagent_result.status
    exp_status = Status.COMPLETED
    assert tru_status == exp_status

    assert len(multiagent_result.results) == 2
    weather_result = multiagent_result.results["weather"]

    weather_message = json.dumps(weather_result.result.message).lower()
    assert "sunny" in weather_message


@pytest.mark.asyncio
async def test_swarm_interrupt_reject(swarm):
    multiagent_result = swarm("What is the weather?")

    tru_status = multiagent_result.status
    exp_status = Status.INTERRUPTED
    assert tru_status == exp_status

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

    responses = [
        {
            "interruptResponse": {
                "interruptId": interrupt.id,
                "response": "REJECT",
            },
        },
    ]
    tru_cancel_id = None
    async for event in swarm.stream_async(responses):
        if event.get("type") == "multiagent_node_cancel":
            tru_cancel_id = event["node_id"]

    multiagent_result = event["result"]

    exp_cancel_id = "weather"
    assert tru_cancel_id == exp_cancel_id

    tru_status = multiagent_result.status
    exp_status = Status.FAILED
    assert tru_status == exp_status

    assert len(multiagent_result.node_history) == 1
    tru_node_id = multiagent_result.node_history[0].node_id
    exp_node_id = "info"
    assert tru_node_id == exp_node_id


# Graph tests


@pytest.fixture
def graph(interrupt_hook, weather_tool):
    from strands.multiagent import GraphBuilder

    info_agent = Agent(name="info")
    weather_agent = Agent(name="weather", tools=[weather_tool])

    builder = GraphBuilder()
    builder.add_node(info_agent, "info")
    builder.add_node(weather_agent, "weather")
    builder.add_edge("info", "weather")
    builder.set_entry_point("info")
    builder.set_hook_providers([interrupt_hook])

    return builder.build()


def test_graph_interrupt(graph):
    multiagent_result = graph("What is the weather?")

    tru_status = multiagent_result.status
    exp_status = Status.INTERRUPTED
    assert tru_status == exp_status

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

    responses = [
        {
            "interruptResponse": {
                "interruptId": interrupt.id,
                "response": "APPROVE",
            },
        },
    ]
    multiagent_result = graph(responses)

    tru_status = multiagent_result.status
    exp_status = Status.COMPLETED
    assert tru_status == exp_status

    assert len(multiagent_result.results) == 2
    weather_result = multiagent_result.results["weather"]

    weather_message = json.dumps(weather_result.result.message).lower()
    assert "sunny" in weather_message


@pytest.mark.asyncio
async def test_graph_interrupt_reject(graph):
    multiagent_result = graph("What is the weather?")

    tru_status = multiagent_result.status
    exp_status = Status.INTERRUPTED
    assert tru_status == exp_status

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

    responses = [
        {
            "interruptResponse": {
                "interruptId": interrupt.id,
                "response": "REJECT",
            },
        },
    ]
    tru_cancel_id = None

    # Graph raises RuntimeError for cancel_node
    with pytest.raises(RuntimeError, match="node rejected"):
        async for event in graph.stream_async(responses):
            if event.get("type") == "multiagent_node_cancel":
                tru_cancel_id = event["node_id"]

    exp_cancel_id = "weather"
    assert tru_cancel_id == exp_cancel_id

    tru_status = graph.state.status
    exp_status = Status.FAILED
    assert tru_status == exp_status

    assert len(graph.state.execution_order) == 1
    tru_node_id = graph.state.execution_order[0].node_id
    exp_node_id = "info"
    assert tru_node_id == exp_node_id
