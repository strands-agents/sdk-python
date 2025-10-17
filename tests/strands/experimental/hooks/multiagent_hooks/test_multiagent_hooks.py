import pytest

from strands import Agent
from strands.experimental.hooks.multiagent_hooks import (
    AfterMultiAgentInvocationEvent,
    AfterNodeCallEvent,
    BeforeNodeCallEvent,
    MultiAgentInitializedEvent,
)
from strands.multiagent.graph import Graph, GraphBuilder
from strands.multiagent.swarm import Swarm
from tests.fixtures.mock_multiagent_hook_provider import MockMultiAgentHookProvider
from tests.fixtures.mocked_model_provider import MockedModelProvider


@pytest.fixture
def hook_provider():
    return MockMultiAgentHookProvider(
        [
            AfterMultiAgentInvocationEvent,
            AfterNodeCallEvent,
            BeforeNodeCallEvent,
            MultiAgentInitializedEvent,
        ]
    )


@pytest.fixture
def mock_model():
    agent_messages = [
        {"role": "assistant", "content": [{"text": "Task completed"}]},
        {"role": "assistant", "content": [{"text": "Task completed by agent 2"}]},
        {"role": "assistant", "content": [{"text": "Additional response"}]},
    ]
    return MockedModelProvider(agent_messages)


@pytest.fixture
def agent0(mock_model):
    return Agent(model=mock_model, system_prompt="You are a helpful assistant.", name="agent0")


@pytest.fixture
def agent1(mock_model):
    return Agent(model=mock_model, system_prompt="You are agent 1.", name="agent1")


@pytest.fixture
def agent2(mock_model):
    return Agent(model=mock_model, system_prompt="You are agent 2.", name="agent2")


@pytest.fixture
def swarm(agent1, agent2, hook_provider):
    swarm = Swarm(nodes=[agent1, agent2], hooks=[hook_provider])
    return swarm


@pytest.fixture
def graph(agent1, agent2, hook_provider):
    builder = GraphBuilder()
    builder.add_node(agent1, "agent1")
    builder.add_node(agent2, "agent2")
    builder.add_edge("agent1", "agent2")
    builder.set_entry_point("agent1")
    graph = Graph(nodes=builder.nodes, edges=builder.edges, entry_points=builder.entry_points, hooks=[hook_provider])
    return graph


def test_swarm_complete_hook_lifecycle(swarm, hook_provider):
    """E2E test verifying complete hook lifecycle for Swarm."""
    result = swarm("test task")

    length, events = hook_provider.get_events()
    assert length == 4
    assert result.status.value == "completed"

    assert next(events) == MultiAgentInitializedEvent(source=swarm)
    assert next(events) == BeforeNodeCallEvent(source=swarm, node_id="agent1")
    assert next(events) == AfterNodeCallEvent(source=swarm, node_id="agent1")
    assert next(events) == AfterMultiAgentInvocationEvent(source=swarm)


def test_graph_complete_hook_lifecycle(graph, hook_provider):
    """E2E test verifying complete hook lifecycle for Graph."""
    result = graph("test task")

    length, events = hook_provider.get_events()
    assert length == 6
    assert result.status.value == "completed"

    assert next(events) == MultiAgentInitializedEvent(source=graph)
    assert next(events) == BeforeNodeCallEvent(source=graph, node_id="agent1")
    assert next(events) == AfterNodeCallEvent(source=graph, node_id="agent1")
    assert next(events) == BeforeNodeCallEvent(source=graph, node_id="agent2")
    assert next(events) == AfterNodeCallEvent(source=graph, node_id="agent2")
    assert next(events) == AfterMultiAgentInvocationEvent(source=graph)
