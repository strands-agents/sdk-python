import pytest

from strands.agent import AgentResult
from strands.multiagent.base import MultiAgentBase, MultiAgentResult, NodeResult, Status


@pytest.fixture
def agent_result():
    """Create a mock AgentResult for testing."""
    return AgentResult(
        message={"role": "assistant", "content": [{"text": "Test response"}]},
        stop_reason="end_turn",
        state={},
        metrics={},
    )


def test_node_result_initialization_and_properties(agent_result):
    """Test NodeResult initialization and property access."""
    # Basic initialization
    node_result = NodeResult(result=agent_result, execution_time=50, status="completed")

    # Verify properties
    assert node_result.result == agent_result
    assert node_result.execution_time == 50
    assert node_result.status == "completed"
    assert node_result.accumulated_usage == {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
    assert node_result.accumulated_metrics == {"latencyMs": 0.0}
    assert node_result.execution_count == 0

    # With custom metrics
    custom_usage = {"inputTokens": 100, "outputTokens": 200, "totalTokens": 300}
    custom_metrics = {"latencyMs": 250.0}
    node_result_custom = NodeResult(
        result=agent_result,
        execution_time=75,
        status="completed",
        accumulated_usage=custom_usage,
        accumulated_metrics=custom_metrics,
        execution_count=5,
    )
    assert node_result_custom.accumulated_usage == custom_usage
    assert node_result_custom.accumulated_metrics == custom_metrics
    assert node_result_custom.execution_count == 5

    # Test default factory creates independent instances
    node_result1 = NodeResult(result=agent_result)
    node_result2 = NodeResult(result=agent_result)
    node_result1.accumulated_usage["inputTokens"] = 100
    assert node_result2.accumulated_usage["inputTokens"] == 0
    assert node_result1.accumulated_usage is not node_result2.accumulated_usage


def test_node_result_get_agent_results(agent_result):
    """Test get_agent_results method with different structures."""
    # Simple case with single AgentResult
    node_result = NodeResult(result=agent_result)
    agent_results = node_result.get_agent_results()
    assert len(agent_results) == 1
    assert agent_results[0] == agent_result

    # Test with Exception as result (should return empty list)
    exception_result = NodeResult(result=Exception("Test exception"), status=Status.FAILED)
    agent_results = exception_result.get_agent_results()
    assert len(agent_results) == 0

    # Complex nested case
    inner_agent_result1 = AgentResult(
        message={"role": "assistant", "content": [{"text": "Response 1"}]}, stop_reason="end_turn", state={}, metrics={}
    )
    inner_agent_result2 = AgentResult(
        message={"role": "assistant", "content": [{"text": "Response 2"}]}, stop_reason="end_turn", state={}, metrics={}
    )

    inner_node_result1 = NodeResult(result=inner_agent_result1)
    inner_node_result2 = NodeResult(result=inner_agent_result2)

    multi_agent_result = MultiAgentResult(results={"node1": inner_node_result1, "node2": inner_node_result2})

    outer_node_result = NodeResult(result=multi_agent_result)
    agent_results = outer_node_result.get_agent_results()

    assert len(agent_results) == 2
    response_texts = [result.message["content"][0]["text"] for result in agent_results]
    assert "Response 1" in response_texts
    assert "Response 2" in response_texts


def test_multi_agent_result_initialization(agent_result):
    """Test MultiAgentResult initialization with defaults and custom values."""
    # Default initialization
    result = MultiAgentResult(results={})
    assert result.results == {}
    assert result.accumulated_usage == {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
    assert result.accumulated_metrics == {"latencyMs": 0.0}
    assert result.execution_count == 0
    assert result.execution_time == 0

    # Custom values``
    node_result = NodeResult(result=agent_result)
    results = {"test_node": node_result}
    usage = {"inputTokens": 50, "outputTokens": 100, "totalTokens": 150}
    metrics = {"latencyMs": 200.0}

    result = MultiAgentResult(
        results=results, accumulated_usage=usage, accumulated_metrics=metrics, execution_count=3, execution_time=300
    )

    assert result.results == results
    assert result.accumulated_usage == usage
    assert result.accumulated_metrics == metrics
    assert result.execution_count == 3
    assert result.execution_time == 300

    # Test default factory creates independent instances
    result1 = MultiAgentResult(results={})
    result2 = MultiAgentResult(results={})
    result1.accumulated_usage["inputTokens"] = 200
    result1.accumulated_metrics["latencyMs"] = 500.0
    assert result2.accumulated_usage["inputTokens"] == 0
    assert result2.accumulated_metrics["latencyMs"] == 0.0
    assert result1.accumulated_usage is not result2.accumulated_usage
    assert result1.accumulated_metrics is not result2.accumulated_metrics


def test_multi_agent_base_abstract_behavior():
    """Test abstract class behavior of MultiAgentBase."""
    # Test that MultiAgentBase cannot be instantiated directly
    with pytest.raises(TypeError):
        MultiAgentBase()

    # Test that incomplete implementations raise TypeError
    class IncompleteMultiAgent(MultiAgentBase):
        pass

    with pytest.raises(TypeError):
        IncompleteMultiAgent()

    # Test that complete implementations can be instantiated
    class CompleteMultiAgent(MultiAgentBase):
        async def invoke_async(self, task: str) -> MultiAgentResult:
            return MultiAgentResult(results={})

    # Should not raise an exception - __call__ is provided by base class
    agent = CompleteMultiAgent()
    assert isinstance(agent, MultiAgentBase)


def test_multi_agent_base_call_method():
    """Test that __call__ method properly delegates to invoke_async."""

    class TestMultiAgent(MultiAgentBase):
        def __init__(self):
            self.invoke_async_called = False
            self.received_task = None
            self.received_kwargs = None

        async def invoke_async(self, task, invocation_state, **kwargs):
            self.invoke_async_called = True
            self.received_task = task
            self.received_kwargs = kwargs
            return MultiAgentResult(
                status=Status.COMPLETED, results={"test": NodeResult(result=Exception("test"), status=Status.COMPLETED)}
            )

    agent = TestMultiAgent()

    # Test with string task
    result = agent("test task", param1="value1", param2="value2")

    assert agent.invoke_async_called
    assert agent.received_task == "test task"
    assert agent.received_kwargs == {"param1": "value1", "param2": "value2"}
    assert isinstance(result, MultiAgentResult)
    assert result.status == Status.COMPLETED
