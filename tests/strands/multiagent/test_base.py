"""Tests for MultiAgentBase module."""

import pytest

from strands.multiagent.base import SharedContext


def test_shared_context_initialization():
    """Test SharedContext initialization."""
    context = SharedContext()
    assert context.context == {}

    # Test with initial context
    initial_context = {"node1": {"key1": "value1"}}
    context = SharedContext(initial_context)
    assert context.context == initial_context


def test_shared_context_add_context():
    """Test adding context to SharedContext."""
    context = SharedContext()
    
    # Create mock nodes
    node1 = type('MockNode', (), {'node_id': 'node1'})()
    node2 = type('MockNode', (), {'node_id': 'node2'})()
    
    # Add context for a node
    context.add_context(node1, "key1", "value1")
    assert context.context["node1"]["key1"] == "value1"
    
    # Add more context for the same node
    context.add_context(node1, "key2", "value2")
    assert context.context["node1"]["key1"] == "value1"
    assert context.context["node1"]["key2"] == "value2"
    
    # Add context for a different node
    context.add_context(node2, "key1", "value3")
    assert context.context["node2"]["key1"] == "value3"
    assert "node2" not in context.context["node1"]


def test_shared_context_get_context():
    """Test getting context from SharedContext."""
    context = SharedContext()
    
    # Create mock nodes
    node1 = type('MockNode', (), {'node_id': 'node1'})()
    node2 = type('MockNode', (), {'node_id': 'node2'})()
    non_existent_node = type('MockNode', (), {'node_id': 'non_existent_node'})()
    
    # Add some test data
    context.add_context(node1, "key1", "value1")
    context.add_context(node1, "key2", "value2")
    context.add_context(node2, "key1", "value3")
    
    # Get specific key
    assert context.get_context(node1, "key1") == "value1"
    assert context.get_context(node1, "key2") == "value2"
    assert context.get_context(node2, "key1") == "value3"
    
    # Get all context for a node
    node1_context = context.get_context(node1)
    assert node1_context == {"key1": "value1", "key2": "value2"}
    
    # Get context for non-existent node
    assert context.get_context(non_existent_node) == {}
    assert context.get_context(non_existent_node, "key") is None


def test_shared_context_validation():
    """Test SharedContext input validation."""
    context = SharedContext()
    
    # Create mock node
    node1 = type('MockNode', (), {'node_id': 'node1'})()
    
    # Test invalid key validation
    with pytest.raises(ValueError, match="Key cannot be None"):
        context.add_context(node1, None, "value")
    
    with pytest.raises(ValueError, match="Key must be a string"):
        context.add_context(node1, 123, "value")
    
    with pytest.raises(ValueError, match="Key cannot be empty"):
        context.add_context(node1, "", "value")
    
    with pytest.raises(ValueError, match="Key cannot be empty"):
        context.add_context(node1, "   ", "value")
    
    # Test JSON serialization validation
    with pytest.raises(ValueError, match="Value is not JSON serializable"):
        context.add_context(node1, "key", lambda x: x)  # Function not serializable
    
    # Test valid values
    context.add_context(node1, "string", "hello")
    context.add_context(node1, "number", 42)
    context.add_context(node1, "boolean", True)
    context.add_context(node1, "list", [1, 2, 3])
    context.add_context(node1, "dict", {"nested": "value"})
    context.add_context(node1, "none", None)


def test_shared_context_isolation():
    """Test that SharedContext provides proper isolation between nodes."""
    context = SharedContext()
    
    # Create mock nodes
    node1 = type('MockNode', (), {'node_id': 'node1'})()
    node2 = type('MockNode', (), {'node_id': 'node2'})()
    
    # Add context for different nodes
    context.add_context(node1, "key1", "value1")
    context.add_context(node2, "key1", "value2")
    
    # Ensure nodes don't interfere with each other
    assert context.get_context(node1, "key1") == "value1"
    assert context.get_context(node2, "key1") == "value2"
    
    # Getting all context for a node should only return that node's context
    assert context.get_context(node1) == {"key1": "value1"}
    assert context.get_context(node2) == {"key1": "value2"}


def test_shared_context_copy_semantics():
    """Test that SharedContext.get_context returns copies to prevent mutation."""
    context = SharedContext()
    
    # Create mock node
    node1 = type('MockNode', (), {'node_id': 'node1'})()
    
    # Add a mutable value
    context.add_context(node1, "mutable", [1, 2, 3])
    
    # Get the context and modify it
    retrieved_context = context.get_context(node1)
    retrieved_context["mutable"].append(4)
    
    # The original should remain unchanged
    assert context.get_context(node1, "mutable") == [1, 2, 3]
    
    # Test that getting all context returns a copy
    all_context = context.get_context(node1)
    all_context["new_key"] = "new_value"
    
    # The original should remain unchanged
    assert "new_key" not in context.get_context(node1)


def test_multi_agent_base_abstract():
    """Test that MultiAgentBase cannot be instantiated directly."""
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
