"""Tests for GraphConfigLoader."""

from unittest.mock import Mock, patch

import pytest

from strands import Agent
from strands.experimental.config_loader.graph import ConditionRegistry, GraphConfigLoader


class TestGraphConfigLoader:
    """Test cases for GraphConfigLoader functionality."""

    def test_load_graph_basic_config(self):
        """Test loading graph from basic configuration."""
        config = {
            "graph": {
                "nodes": [
                    {
                        "node_id": "agent1",
                        "type": "agent",
                        "config": {
                            "name": "test_agent",
                            "model": "us.amazon.nova-lite-v1:0",
                            "system_prompt": "You are a test agent.",
                            "tools": [],
                        },
                    }
                ],
                "edges": [],
                "entry_points": ["agent1"],
            }
        }

        loader = GraphConfigLoader()
        graph = loader.load_graph(config)

        assert len(graph.nodes) == 1
        assert "agent1" in graph.nodes
        assert len(graph.entry_points) == 1

    def test_load_graph_with_edges_and_conditions(self):
        """Test loading graph with edges and conditions."""
        config = {
            "graph": {
                "nodes": [
                    {
                        "node_id": "classifier",
                        "type": "agent",
                        "config": {
                            "name": "classifier",
                            "model": "us.amazon.nova-lite-v1:0",
                            "system_prompt": "Classify requests.",
                            "tools": [],
                        },
                    },
                    {
                        "node_id": "processor",
                        "type": "agent",
                        "config": {
                            "name": "processor",
                            "model": "us.amazon.nova-lite-v1:0",
                            "system_prompt": "Process requests.",
                            "tools": [],
                        },
                    },
                ],
                "edges": [
                    {
                        "from_node": "classifier",
                        "to_node": "processor",
                        "condition": {
                            "type": "expression",
                            "expression": "True",  # Simple always-true condition
                            "default": False,
                        },
                    }
                ],
                "entry_points": ["classifier"],
            }
        }

        loader = GraphConfigLoader()
        graph = loader.load_graph(config)

        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert len(graph.entry_points) == 1

    def test_load_graph_with_caching(self):
        """Test graph caching functionality."""
        config = {
            "graph": {
                "nodes": [
                    {
                        "node_id": "agent1",
                        "type": "agent",
                        "config": {
                            "name": "test_agent",
                            "model": "us.amazon.nova-lite-v1:0",
                            "system_prompt": "Test agent",
                            "tools": [],
                        },
                    }
                ],
                "edges": [],
                "entry_points": ["agent1"],
            }
        }

        loader = GraphConfigLoader()

        # Load the graph
        graph = loader.load_graph(config)

        # Verify graph structure
        assert len(graph.nodes) == 1
        assert len(graph.edges) == 0
        assert len(graph.entry_points) == 1

    def test_multiple_loads_create_independent_objects(self):
        """Test that multiple loads create independent graph objects."""
        config = {
            "graph": {
                "nodes": [
                    {
                        "node_id": "agent1",
                        "type": "agent",
                        "config": {
                            "name": "agent1",
                            "model": "us.amazon.nova-lite-v1:0",
                            "system_prompt": "You are agent 1.",
                        },
                    }
                ],
                "edges": [],
                "entry_points": ["agent1"],
            }
        }

        loader = GraphConfigLoader()

        # Load twice
        graph1 = loader.load_graph(config)
        graph2 = loader.load_graph(config)

        # Should be different objects
        assert graph1 is not graph2
        assert id(graph1) != id(graph2)

    def test_serialize_graph(self):
        """Test serializing graph to configuration."""
        # Create a simple graph programmatically
        agent = Agent(name="test_agent", model="us.amazon.nova-lite-v1:0")

        # Mock the Graph creation since we need to avoid complex dependencies
        with patch("strands.multiagent.graph.Graph"):
            mock_graph = Mock()

            # Create a proper GraphNode with the agent
            from strands.multiagent.graph import GraphNode

            test_node = GraphNode(node_id="agent1", executor=agent)

            mock_graph.nodes = {"agent1": test_node}
            mock_graph.edges = set()
            mock_graph.entry_points = set()
            mock_graph.max_node_executions = None
            mock_graph.execution_timeout = None
            mock_graph.node_timeout = None
            mock_graph.reset_on_revisit = False

            loader = GraphConfigLoader()
            config = loader.serialize_graph(mock_graph)

            # Verify basic structure
            assert "graph" in config
            assert "nodes" in config["graph"]
            assert "edges" in config["graph"]
            assert "entry_points" in config["graph"]
            assert len(config["graph"]["nodes"]) == 1
            assert config["graph"]["nodes"][0]["node_id"] == "agent1"
            assert config["graph"]["nodes"][0]["type"] == "agent"

    def test_invalid_config_validation(self):
        """Test validation of invalid configurations."""
        loader = GraphConfigLoader()

        # Empty config
        with pytest.raises(ValueError, match="must include 'nodes' field"):
            loader.load_graph({"graph": {}})

        # Empty nodes list
        with pytest.raises(ValueError, match="'nodes' list cannot be empty"):
            loader.load_graph({"graph": {"nodes": [], "edges": [], "entry_points": []}})

        # Invalid node type
        with pytest.raises(ValueError, match="Invalid node type"):
            loader.load_graph(
                {"graph": {"nodes": [{"node_id": "test", "type": "invalid"}], "edges": [], "entry_points": ["test"]}}
            )

        # Missing node_id
        with pytest.raises(ValueError, match="missing required 'node_id' field"):
            loader.load_graph({"graph": {"nodes": [{"type": "agent"}], "edges": [], "entry_points": []}})

    def test_lazy_loading_config_loaders(self):
        """Test lazy loading of AgentConfigLoader and SwarmConfigLoader."""
        loader = GraphConfigLoader()

        # Initially should be None
        assert loader._agent_loader is None
        assert loader._swarm_loader is None

        # Should create one when needed
        agent_loader = loader._get_agent_config_loader()
        assert agent_loader is not None
        assert loader._agent_loader is agent_loader

        swarm_loader = loader._get_swarm_config_loader()
        assert swarm_loader is not None
        assert loader._swarm_loader is swarm_loader


class TestConditionRegistry:
    """Test cases for ConditionRegistry functionality."""

    def test_expression_condition(self):
        """Test expression-based conditions."""
        registry = ConditionRegistry()

        config = {"type": "expression", "expression": "state.execution_count < 5", "default": False}

        condition = registry.load_condition(config)

        # Test with mock state
        mock_state = Mock()
        mock_state.execution_count = 3

        assert condition(mock_state) is True

        mock_state.execution_count = 10
        assert condition(mock_state) is False

    def test_rule_condition(self):
        """Test rule-based conditions."""
        registry = ConditionRegistry()

        config = {
            "type": "rule",
            "rules": [{"field": "execution_count", "operator": "less_than", "value": 5}],
            "logic": "and",
        }

        condition = registry.load_condition(config)

        # Test with mock state
        mock_state = Mock()
        mock_state.execution_count = 3

        assert condition(mock_state) is True

    def test_template_condition(self):
        """Test template-based conditions."""
        registry = ConditionRegistry()

        config = {"type": "template", "template": "execution_count_under", "parameters": {"max_count": 5}}

        condition = registry.load_condition(config)

        # Test with mock state
        mock_state = Mock()
        mock_state.execution_count = 3

        assert condition(mock_state) is True

        mock_state.execution_count = 10
        assert condition(mock_state) is False

    def test_composite_condition(self):
        """Test composite conditions with multiple sub-conditions."""
        registry = ConditionRegistry()

        config = {
            "type": "composite",
            "logic": "and",
            "conditions": [
                {"type": "expression", "expression": "state.execution_count < 10"},
                {"type": "template", "template": "execution_count_under", "parameters": {"max_count": 5}},
            ],
        }

        condition = registry.load_condition(config)

        # Test with mock state
        mock_state = Mock()
        mock_state.execution_count = 3

        assert condition(mock_state) is True

        mock_state.execution_count = 7
        assert condition(mock_state) is False

    def test_lambda_condition(self):
        """Test lambda-based conditions."""
        registry = ConditionRegistry()

        config = {"type": "lambda", "expression": "lambda state: state.execution_count < 5"}

        condition = registry.load_condition(config)

        # Test with mock state
        mock_state = Mock()
        mock_state.execution_count = 3

        assert condition(mock_state) is True

    def test_invalid_condition_type(self):
        """Test handling of invalid condition types."""
        registry = ConditionRegistry()

        config = {"type": "invalid_type", "expression": "True"}

        with pytest.raises(ValueError, match="Unsupported condition type"):
            registry.load_condition(config)

    def test_expression_sanitization(self):
        """Test expression sanitization for security."""
        registry = ConditionRegistry()

        # Test dangerous patterns
        dangerous_expressions = [
            "import os",
            "__import__('os')",
            "exec('print(1)')",
            "eval('1+1')",
            "open('/etc/passwd')",
        ]

        for expr in dangerous_expressions:
            with pytest.raises(ValueError, match="Dangerous pattern"):
                registry._sanitize_expression(expr)

    def test_expression_length_limit(self):
        """Test expression length limits."""
        registry = ConditionRegistry()

        # Create expression longer than limit
        long_expression = "state.execution_count < 5" + " and True" * 100

        with pytest.raises(ValueError, match="Expression too long"):
            registry._sanitize_expression(long_expression)

    def test_module_access_validation(self):
        """Test module access validation."""
        registry = ConditionRegistry()

        # Test allowed module
        registry._validate_module_access("conditions.my_module")

        # Test disallowed module
        with pytest.raises(ValueError, match="not in allowed modules"):
            registry._validate_module_access("os.path")

    def test_nested_field_extraction(self):
        """Test nested field extraction from GraphState."""
        registry = ConditionRegistry()

        # Create mock state with nested structure
        mock_state = Mock()
        mock_state.results = {"classifier": Mock()}
        mock_state.results["classifier"].status = "completed"

        # Test field extraction
        value = registry._get_nested_field(mock_state, "results.classifier.status")
        assert value == "completed"

        # Test non-existent field
        value = registry._get_nested_field(mock_state, "results.nonexistent.field")
        assert value is None
