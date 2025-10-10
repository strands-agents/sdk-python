"""Tests for multi-agent tools functionality in ToolConfigLoader."""

from unittest.mock import Mock, patch

import pytest

from strands.experimental.config_loader.tools.tool_config_loader import (
    GraphAsToolWrapper,
    SwarmAsToolWrapper,
    ToolConfigLoader,
)


class TestConventionBasedTypeDetection:
    """Test convention-based type detection in ToolConfigLoader."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = ToolConfigLoader()

    def test_determine_config_type_swarm(self):
        """Test detection of swarm configuration."""
        config = {
            "name": "test_swarm",
            "swarm": {"agents": []},
        }
        assert self.loader._determine_config_type(config) == "swarm"

    def test_determine_config_type_graph(self):
        """Test detection of graph configuration."""
        config = {
            "name": "test_graph",
            "graph": {"nodes": [], "edges": []},
        }
        assert self.loader._determine_config_type(config) == "graph"

    def test_determine_config_type_agent(self):
        """Test detection of agent configuration."""
        config = {
            "name": "test_agent",
            "agent": {"model": "test-model"},
        }
        assert self.loader._determine_config_type(config) == "agent"

    def test_determine_config_type_legacy_tool(self):
        """Test detection of legacy tool configuration."""
        config = {
            "name": "test_tool",
            "module": "test.module",
        }
        assert self.loader._determine_config_type(config) == "legacy_tool"

    def test_determine_config_type_default(self):
        """Test default detection (agent) when no specific keys present."""
        config = {
            "name": "test_default",
            "description": "Some tool",
        }
        assert self.loader._determine_config_type(config) == "agent"

    def test_determine_config_type_priority(self):
        """Test priority order when multiple keys are present."""
        # Swarm has highest priority
        config = {
            "name": "test_priority",
            "swarm": {"agents": []},
            "graph": {"nodes": []},
            "agent": {"model": "test"},
        }
        assert self.loader._determine_config_type(config) == "swarm"

        # Graph has second priority
        config = {
            "name": "test_priority",
            "graph": {"nodes": []},
            "agent": {"model": "test"},
        }
        assert self.loader._determine_config_type(config) == "graph"


class TestSwarmAsToolWrapper:
    """Test SwarmAsToolWrapper functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_swarm = Mock()
        self.mock_swarm.return_value = "Swarm response"

    def test_swarm_wrapper_initialization(self):
        """Test SwarmAsToolWrapper initialization."""
        wrapper = SwarmAsToolWrapper(
            swarm=self.mock_swarm,
            tool_name="test_swarm",
            description="Test swarm tool",
        )

        assert wrapper.tool_name == "test_swarm"
        assert wrapper.tool_type == "swarm"
        assert wrapper._swarm == self.mock_swarm

    def test_swarm_wrapper_tool_spec(self):
        """Test SwarmAsToolWrapper tool specification generation."""
        input_schema = {
            "type": "object",
            "properties": {"topic": {"type": "string", "description": "Research topic"}},
            "required": ["topic"],
        }

        wrapper = SwarmAsToolWrapper(
            swarm=self.mock_swarm,
            tool_name="research_swarm",
            description="Research swarm tool",
            input_schema=input_schema,
        )

        spec = wrapper.tool_spec
        assert spec["name"] == "research_swarm"
        assert spec["description"] == "Research swarm tool"
        assert spec["inputSchema"]["properties"]["topic"]["type"] == "string"

    def test_swarm_wrapper_default_query_parameter(self):
        """Test that default query parameter is added when no prompt template."""
        wrapper = SwarmAsToolWrapper(
            swarm=self.mock_swarm,
            tool_name="test_swarm",
        )

        spec = wrapper.tool_spec
        assert "query" in spec["inputSchema"]["properties"]
        assert "query" in spec["inputSchema"]["required"]

    def test_swarm_wrapper_parameter_substitution(self):
        """Test parameter substitution in prompts."""
        wrapper = SwarmAsToolWrapper(
            swarm=self.mock_swarm,
            tool_name="test_swarm",
            prompt="Research {topic} with {depth} analysis",
        )

        substitutions = {"topic": "AI", "depth": "comprehensive"}
        result = wrapper._substitute_args(wrapper._prompt, substitutions)
        assert result == "Research AI with comprehensive analysis"

    @pytest.mark.asyncio
    async def test_swarm_wrapper_stream_execution(self):
        """Test SwarmAsToolWrapper stream execution."""
        wrapper = SwarmAsToolWrapper(
            swarm=self.mock_swarm,
            tool_name="test_swarm",
        )

        tool_use = {
            "toolUseId": "test_id",
            "input": {"query": "Test query"},
        }

        results = []
        async for result in wrapper.stream(tool_use, {}):
            results.append(result)

        assert len(results) == 1
        assert results[0]["status"] == "success"
        assert "Swarm response" in str(results[0]["content"][0]["text"])
        self.mock_swarm.assert_called_once_with("Test query")


class TestGraphAsToolWrapper:
    """Test GraphAsToolWrapper functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_graph = Mock()
        self.mock_graph.return_value = "Graph response"

    def test_graph_wrapper_initialization(self):
        """Test GraphAsToolWrapper initialization."""
        wrapper = GraphAsToolWrapper(
            graph=self.mock_graph,
            tool_name="test_graph",
            description="Test graph tool",
        )

        assert wrapper.tool_name == "test_graph"
        assert wrapper.tool_type == "graph"
        assert wrapper._graph == self.mock_graph

    def test_graph_wrapper_tool_spec(self):
        """Test GraphAsToolWrapper tool specification generation."""
        input_schema = {
            "type": "object",
            "properties": {"document": {"type": "string", "description": "Document to process"}},
            "required": ["document"],
        }

        wrapper = GraphAsToolWrapper(
            graph=self.mock_graph,
            tool_name="doc_processor",
            description="Document processor graph",
            input_schema=input_schema,
        )

        spec = wrapper.tool_spec
        assert spec["name"] == "doc_processor"
        assert spec["description"] == "Document processor graph"
        assert spec["inputSchema"]["properties"]["document"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_graph_wrapper_stream_execution_with_entry_point(self):
        """Test GraphAsToolWrapper stream execution with entry point."""
        wrapper = GraphAsToolWrapper(
            graph=self.mock_graph,
            tool_name="test_graph",
            entry_point="validator",
        )

        tool_use = {
            "toolUseId": "test_id",
            "input": {"query": "Test query"},
        }

        results = []
        async for result in wrapper.stream(tool_use, {}):
            results.append(result)

        assert len(results) == 1
        assert results[0]["status"] == "success"
        self.mock_graph.assert_called_once_with("Test query", entry_point="validator")


class TestToolConfigLoaderMultiAgent:
    """Test ToolConfigLoader multi-agent functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = ToolConfigLoader()

    @patch("strands.experimental.config_loader.swarm.swarm_config_loader.SwarmConfigLoader")
    def test_load_swarm_as_tool(self, mock_swarm_loader_class):
        """Test loading swarm as tool."""
        # Mock the swarm loader and swarm
        mock_swarm_loader = Mock()
        mock_swarm = Mock()
        mock_swarm_loader.load_swarm.return_value = mock_swarm
        mock_swarm_loader_class.return_value = mock_swarm_loader

        # Mock the _get_swarm_config_loader method
        self.loader._get_swarm_config_loader = Mock(return_value=mock_swarm_loader)

        config = {
            "name": "research_team",
            "description": "Research team swarm",
            "swarm": {"agents": [{"name": "researcher", "model": "test-model"}]},
        }

        tool = self.loader._load_swarm_as_tool(config)

        assert isinstance(tool, SwarmAsToolWrapper)
        assert tool.tool_name == "research_team"
        assert tool.tool_type == "swarm"
        mock_swarm_loader.load_swarm.assert_called_once()

    @patch("strands.experimental.config_loader.graph.graph_config_loader.GraphConfigLoader")
    def test_load_graph_as_tool(self, mock_graph_loader_class):
        """Test loading graph as tool."""
        # Mock the graph loader and graph
        mock_graph_loader = Mock()
        mock_graph = Mock()
        mock_graph_loader.load_graph.return_value = mock_graph
        mock_graph_loader_class.return_value = mock_graph_loader

        # Mock the _get_graph_config_loader method
        self.loader._get_graph_config_loader = Mock(return_value=mock_graph_loader)

        config = {
            "name": "doc_processor",
            "description": "Document processor graph",
            "graph": {
                "nodes": [{"node_id": "validator", "type": "agent"}],
                "edges": [],
                "entry_points": ["validator"],
            },
        }

        tool = self.loader._load_graph_as_tool(config)

        assert isinstance(tool, GraphAsToolWrapper)
        assert tool.tool_name == "doc_processor"
        assert tool.tool_type == "graph"
        mock_graph_loader.load_graph.assert_called_once()

    def test_load_config_tool_dispatch(self):
        """Test that _load_config_tool dispatches to correct loader based on type."""
        with patch.object(self.loader, "_load_swarm_as_tool") as mock_swarm:
            config = {"name": "test", "swarm": {}}
            self.loader._load_config_tool(config)
            mock_swarm.assert_called_once_with(config)

        with patch.object(self.loader, "_load_graph_as_tool") as mock_graph:
            config = {"name": "test", "graph": {}}
            self.loader._load_config_tool(config)
            mock_graph.assert_called_once_with(config)

        with patch.object(self.loader, "_load_agent_as_tool") as mock_agent:
            config = {"name": "test", "agent": {}}
            self.loader._load_config_tool(config)
            mock_agent.assert_called_once_with(config)

    def test_load_tool_with_dict_config(self):
        """Test load_tool with dictionary configuration."""
        with patch.object(self.loader, "_load_config_tool") as mock_load_config:
            mock_tool = Mock()
            mock_load_config.return_value = mock_tool

            config = {"name": "test", "swarm": {}}
            result = self.loader.load_tool(config)

            assert result == mock_tool
            mock_load_config.assert_called_once_with(config)

    def test_load_tools_with_mixed_configs(self):
        """Test load_tools with mixed configuration types."""
        with patch.object(self.loader, "load_tool") as mock_load_tool:
            mock_tool1 = Mock()
            mock_tool2 = Mock()
            mock_load_tool.side_effect = [mock_tool1, mock_tool2]

            configs = [
                {"name": "swarm_tool", "swarm": {}},
                "string_tool",
            ]

            result = self.loader.load_tools(configs)

            assert len(result) == 2
            assert result[0] == mock_tool1
            assert result[1] == mock_tool2
            assert mock_load_tool.call_count == 2

    def test_validation_errors(self):
        """Test validation errors for invalid configurations."""
        # Missing name field
        with pytest.raises(ValueError, match="must include 'name' field"):
            self.loader._load_swarm_as_tool({"swarm": {}})

        # Missing swarm field
        with pytest.raises(ValueError, match="must include 'swarm' field"):
            self.loader._load_swarm_as_tool({"name": "test"})

        # Missing graph field
        with pytest.raises(ValueError, match="must include 'graph' field"):
            self.loader._load_graph_as_tool({"name": "test"})

    def test_multiple_loads_create_independent_objects(self):
        """Test that multiple loads create independent tool objects."""
        mock_swarm_loader = Mock()
        mock_swarm = Mock()
        mock_swarm_loader.load_swarm.return_value = mock_swarm

        # Mock the _get_swarm_config_loader method
        self.loader._get_swarm_config_loader = Mock(return_value=mock_swarm_loader)

        config = {
            "name": "test_swarm",
            "swarm": {"agents": []},
        }

        # Load tool twice
        tool1 = self.loader._load_swarm_as_tool(config)
        tool2 = self.loader._load_swarm_as_tool(config)

        # Should create different tool wrapper objects each time
        assert tool1 is not tool2
        # Swarm loader should be called each time
        assert mock_swarm_loader.load_swarm.call_count == 2
