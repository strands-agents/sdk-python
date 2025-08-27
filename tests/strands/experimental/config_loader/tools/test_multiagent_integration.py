"""Integration tests for multi-agent tools functionality."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from strands.experimental.config_loader.tools.tool_config_loader import ToolConfigLoader


class TestMultiAgentToolsIntegration:
    """Integration tests for multi-agent tools."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = ToolConfigLoader()

    def create_temp_config_file(self, config_data):
        """Create a temporary YAML configuration file."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
        yaml.dump(config_data, temp_file, default_flow_style=False)
        temp_file.close()
        return temp_file.name

    def test_swarm_tool_end_to_end(self):
        """Test end-to-end swarm tool loading and configuration."""
        # Mock the swarm loader and swarm
        mock_swarm_loader = Mock()
        mock_swarm = Mock()
        mock_swarm.return_value = "Research completed successfully"
        mock_swarm_loader.load_swarm.return_value = mock_swarm

        # Mock the _get_swarm_config_loader method
        self.loader._get_swarm_config_loader = Mock(return_value=mock_swarm_loader)

        # Create test configuration
        config_data = {
            "tools": [
                {
                    "name": "research_team",
                    "description": "Multi-agent research team",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string", "description": "Research topic"},
                            "depth": {
                                "type": "string",
                                "enum": ["basic", "detailed", "comprehensive"],
                                "default": "detailed",
                            },
                        },
                        "required": ["topic"],
                    },
                    "prompt": "Research the topic: {topic} with {depth} analysis",
                    "entry_agent": "coordinator",
                    "swarm": {
                        "max_handoffs": 10,
                        "agents": [
                            {
                                "name": "coordinator",
                                "model": "test-model",
                                "system_prompt": "You coordinate research tasks",
                            },
                            {"name": "researcher", "model": "test-model", "system_prompt": "You conduct research"},
                        ],
                    },
                }
            ]
        }

        # Create temporary config file
        config_file = self.create_temp_config_file(config_data)

        try:
            # Load configuration
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            # Load the swarm tool
            swarm_tool = self.loader.load_tool(config["tools"][0])

            # Verify tool properties
            assert swarm_tool.tool_name == "research_team"
            assert swarm_tool.tool_type == "swarm"
            assert "Multi-agent research team" in swarm_tool.tool_spec["description"]

            # Verify input schema
            spec = swarm_tool.tool_spec
            assert "topic" in spec["inputSchema"]["properties"]
            assert "depth" in spec["inputSchema"]["properties"]
            assert spec["inputSchema"]["properties"]["depth"]["default"] == "detailed"

            # Test tool execution
            tool_use = {
                "toolUseId": "test_execution",
                "input": {"topic": "Artificial Intelligence", "depth": "comprehensive"},
            }

            # Execute the tool
            results = []
            import asyncio

            async def run_tool():
                async for result in swarm_tool.stream(tool_use, {}):
                    results.append(result)

            asyncio.run(run_tool())

            # Verify execution
            assert len(results) == 1
            assert results[0]["status"] == "success"
            mock_swarm.assert_called_once_with(
                "Research the topic: Artificial Intelligence with comprehensive analysis", entry_agent="coordinator"
            )

        finally:
            # Clean up
            Path(config_file).unlink()

    def test_graph_tool_end_to_end(self):
        """Test end-to-end graph tool loading and configuration."""
        # Mock the graph loader and graph
        mock_graph_loader = Mock()
        mock_graph = Mock()
        mock_graph.return_value = "Document processed successfully"
        mock_graph_loader.load_graph.return_value = mock_graph

        # Mock the _get_graph_config_loader method
        self.loader._get_graph_config_loader = Mock(return_value=mock_graph_loader)

        # Create test configuration
        config_data = {
            "tools": [
                {
                    "name": "document_processor",
                    "description": "Document processing pipeline",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "document": {"type": "string", "description": "Document to process"},
                            "output_format": {
                                "type": "string",
                                "enum": ["summary", "analysis", "report"],
                                "default": "summary",
                            },
                        },
                        "required": ["document"],
                    },
                    "prompt": "Process this document: {document} and generate {output_format}",
                    "entry_point": "validator",
                    "graph": {
                        "max_node_executions": 20,
                        "nodes": [
                            {
                                "node_id": "validator",
                                "type": "agent",
                                "config": {
                                    "name": "validator",
                                    "model": "test-model",
                                    "system_prompt": "Validate documents",
                                },
                            },
                            {
                                "node_id": "processor",
                                "type": "agent",
                                "config": {
                                    "name": "processor",
                                    "model": "test-model",
                                    "system_prompt": "Process documents",
                                },
                            },
                        ],
                        "edges": [{"from_node": "validator", "to_node": "processor"}],
                        "entry_points": ["validator"],
                    },
                }
            ]
        }

        # Create temporary config file
        config_file = self.create_temp_config_file(config_data)

        try:
            # Load configuration
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            # Load the graph tool
            graph_tool = self.loader.load_tool(config["tools"][0])

            # Verify tool properties
            assert graph_tool.tool_name == "document_processor"
            assert graph_tool.tool_type == "graph"
            assert "Document processing pipeline" in graph_tool.tool_spec["description"]

            # Test tool execution
            tool_use = {
                "toolUseId": "test_execution",
                "input": {"document": "Sample document content", "output_format": "analysis"},
            }

            # Execute the tool
            results = []
            import asyncio

            async def run_tool():
                async for result in graph_tool.stream(tool_use, {}):
                    results.append(result)

            asyncio.run(run_tool())

            # Verify execution
            assert len(results) == 1
            assert results[0]["status"] == "success"
            mock_graph.assert_called_once_with(
                "Process this document: Sample document content and generate analysis", entry_point="validator"
            )

        finally:
            # Clean up
            Path(config_file).unlink()

    def test_mixed_tool_types_loading(self):
        """Test loading multiple different tool types together."""
        # Mock loaders
        mock_swarm_loader = Mock()
        mock_swarm = Mock()
        mock_swarm_loader.load_swarm.return_value = mock_swarm

        mock_graph_loader = Mock()
        mock_graph = Mock()
        mock_graph_loader.load_graph.return_value = mock_graph

        mock_agent_tool = Mock()
        mock_agent_tool.tool_name = "agent_tool"

        # Mock the loader methods
        self.loader._get_swarm_config_loader = Mock(return_value=mock_swarm_loader)
        self.loader._get_graph_config_loader = Mock(return_value=mock_graph_loader)
        self.loader._load_agent_as_tool = Mock(return_value=mock_agent_tool)

        # Create mixed configuration
        configs = [
            {"name": "swarm_tool", "swarm": {"agents": []}},
            {"name": "graph_tool", "graph": {"nodes": [], "edges": [], "entry_points": []}},
            {"name": "agent_tool", "agent": {"model": "test-model"}},
        ]

        # Load all tools
        tools = self.loader.load_tools(configs)

        # Verify all tools were loaded
        assert len(tools) == 3
        assert tools[0].tool_type == "swarm"
        assert tools[1].tool_type == "graph"
        assert tools[2] == mock_agent_tool

    def test_convention_based_detection_in_practice(self):
        """Test that convention-based detection works correctly in practice."""
        test_cases = [
            ({"name": "test", "swarm": {}}, "swarm"),
            ({"name": "test", "graph": {}}, "graph"),
            ({"name": "test", "agent": {}}, "agent"),
            ({"name": "test", "module": "test.module"}, "legacy_tool"),
            ({"name": "test", "description": "test"}, "agent"),  # default
        ]

        for config, expected_type in test_cases:
            detected_type = self.loader._determine_config_type(config)
            assert detected_type == expected_type, f"Failed for config: {config}"

    def test_error_handling(self):
        """Test error handling for invalid configurations."""
        # Test missing required fields
        with pytest.raises(ValueError, match="must include 'name' field"):
            self.loader._load_swarm_as_tool({"swarm": {}})

        with pytest.raises(ValueError, match="must include 'graph' field"):
            self.loader._load_graph_as_tool({"name": "test"})

        # Test invalid tool specification in load_tools
        with pytest.raises(ValueError, match="Invalid tool specification"):
            self.loader.load_tools([123])  # Invalid type

    def test_multiple_loads_create_independent_objects(self):
        """Test that multiple loads create independent tool objects."""
        mock_swarm_loader = Mock()
        mock_swarm = Mock()
        mock_swarm_loader.load_swarm.return_value = mock_swarm

        # Mock the _get_swarm_config_loader method
        self.loader._get_swarm_config_loader = Mock(return_value=mock_swarm_loader)

        config = {"name": "test_swarm", "swarm": {"agents": []}}

        # Load the same tool multiple times
        tool1 = self.loader.load_tool(config)
        tool2 = self.loader.load_tool(config)
        tool3 = self.loader.load_tool(config)

        # Should create different tool wrapper objects each time
        assert tool1 is not tool2
        assert tool2 is not tool3
        assert tool1 is not tool3

        # Swarm loader should be called each time
        assert mock_swarm_loader.load_swarm.call_count == 3
