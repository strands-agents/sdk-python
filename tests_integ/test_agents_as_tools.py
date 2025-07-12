"""Integration tests for agents-as-tools functionality.

This module tests the core agents-as-tools pattern where Agent instances can be
used as tools by other agents, including registry integration and end-to-end workflows.
"""

from unittest.mock import AsyncMock

import pytest

from strands import Agent, tool
from strands.tools.agent_tool_wrapper import AgentToolWrapper
from strands.tools.registry import ToolRegistry
from strands.types.tools import ToolUse


class TestAgentsAsToolsIntegration:
    """Integration tests for agents-as-tools functionality."""

    def test_agent_tool_wrapper_basic_functionality(self):
        """Test basic agent tool wrapper functionality."""
        # Create a simple agent
        sub_agent = Agent(
            name="calculator_agent",
            description="A simple calculator agent that can perform basic arithmetic",
            load_tools_from_directory=False
        )

        # Wrap it as a tool
        wrapper = AgentToolWrapper(sub_agent)

        # Verify wrapper properties
        assert wrapper.tool_name == "calculator_agent"
        assert wrapper.tool_type == "agent"
        assert wrapper.tool_spec["name"] == "calculator_agent"
        assert wrapper.tool_spec["description"] == "A simple calculator agent that can perform basic arithmetic"
        assert "query" in wrapper.tool_spec["inputSchema"]["properties"]

    @pytest.mark.asyncio
    async def test_agent_tool_wrapper_execution(self):
        """Test agent tool wrapper execution flow."""
        # Create a mock sub-agent
        sub_agent = Agent(
            name="text_processor",
            description="An agent that processes text input",
            load_tools_from_directory=False
        )

        # Mock the invoke_async method
        sub_agent.invoke_async = AsyncMock(return_value="Processed: Hello World")

        # Wrap it as a tool
        wrapper = AgentToolWrapper(sub_agent)

        # Create a tool use request
        tool_use: ToolUse = {
            "toolUseId": "test_123",
            "name": "text_processor",
            "input": {"query": "Hello World"}
        }

        # Execute the tool
        results = []
        async for result in wrapper.stream(tool_use, {}):
            results.append(result)

        # Verify execution
        assert len(results) == 1
        result = results[0]
        assert result["toolUseId"] == "test_123"
        assert result["status"] == "success"
        assert result["content"][0]["text"] == "Processed: Hello World"

        # Verify sub-agent was called with correct query
        sub_agent.invoke_async.assert_called_once_with("Hello World")

    def test_tool_registry_agent_detection(self):
        """Test that tool registry can detect and register agents as tools."""
        # Create a tool registry
        registry = ToolRegistry()

        # Create an agent
        agent = Agent(
            name="data_analyzer",
            description="An agent that analyzes data patterns",
            load_tools_from_directory=False
        )

        # Process the agent through the registry
        tool_names = registry.process_tools([agent])

        # Verify the agent was registered as a tool
        assert len(tool_names) == 1
        assert tool_names[0] == "data_analyzer"
        assert "data_analyzer" in registry.registry

        # Verify the registered tool is an AgentToolWrapper
        registered_tool = registry.registry["data_analyzer"]
        assert isinstance(registered_tool, AgentToolWrapper)
        assert registered_tool.tool_name == "data_analyzer"
        assert registered_tool.tool_type == "agent"

    @pytest.mark.asyncio
    async def test_agent_using_agent_as_tool(self):
        """Test end-to-end scenario where one agent uses another agent as a tool."""
        # Create a function tool for the sub-agent
        @tool
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        # Create a sub-agent with the function tool
        sub_agent = Agent(
            name="math_helper",
            description="A helper agent that performs mathematical operations",
            tools=[add_numbers],
            load_tools_from_directory=False
        )

        # Create a main agent that will use the sub-agent as a tool
        main_agent = Agent(
            name="main_agent",
            description="A main agent that delegates tasks to other agents",
            tools=[sub_agent],  # Pass the agent as a tool
            load_tools_from_directory=False
        )

        # Verify the sub-agent was registered as a tool
        assert hasattr(main_agent.tool, "math_helper")

        # Get the tool spec for the sub-agent
        tool_specs = main_agent.tool_registry.get_all_tool_specs()
        math_helper_spec = next((spec for spec in tool_specs if spec["name"] == "math_helper"), None)

        assert math_helper_spec is not None
        assert math_helper_spec["name"] == "math_helper"
        assert math_helper_spec["description"] == "A helper agent that performs mathematical operations"
        assert "query" in math_helper_spec["inputSchema"]["json"]["properties"]

    @pytest.mark.asyncio
    async def test_agent_tool_error_handling(self):
        """Test error handling in agent tools."""
        # Create an agent that will raise an exception
        failing_agent = Agent(
            name="failing_agent",
            description="An agent that fails during execution",
            load_tools_from_directory=False
        )

        # Mock the invoke_async to raise an exception
        failing_agent.invoke_async = AsyncMock(side_effect=Exception("Simulated failure"))

        # Wrap it as a tool
        wrapper = AgentToolWrapper(failing_agent)

        # Create a tool use request
        tool_use: ToolUse = {
            "toolUseId": "error_test_123",
            "name": "failing_agent",
            "input": {"query": "This will fail"}
        }

        # Execute the tool
        results = []
        async for result in wrapper.stream(tool_use, {}):
            results.append(result)

        # Verify error handling
        assert len(results) == 1
        result = results[0]
        assert result["toolUseId"] == "error_test_123"
        assert result["status"] == "error"
        assert "Error executing 'failing_agent'" in result["content"][0]["text"]
        assert "Simulated failure" in result["content"][0]["text"]

    def test_agent_tool_validation(self):
        """Test agent tool validation requirements."""
        # Test with valid agent
        valid_agent = Agent(
            name="valid_agent",
            description="A properly configured agent",
            load_tools_from_directory=False
        )

        # Should not raise an exception
        wrapper = AgentToolWrapper(valid_agent)
        assert wrapper.tool_name == "valid_agent"

        # Test with agent having default name (should fail)
        invalid_agent = Agent(
            description="An agent with default name",
            load_tools_from_directory=False
        )
        # The default name is "Strands Agents"

        with pytest.raises(ValueError, match="Agent must have both 'name' and 'description' parameters"):
            AgentToolWrapper(invalid_agent)

    def test_multiple_agents_as_tools(self):
        """Test using multiple agents as tools in a single main agent."""
        # Create multiple sub-agents
        agent1 = Agent(
            name="summarizer",
            description="An agent that summarizes text",
            load_tools_from_directory=False
        )

        agent2 = Agent(
            name="translator",
            description="An agent that translates text",
            load_tools_from_directory=False
        )

        agent3 = Agent(
            name="analyzer",
            description="An agent that analyzes sentiment",
            load_tools_from_directory=False
        )

        # Create a main agent with multiple sub-agents as tools
        main_agent = Agent(
            name="orchestrator",
            description="An orchestrator agent that coordinates multiple sub-agents",
            tools=[agent1, agent2, agent3],
            load_tools_from_directory=False
        )

        # Verify all sub-agents were registered as tools
        tool_specs = main_agent.tool_registry.get_all_tool_specs()
        tool_names = [spec["name"] for spec in tool_specs]

        assert "summarizer" in tool_names
        assert "translator" in tool_names
        assert "analyzer" in tool_names

        # Verify tools are accessible
        assert hasattr(main_agent.tool, "summarizer")
        assert hasattr(main_agent.tool, "translator")
        assert hasattr(main_agent.tool, "analyzer")

    def test_mixed_tools_and_agents(self):
        """Test mixing regular function tools with agent tools."""
        # Create a function tool
        @tool
        def format_text(text: str) -> str:
            """Format text to uppercase."""
            return text.upper()

        # Create an agent tool
        agent_tool = Agent(
            name="content_creator",
            description="An agent that creates content",
            load_tools_from_directory=False
        )

        # Create a main agent with both function and agent tools
        main_agent = Agent(
            name="content_manager",
            description="A manager that handles content processing",
            tools=[format_text, agent_tool],
            load_tools_from_directory=False
        )

        # Verify both tools are available
        tool_specs = main_agent.tool_registry.get_all_tool_specs()
        tool_names = [spec["name"] for spec in tool_specs]

        assert "format_text" in tool_names
        assert "content_creator" in tool_names

        # Verify tools are accessible
        assert hasattr(main_agent.tool, "format_text")
        assert hasattr(main_agent.tool, "content_creator")

        # Verify tool types are correct
        function_tool = main_agent.tool_registry.registry["format_text"]
        agent_tool_wrapper = main_agent.tool_registry.registry["content_creator"]

        assert function_tool.tool_type == "function"
        assert agent_tool_wrapper.tool_type == "agent"

    @pytest.mark.asyncio
    async def test_agent_tool_context_isolation(self):
        """Test that agent tools maintain proper context isolation."""
        # Create two identical agents
        agent1 = Agent(
            name="counter1",
            description="A counter agent instance 1",
            load_tools_from_directory=False
        )

        agent2 = Agent(
            name="counter2",
            description="A counter agent instance 2",
            load_tools_from_directory=False
        )

        # Mock different behaviors for each agent
        agent1.invoke_async = AsyncMock(return_value="Agent1 response")
        agent2.invoke_async = AsyncMock(return_value="Agent2 response")

        # Wrap both as tools
        wrapper1 = AgentToolWrapper(agent1)
        wrapper2 = AgentToolWrapper(agent2)

        # Execute both tools
        tool_use1: ToolUse = {
            "toolUseId": "test1",
            "name": "counter1",
            "input": {"query": "Test query"}
        }

        tool_use2: ToolUse = {
            "toolUseId": "test2",
            "name": "counter2",
            "input": {"query": "Test query"}
        }

        # Execute first tool
        results1 = []
        async for result in wrapper1.stream(tool_use1, {}):
            results1.append(result)

        # Execute second tool
        results2 = []
        async for result in wrapper2.stream(tool_use2, {}):
            results2.append(result)

        # Verify each agent was called independently
        assert len(results1) == 1
        assert len(results2) == 1
        assert results1[0]["content"][0]["text"] == "Agent1 response"
        assert results2[0]["content"][0]["text"] == "Agent2 response"

        # Verify each agent was called once
        agent1.invoke_async.assert_called_once()
        agent2.invoke_async.assert_called_once()
