"""Tests for AgentToolWrapper."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from strands.agent.agent import Agent
from strands.tools.agent_tool_wrapper import AgentToolWrapper
from strands.types.tools import ToolSpec, ToolUse


class TestAgentToolWrapper:
    """Test suite for AgentToolWrapper class."""

    def test_init_with_valid_agent(self):
        """Test initialization with a valid agent."""
        # Create a mock agent with required attributes
        agent = MagicMock(spec=Agent)
        agent.name = "test_agent"
        agent.description = "A test agent for testing"

        wrapper = AgentToolWrapper(agent)

        assert wrapper._agent == agent
        assert wrapper._name == "test_agent"
        assert wrapper._description == "A test agent for testing"

    def test_init_with_agent_missing_name(self):
        """Test initialization fails when agent has no name."""
        agent = MagicMock(spec=Agent)
        agent.name = None
        agent.description = "A test agent"

        with pytest.raises(ValueError, match="Agent must have both 'name' and 'description' parameters"):
            AgentToolWrapper(agent)

    def test_init_with_agent_missing_description(self):
        """Test initialization fails when agent has no description."""
        agent = MagicMock(spec=Agent)
        agent.name = "test_agent"
        agent.description = None

        with pytest.raises(ValueError, match="Agent must have both 'name' and 'description' parameters"):
            AgentToolWrapper(agent)

    def test_init_with_agent_default_name(self):
        """Test initialization fails when agent has default name."""
        agent = MagicMock(spec=Agent)
        agent.name = "Strands Agents"
        agent.description = "A test agent"

        with pytest.raises(ValueError, match="Agent must have both 'name' and 'description' parameters"):
            AgentToolWrapper(agent)

    def test_init_with_agent_empty_description(self):
        """Test initialization fails when agent has empty description."""
        agent = MagicMock(spec=Agent)
        agent.name = "test_agent"
        agent.description = ""

        with pytest.raises(ValueError, match="Agent must have both 'name' and 'description' parameters"):
            AgentToolWrapper(agent)

    def test_tool_name_property(self):
        """Test tool_name property returns agent name."""
        agent = MagicMock(spec=Agent)
        agent.name = "my_agent"
        agent.description = "My agent description"

        wrapper = AgentToolWrapper(agent)

        assert wrapper.tool_name == "my_agent"

    def test_tool_spec_property(self):
        """Test tool_spec property returns correct specification."""
        agent = MagicMock(spec=Agent)
        agent.name = "my_agent"
        agent.description = "My agent description"

        wrapper = AgentToolWrapper(agent)

        expected_spec = ToolSpec(
            name="my_agent",
            description="My agent description",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query or task to send to the sub-agent"}
                },
                "required": ["query"],
            },
        )

        assert wrapper.tool_spec == expected_spec

    def test_tool_type_property(self):
        """Test tool_type property returns 'agent'."""
        agent = MagicMock(spec=Agent)
        agent.name = "my_agent"
        agent.description = "My agent description"

        wrapper = AgentToolWrapper(agent)

        assert wrapper.tool_type == "agent"

    def test_get_display_properties(self):
        """Test get_display_properties returns correct properties."""
        agent = MagicMock(spec=Agent)
        agent.name = "my_agent"
        agent.description = "My agent description"

        wrapper = AgentToolWrapper(agent)

        expected_properties = {
            "Name": "my_agent",
            "Type": "agent",
            "Description": "My agent description",
        }

        assert wrapper.get_display_properties() == expected_properties

    @pytest.mark.asyncio
    async def test_stream_success(self):
        """Test successful stream execution."""
        # Create a mock agent
        agent = MagicMock(spec=Agent)
        agent.name = "test_agent"
        agent.description = "A test agent"

        # Mock the invoke_async method to return a result
        mock_result = "Agent response to the query"
        agent.invoke_async = AsyncMock(return_value=mock_result)

        wrapper = AgentToolWrapper(agent)

        # Create a tool use request
        tool_use: ToolUse = {
            "toolUseId": "test_tool_use_123",
            "name": "test_agent",
            "input": {"query": "What is the weather like?"},
        }

        # Execute the stream
        events = []
        async for event in wrapper.stream(tool_use, {}):
            events.append(event)

        # Verify the agent was called with the query
        agent.invoke_async.assert_called_once_with("What is the weather like?")

        # Verify the result
        assert len(events) == 1
        result = events[0]

        assert isinstance(result, dict)
        assert result["toolUseId"] == "test_tool_use_123"
        assert result["status"] == "success"
        assert len(result["content"]) == 1
        assert result["content"][0]["text"] == "Agent response to the query"

    @pytest.mark.asyncio
    async def test_stream_with_empty_query(self):
        """Test stream execution with empty query."""
        agent = MagicMock(spec=Agent)
        agent.name = "test_agent"
        agent.description = "A test agent"

        mock_result = "Default response"
        agent.invoke_async = AsyncMock(return_value=mock_result)

        wrapper = AgentToolWrapper(agent)

        # Create a tool use request with empty query
        tool_use: ToolUse = {"toolUseId": "test_tool_use_123", "name": "test_agent", "input": {}}

        # Execute the stream
        events = []
        async for event in wrapper.stream(tool_use, {}):
            events.append(event)

        # Verify the agent was called with empty string
        agent.invoke_async.assert_called_once_with("")

        # Verify the result
        assert len(events) == 1
        result = events[0]
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_stream_with_agent_exception(self):
        """Test stream execution when agent raises an exception."""
        agent = MagicMock(spec=Agent)
        agent.name = "test_agent"
        agent.description = "A test agent"

        # Mock the invoke_async method to raise an exception
        agent.invoke_async = AsyncMock(side_effect=Exception("Agent error"))

        wrapper = AgentToolWrapper(agent)

        # Create a tool use request
        tool_use: ToolUse = {
            "toolUseId": "test_tool_use_123",
            "name": "test_agent",
            "input": {"query": "What is the weather like?"},
        }

        # Execute the stream
        events = []
        async for event in wrapper.stream(tool_use, {}):
            events.append(event)

        # Verify the agent was called
        agent.invoke_async.assert_called_once_with("What is the weather like?")

        # Verify the error result
        assert len(events) == 1
        result = events[0]

        assert result["toolUseId"] == "test_tool_use_123"
        assert result["status"] == "error"
        assert len(result["content"]) == 1
        assert "Error executing sub-agent 'test_agent': Agent error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_stream_with_complex_agent_result(self):
        """Test stream execution with complex agent result."""
        agent = MagicMock(spec=Agent)
        agent.name = "test_agent"
        agent.description = "A test agent"

        # Mock agent result with a complex object
        class ComplexResult:
            def __init__(self, value):
                self.value = value

            def __str__(self):
                return f"ComplexResult(value={self.value})"

        mock_result = ComplexResult("complex_data")
        agent.invoke_async = AsyncMock(return_value=mock_result)

        wrapper = AgentToolWrapper(agent)

        # Create a tool use request
        tool_use: ToolUse = {
            "toolUseId": "test_tool_use_123",
            "name": "test_agent",
            "input": {"query": "Process complex data"},
        }

        # Execute the stream
        events = []
        async for event in wrapper.stream(tool_use, {}):
            events.append(event)

        # Verify the result is converted to string
        assert len(events) == 1
        result = events[0]
        assert result["status"] == "success"
        assert result["content"][0]["text"] == "ComplexResult(value=complex_data)"

    @pytest.mark.asyncio
    async def test_stream_with_additional_kwargs(self):
        """Test stream execution with additional kwargs."""
        agent = MagicMock(spec=Agent)
        agent.name = "test_agent"
        agent.description = "A test agent"

        mock_result = "Agent response"
        agent.invoke_async = AsyncMock(return_value=mock_result)

        wrapper = AgentToolWrapper(agent)

        # Create a tool use request
        tool_use: ToolUse = {"toolUseId": "test_tool_use_123", "name": "test_agent", "input": {"query": "Test query"}}

        # Execute the stream with additional kwargs
        events = []
        async for event in wrapper.stream(tool_use, {}, extra_param="value"):
            events.append(event)

        # Verify the agent was called (kwargs should not affect the call)
        agent.invoke_async.assert_called_once_with("Test query")

        # Verify the result
        assert len(events) == 1
        result = events[0]
        assert result["status"] == "success"

    def test_inheritance_from_agent_tool(self):
        """Test that AgentToolWrapper properly inherits from AgentTool."""
        agent = MagicMock(spec=Agent)
        agent.name = "test_agent"
        agent.description = "A test agent"

        wrapper = AgentToolWrapper(agent)

        # Test that it has the required abstract methods
        assert hasattr(wrapper, "tool_name")
        assert hasattr(wrapper, "tool_spec")
        assert hasattr(wrapper, "tool_type")
        assert hasattr(wrapper, "stream")

        # Test that it inherits from AgentTool
        from strands.types.tools import AgentTool

        assert isinstance(wrapper, AgentTool)

    def test_validation_error_message_format(self):
        """Test that validation error message includes proper formatting."""
        agent = MagicMock(spec=Agent)
        agent.name = None
        agent.description = None

        with pytest.raises(ValueError) as exc_info:
            AgentToolWrapper(agent)

        error_message = str(exc_info.value)
        assert "Agent must have both 'name' and 'description' parameters" in error_message
        assert "'name' must not be default agent name: 'Strands Agents'" in error_message
        assert "Agent(name='tool_name', description='tool_description', ...)" in error_message

    @pytest.mark.asyncio
    async def test_stream_preserves_tool_use_id(self):
        """Test that stream preserves the original tool use ID."""
        agent = MagicMock(spec=Agent)
        agent.name = "test_agent"
        agent.description = "A test agent"

        mock_result = "Test response"
        agent.invoke_async = AsyncMock(return_value=mock_result)

        wrapper = AgentToolWrapper(agent)

        # Test with different tool use IDs
        tool_use_ids = ["id_1", "id_2", "special_id_123"]

        for tool_use_id in tool_use_ids:
            tool_use: ToolUse = {"toolUseId": tool_use_id, "name": "test_agent", "input": {"query": "Test"}}

            events = []
            async for event in wrapper.stream(tool_use, {}):
                events.append(event)

            assert len(events) == 1
            assert events[0]["toolUseId"] == tool_use_id

    @pytest.mark.asyncio
    async def test_end_to_end_agent_wrapper_integration(self):
        """Test end-to-end integration with a real agent-like object."""

        # Create a more realistic agent-like object
        class TestAgent:
            def __init__(self):
                self.name = "calculator_agent"
                self.description = "An agent that performs basic calculations"
                self.call_count = 0

            async def invoke_async(self, query: str) -> str:
                self.call_count += 1
                if "add" in query.lower():
                    return f"Added numbers: {query}"
                elif "multiply" in query.lower():
                    return f"Multiplied numbers: {query}"
                else:
                    return f"Processed: {query}"

        test_agent = TestAgent()
        wrapper = AgentToolWrapper(test_agent)

        # Test the tool spec
        tool_spec = wrapper.tool_spec
        assert tool_spec["name"] == "calculator_agent"
        assert tool_spec["description"] == "An agent that performs basic calculations"
        assert "query" in tool_spec["inputSchema"]["properties"]

        # Test multiple tool executions
        test_queries = ["add 5 and 3", "multiply 4 by 7", "what is the square root of 16"]

        for i, query in enumerate(test_queries):
            tool_use: ToolUse = {"toolUseId": f"calc_tool_{i}", "name": "calculator_agent", "input": {"query": query}}

            events = []
            async for event in wrapper.stream(tool_use, {}):
                events.append(event)

            assert len(events) == 1
            result = events[0]
            assert result["toolUseId"] == f"calc_tool_{i}"
            assert result["status"] == "success"
            assert query in result["content"][0]["text"]

        # Verify the agent was called for each query
        assert test_agent.call_count == len(test_queries)

    def test_agent_tool_wrapper_with_real_agent_validation(self):
        """Test validation works with various agent configurations."""

        # Test with valid agent
        class ValidAgent:
            def __init__(self):
                self.name = "valid_agent"
                self.description = "A valid agent"
                self.invoke_async = lambda x: x

        valid_agent = ValidAgent()
        wrapper = AgentToolWrapper(valid_agent)
        assert wrapper.tool_name == "valid_agent"

        # Test with agent having empty name
        class EmptyNameAgent:
            def __init__(self):
                self.name = ""
                self.description = "An agent with empty name"
                self.invoke_async = lambda x: x

        empty_name_agent = EmptyNameAgent()
        with pytest.raises(ValueError, match="Agent must have both 'name' and 'description' parameters"):
            AgentToolWrapper(empty_name_agent)

        # Test with agent having None description
        class NoneDescAgent:
            def __init__(self):
                self.name = "test_agent"
                self.description = None
                self.invoke_async = lambda x: x

        none_desc_agent = NoneDescAgent()
        with pytest.raises(ValueError, match="Agent must have both 'name' and 'description' parameters"):
            AgentToolWrapper(none_desc_agent)

        # Test with default agent name
        class DefaultNameAgent:
            def __init__(self):
                self.name = "Strands Agents"  # Default name
                self.description = "An agent with default name"
                self.invoke_async = lambda x: x

        default_name_agent = DefaultNameAgent()
        with pytest.raises(ValueError, match="Agent must have both 'name' and 'description' parameters"):
            AgentToolWrapper(default_name_agent)
