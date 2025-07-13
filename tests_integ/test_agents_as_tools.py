"""Integration tests for agents-as-tools functionality.

This module tests the core agents-as-tools pattern where Agent instances can be
used as tools by other agents, including registry integration and end-to-end workflows.
"""

import pytest

from strands import Agent, tool
from strands.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_agent_using_agent_as_tool_real_invocation():
    """Test end-to-end scenario where one agent uses another agent as a tool with real invocation."""

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
        load_tools_from_directory=False,
    )

    # Create a main agent that will use the sub-agent as a tool
    main_agent = Agent(
        name="main_agent",
        description="A main agent that delegates tasks to other agents",
        tools=[sub_agent],  # Pass the agent as a tool
        load_tools_from_directory=False,
    )

    # Verify the sub-agent was registered as a tool
    assert hasattr(main_agent.tool, "math_helper")

    # Get the tool spec for the sub-agent
    tool_specs = main_agent.tool_registry.get_all_tool_specs()
    math_helper_spec = next((spec for spec in tool_specs if spec["name"] == "math_helper"), None)

    assert math_helper_spec is not None
    assert math_helper_spec["name"] == "math_helper"
    assert math_helper_spec["description"] == "A helper agent that performs mathematical operations"
    assert "prompt" in math_helper_spec["inputSchema"]["json"]["properties"]


@pytest.mark.asyncio
async def test_parent_agent_invokes_sub_agent():
    """Test that a parent agent can invoke a sub-agent and the sub-agent's messages are populated."""

    # Create a sub-agent that will be used as a tool
    tool_agent = Agent(
        name="calculator",
        description="A calculator agent that can perform basic arithmetic",
        load_tools_from_directory=False,
    )

    # Create a parent agent that uses the sub-agent as a tool
    parent_agent = Agent(
        name="orchestrator",
        description="An orchestrator that delegates mathematical tasks",
        tools=[tool_agent],
        load_tools_from_directory=False,
    )

    # Clear any existing messages
    tool_agent.messages = []
    parent_agent.messages = []

    # Invoke the parent agent with a task that should trigger the sub-agent
    response = await parent_agent.invoke_async(
        "Please use the calculator to help me with a simple math problem: what is 2 + 2?"
    )

    # Assert that the sub-agent was called (has messages)
    assert len(tool_agent.messages) > 0, "Sub-agent should have been invoked and have messages"

    # Verify the parent agent also has messages
    assert len(parent_agent.messages) > 0, "Parent agent should have messages from the conversation"

    # Verify the response is not empty
    assert response is not None and len(str(response).strip()) > 0, "Response should not be empty"


def test_tool_registry_agent_detection():
    """Test that tool registry can detect and register agents as tools."""
    # Create a tool registry
    registry = ToolRegistry()

    # Create an agent
    agent = Agent(
        name="data_analyzer", description="An agent that analyzes data patterns", load_tools_from_directory=False
    )

    # Process the agent through the registry
    tool_names = registry.process_tools([agent])

    # Verify the agent was registered as a tool
    assert len(tool_names) == 1
    assert tool_names[0] == "data_analyzer"
    assert "data_analyzer" in registry.registry

    # Verify the registered tool is an AgentToolWrapper
    from strands.tools.agent_tool_wrapper import AgentToolWrapper

    registered_tool = registry.registry["data_analyzer"]
    assert isinstance(registered_tool, AgentToolWrapper)
    assert registered_tool.tool_name == "data_analyzer"
    assert registered_tool.tool_type == "agent"


def test_multiple_agents_as_tools():
    """Test using multiple agents as tools in a single main agent."""
    # Create multiple sub-agents
    agent1 = Agent(name="summarizer", description="An agent that summarizes text", load_tools_from_directory=False)

    agent2 = Agent(name="translator", description="An agent that translates text", load_tools_from_directory=False)

    agent3 = Agent(name="analyzer", description="An agent that analyzes sentiment", load_tools_from_directory=False)

    # Create a main agent with multiple sub-agents as tools
    main_agent = Agent(
        name="orchestrator",
        description="An orchestrator agent that coordinates multiple sub-agents",
        tools=[agent1, agent2, agent3],
        load_tools_from_directory=False,
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


def test_mixed_tools_and_agents():
    """Test mixing regular function tools with agent tools."""

    # Create a function tool
    @tool
    def format_text(text: str) -> str:
        """Format text to uppercase."""
        return text.upper()

    # Create an agent tool
    agent_tool = Agent(
        name="content_creator", description="An agent that creates content", load_tools_from_directory=False
    )

    # Create a main agent with both function and agent tools
    main_agent = Agent(
        name="content_manager",
        description="A manager that handles content processing",
        tools=[format_text, agent_tool],
        load_tools_from_directory=False,
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
async def test_nested_agent_tool_invocation():
    """Test that nested agent-as-tool invocations work correctly."""

    # Create a bottom-level agent with a simple tool
    @tool
    def get_current_time() -> str:
        """Get the current time."""
        return "12:00 PM"

    time_agent = Agent(
        name="time_keeper",
        description="An agent that provides time information",
        tools=[get_current_time],
        load_tools_from_directory=False,
    )

    # Create a middle-level agent that uses the time agent
    middle_agent = Agent(
        name="scheduler",
        description="An agent that helps with scheduling using time information",
        tools=[time_agent],
        load_tools_from_directory=False,
    )

    # Create a top-level agent that uses the middle agent
    top_agent = Agent(
        name="assistant",
        description="A top-level assistant that handles various tasks",
        tools=[middle_agent],
        load_tools_from_directory=False,
    )

    # Clear messages
    time_agent.messages = []
    middle_agent.messages = []
    top_agent.messages = []

    # Invoke the top agent with a task that should cascade through the agents
    response = await top_agent.invoke_async("Please help me schedule something by checking the current time")

    # Verify all agents were involved
    assert len(time_agent.messages) > 0, "Bottom-level agent should have been invoked"
    assert len(middle_agent.messages) > 0, "Middle-level agent should have been invoked"
    assert len(top_agent.messages) > 0, "Top-level agent should have been invoked"

    # Verify response is not empty
    assert response is not None and len(str(response).strip()) > 0


@pytest.mark.asyncio
async def test_agent_tool_with_complex_workflow():
    """Test a more complex workflow with multiple agent tools working together."""

    # Create specialized agents
    @tool
    def analyze_sentiment(text: str) -> str:
        """Analyze the sentiment of text."""
        if "happy" in text.lower() or "good" in text.lower():
            return "positive"
        elif "sad" in text.lower() or "bad" in text.lower():
            return "negative"
        else:
            return "neutral"

    sentiment_agent = Agent(
        name="sentiment_analyzer",
        description="An agent that analyzes sentiment in text",
        tools=[analyze_sentiment],
        load_tools_from_directory=False,
    )

    @tool
    def count_words(text: str) -> int:
        """Count the number of words in text."""
        return len(text.split())

    word_counter = Agent(
        name="word_counter",
        description="An agent that counts words in text",
        tools=[count_words],
        load_tools_from_directory=False,
    )

    # Create a coordinator agent that uses both specialized agents
    coordinator = Agent(
        name="text_analyzer",
        description="A coordinator that performs comprehensive text analysis",
        tools=[sentiment_agent, word_counter],
        load_tools_from_directory=False,
    )

    # Clear messages
    sentiment_agent.messages = []
    word_counter.messages = []
    coordinator.messages = []

    # Test the workflow
    test_text = "This is a happy day with good weather"
    response = await coordinator.invoke_async(
        f"Please analyze this text: '{test_text}'. I need both sentiment analysis and word count."
    )

    # Verify all agents were involved
    assert len(sentiment_agent.messages) > 0, "Sentiment agent should have been invoked"
    assert len(word_counter.messages) > 0, "Word counter agent should have been invoked"
    assert len(coordinator.messages) > 0, "Coordinator agent should have been invoked"

    # Verify response contains analysis
    assert response is not None and len(str(response).strip()) > 0
