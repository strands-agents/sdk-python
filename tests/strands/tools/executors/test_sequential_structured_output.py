"""Tests for sequential executor with structured output support."""

import pytest
from pydantic import BaseModel

import strands
from strands.tools.executors import SequentialToolExecutor
from strands.tools.structured_output.structured_output_context import StructuredOutputContext
from strands.tools.structured_output.structured_output_tool import StructuredOutputTool
from strands.types.tools import ToolUse


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""

    name: str
    age: int


class AnotherModel(BaseModel):
    """Another Pydantic model for testing."""

    city: str
    population: int


@pytest.fixture
def executor():
    """Create a sequential executor instance."""
    return SequentialToolExecutor()


@pytest.fixture
def structured_output_context():
    """Create a structured output context with SampleModel."""
    return StructuredOutputContext(structured_output_model=SampleModel)


@pytest.fixture
def capture_tool():
    """Create a tool that captures kwargs passed to it."""
    captured_kwargs = {}

    @strands.tool(name="capture_tool")
    def func():
        return "captured"

    # Override the stream method to capture kwargs
    original_stream = func.stream

    async def capturing_stream(tool_use, invocation_state, **kwargs):
        captured_kwargs.update(kwargs)
        async for event in original_stream(tool_use, invocation_state, **kwargs):
            yield event

    func.stream = capturing_stream
    func.captured_kwargs = captured_kwargs
    return func


@pytest.mark.asyncio
async def test_sequential_executor_passes_structured_output_context(
    executor,
    agent,
    tool_results,
    cycle_trace,
    cycle_span,
    invocation_state,
    structured_output_context,
    capture_tool,
    alist,
):
    """Test that sequential executor properly passes structured output context to tools."""
    # Register the capture tool
    agent.tool_registry.register_tool(capture_tool)

    # Set up tool uses
    tool_uses: list[ToolUse] = [
        {"name": "capture_tool", "toolUseId": "1", "input": {}},
    ]

    # Execute tools with structured output context
    stream = executor._execute(
        agent, tool_uses, tool_results, cycle_trace, cycle_span, invocation_state, structured_output_context
    )

    # Collect events
    events = await alist(stream)

    # Verify the structured_output_context was passed to the tool
    assert "structured_output_context" in capture_tool.captured_kwargs
    assert capture_tool.captured_kwargs["structured_output_context"] is structured_output_context

    # Verify event was generated
    assert len(events) == 1
    assert events[0].tool_use_id == "1"


@pytest.mark.asyncio
async def test_sequential_execution_order(
    executor, agent, tool_results, cycle_trace, cycle_span, invocation_state, structured_output_context, alist
):
    """Test that tools are executed in sequential order."""
    execution_order = []

    # Create tools that track execution order
    @strands.tool(name="first_tool")
    def first_func():
        execution_order.append("first")
        return "first done"

    @strands.tool(name="second_tool")
    def second_func():
        execution_order.append("second")
        return "second done"

    @strands.tool(name="third_tool")
    def third_func():
        execution_order.append("third")
        return "third done"

    # Register tools
    agent.tool_registry.register_tool(first_func)
    agent.tool_registry.register_tool(second_func)
    agent.tool_registry.register_tool(third_func)

    # Set up tool uses
    tool_uses: list[ToolUse] = [
        {"name": "first_tool", "toolUseId": "1", "input": {}},
        {"name": "second_tool", "toolUseId": "2", "input": {}},
        {"name": "third_tool", "toolUseId": "3", "input": {}},
    ]

    # Execute tools
    stream = executor._execute(
        agent, tool_uses, tool_results, cycle_trace, cycle_span, invocation_state, structured_output_context
    )

    # Collect events
    events = await alist(stream)

    # Verify sequential execution order
    assert execution_order == ["first", "second", "third"]

    # Verify events are in order
    assert len(events) == 3
    assert events[0].tool_use_id == "1"
    assert events[1].tool_use_id == "2"
    assert events[2].tool_use_id == "3"


@pytest.mark.asyncio
async def test_sequential_executor_with_structured_output_tool(
    executor, agent, tool_results, cycle_trace, cycle_span, invocation_state, structured_output_context, alist
):
    """Test sequential execution with structured output tool."""
    # Create and register structured output tool
    structured_tool = StructuredOutputTool(SampleModel)
    agent.tool_registry.register_tool(structured_tool)

    # Create and register a regular tool
    @strands.tool(name="regular_tool")
    def regular_func():
        return "regular result"

    agent.tool_registry.register_tool(regular_func)

    # Set up tool uses - structured tool followed by regular tool
    tool_uses: list[ToolUse] = [
        {"name": SampleModel.__name__, "toolUseId": "1", "input": {"name": "Alice", "age": 30}},
        {"name": "regular_tool", "toolUseId": "2", "input": {}},
    ]

    # Execute tools
    stream = executor._execute(
        agent, tool_uses, tool_results, cycle_trace, cycle_span, invocation_state, structured_output_context
    )

    # Collect events
    events = await alist(stream)

    # Verify both tools were executed in order
    assert len(events) == 2
    assert events[0].tool_use_id == "1"
    assert events[0].tool_result["status"] == "success"
    assert events[1].tool_use_id == "2"
    assert events[1].tool_result["status"] == "success"

    # Verify structured output was stored
    result = structured_output_context.get_result("1")
    assert result is not None
    assert result.name == "Alice"
    assert result.age == 30


@pytest.mark.asyncio
async def test_sequential_executor_stops_on_error(
    executor, agent, tool_results, cycle_trace, cycle_span, invocation_state, structured_output_context, alist
):
    """Test that sequential executor continues even when a tool fails."""
    execution_tracker = []

    # Create tools with one that fails
    @strands.tool(name="success_tool_1")
    def success_func_1():
        execution_tracker.append("success1")
        return "success1"

    @strands.tool(name="failing_tool")
    def failing_func():
        execution_tracker.append("failing")
        raise ValueError("Tool failed!")

    @strands.tool(name="success_tool_2")
    def success_func_2():
        execution_tracker.append("success2")
        return "success2"

    # Register tools
    agent.tool_registry.register_tool(success_func_1)
    agent.tool_registry.register_tool(failing_func)
    agent.tool_registry.register_tool(success_func_2)

    # Set up tool uses
    tool_uses: list[ToolUse] = [
        {"name": "success_tool_1", "toolUseId": "1", "input": {}},
        {"name": "failing_tool", "toolUseId": "2", "input": {}},
        {"name": "success_tool_2", "toolUseId": "3", "input": {}},
    ]

    # Execute tools
    stream = executor._execute(
        agent, tool_uses, tool_results, cycle_trace, cycle_span, invocation_state, structured_output_context
    )

    # Collect events
    events = await alist(stream)

    # Verify all tools were executed despite the failure
    assert len(execution_tracker) == 3
    assert execution_tracker == ["success1", "failing", "success2"]

    # Verify all events were generated
    assert len(events) == 3
    assert events[0].tool_result["status"] == "success"
    assert events[1].tool_result["status"] == "error"
    assert "Tool failed!" in events[1].tool_result["content"][0]["text"]
    assert events[2].tool_result["status"] == "success"


@pytest.mark.asyncio
async def test_sequential_executor_with_multiple_structured_tools(
    executor, agent, tool_results, cycle_trace, cycle_span, invocation_state, structured_output_context, alist
):
    """Test sequential execution with multiple different structured output tools."""
    # Create and register structured output tools for different models
    sample_tool = StructuredOutputTool(SampleModel)
    another_tool = StructuredOutputTool(AnotherModel)

    agent.tool_registry.register_tool(sample_tool)
    agent.tool_registry.register_tool(another_tool)

    # Set up tool uses for both structured tools
    tool_uses: list[ToolUse] = [
        {"name": SampleModel.__name__, "toolUseId": "1", "input": {"name": "Bob", "age": 25}},
        {"name": AnotherModel.__name__, "toolUseId": "2", "input": {"city": "Seattle", "population": 750000}},
        {"name": SampleModel.__name__, "toolUseId": "3", "input": {"name": "Charlie", "age": 35}},
    ]

    # Execute tools with the structured output context
    stream = executor._execute(
        agent, tool_uses, tool_results, cycle_trace, cycle_span, invocation_state, structured_output_context
    )

    # Collect events
    events = await alist(stream)

    # Verify all tools were executed in order
    assert len(events) == 3

    # Verify each event
    assert events[0].tool_use_id == "1"
    assert events[0].tool_result["status"] == "success"

    assert events[1].tool_use_id == "2"
    assert events[1].tool_result["status"] == "success"

    assert events[2].tool_use_id == "3"
    assert events[2].tool_result["status"] == "success"

    # Verify all results were stored correctly
    result1 = structured_output_context.get_result("1")
    assert isinstance(result1, SampleModel)
    assert result1.name == "Bob"
    assert result1.age == 25

    result2 = structured_output_context.get_result("2")
    assert isinstance(result2, AnotherModel)
    assert result2.city == "Seattle"
    assert result2.population == 750000

    result3 = structured_output_context.get_result("3")
    assert isinstance(result3, SampleModel)
    assert result3.name == "Charlie"
    assert result3.age == 35
