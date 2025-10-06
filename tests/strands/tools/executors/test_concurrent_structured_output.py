"""Tests for concurrent executor with structured output support."""

import pytest
from pydantic import BaseModel

import strands
from strands.tools.executors import ConcurrentToolExecutor
from strands.tools.structured_output.structured_output_context import StructuredOutputContext
from strands.tools.structured_output.structured_output_tool import StructuredOutputTool
from strands.types._events import ToolResultEvent
from strands.types.tools import ToolUse


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""

    name: str
    age: int


@pytest.fixture
def executor():
    """Create a concurrent executor instance."""
    return ConcurrentToolExecutor()


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
async def test_concurrent_executor_passes_structured_output_context(
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
    """Test that concurrent executor properly passes structured output context to tools."""
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
async def test_structured_output_tool_integration(structured_output_context):
    """Test StructuredOutputTool integration with concurrent executor."""
    # Create a structured output tool
    structured_tool = StructuredOutputTool(SampleModel)

    # Test successful validation
    tool_use: ToolUse = {"name": SampleModel.__name__, "toolUseId": "test-1", "input": {"name": "Alice", "age": 30}}

    invocation_state = {}

    # Stream the tool
    events = []
    async for event in structured_tool.stream(
        tool_use, invocation_state, structured_output_context=structured_output_context
    ):
        events.append(event)

    # Verify the result
    assert len(events) == 1
    assert isinstance(events[0], ToolResultEvent)
    assert events[0].tool_use_id == "test-1"
    assert events[0].tool_result["status"] == "success"

    # Verify structured output was stored in context
    result = structured_output_context.get_result("test-1")
    assert result is not None
    assert result.name == "Alice"
    assert result.age == 30


@pytest.mark.asyncio
async def test_structured_output_tool_validation_error(structured_output_context):
    """Test StructuredOutputTool handles validation errors properly."""
    # Create a structured output tool
    structured_tool = StructuredOutputTool(SampleModel)

    # Test with invalid input (age as string instead of int)
    tool_use: ToolUse = {
        "name": SampleModel.__name__,
        "toolUseId": "test-2",
        "input": {"name": "Bob", "age": "invalid"},
    }

    invocation_state = {}

    # Stream the tool
    events = []
    async for event in structured_tool.stream(
        tool_use, invocation_state, structured_output_context=structured_output_context
    ):
        events.append(event)

    # Verify the error result
    assert len(events) == 1
    assert isinstance(events[0], ToolResultEvent)
    assert events[0].tool_use_id == "test-2"
    assert events[0].tool_result["status"] == "error"
    assert "validation failed" in events[0].tool_result["content"][0]["text"].lower()

    # Verify no result was stored due to validation error
    assert structured_output_context.get_result("test-2") is None


@pytest.mark.asyncio
async def test_concurrent_executor_with_multiple_tools(
    executor, agent, tool_results, cycle_trace, cycle_span, invocation_state, structured_output_context, alist
):
    """Test concurrent execution with both regular and structured output tools."""
    # Track which tools were called
    tools_called = []

    # Create a regular tool
    @strands.tool(name="track_tool")
    def track_func():
        tools_called.append("regular")
        return "tracked"

    # Register the regular tool
    agent.tool_registry.register_tool(track_func)

    # Create and register structured output tool
    structured_tool = StructuredOutputTool(SampleModel)
    agent.tool_registry.register_tool(structured_tool)

    # Set up multiple tool uses
    tool_uses: list[ToolUse] = [
        {"name": "track_tool", "toolUseId": "1", "input": {}},
        {"name": SampleModel.__name__, "toolUseId": "2", "input": {"name": "Charlie", "age": 25}},
        {"name": "track_tool", "toolUseId": "3", "input": {}},
    ]

    # Execute tools
    stream = executor._execute(
        agent, tool_uses, tool_results, cycle_trace, cycle_span, invocation_state, structured_output_context
    )

    # Collect events
    events = sorted(await alist(stream), key=lambda e: e.tool_use_id)

    # Verify all tools were executed
    assert len(events) == 3

    # Verify regular tools were called
    assert tools_called.count("regular") == 2

    # Verify structured output tool result
    structured_event = next((e for e in events if e.tool_use_id == "2"), None)
    assert structured_event is not None
    assert structured_event.tool_result["status"] == "success"

    # Verify structured output was stored
    result = structured_output_context.get_result("2")
    assert result is not None
    assert result.name == "Charlie"
    assert result.age == 25
