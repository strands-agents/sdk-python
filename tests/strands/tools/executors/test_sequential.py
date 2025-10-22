from pydantic import BaseModel
import pytest

from strands.hooks import BeforeToolCallEvent
from strands.interrupt import Interrupt
from strands.tools.decorator import tool
from strands.tools.executors import SequentialToolExecutor
from strands.tools.structured_output._structured_output_context import StructuredOutputContext
from strands.types._events import ToolInterruptEvent, ToolResultEvent
from strands.types.tools import ToolUse


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""

    name: str
    age: int


@pytest.fixture
def executor():
    return SequentialToolExecutor()


@pytest.fixture
def structured_output_context():
    """Create a structured output context with SampleModel."""
    return StructuredOutputContext(structured_output_model=SampleModel)


@pytest.fixture
def capture_tool():
    """Create a tool that captures kwargs passed to it."""
    captured_kwargs = {}

    @tool(name="capture_tool")
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
async def test_sequential_executor_execute(
    executor, agent, tool_results, cycle_trace, cycle_span, invocation_state, alist
):
    tool_uses = [
        {"name": "weather_tool", "toolUseId": "1", "input": {}},
        {"name": "temperature_tool", "toolUseId": "2", "input": {}},
    ]
    structured_output_context = StructuredOutputContext(None)
    stream = executor._execute(
        agent, tool_uses, tool_results, cycle_trace, cycle_span, invocation_state, structured_output_context
    )

    tru_events = await alist(stream)
    exp_events = [
        ToolResultEvent({"toolUseId": "1", "status": "success", "content": [{"text": "sunny"}]}),
        ToolResultEvent({"toolUseId": "2", "status": "success", "content": [{"text": "75F"}]}),
    ]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = [exp_events[0].tool_result, exp_events[1].tool_result]
    assert tru_results == exp_results


@pytest.mark.asyncio
async def test_sequential_executor_interrupt(
    executor, agent, tool_results, cycle_trace, cycle_span, invocation_state, alist
):
    interrupt = Interrupt(
        id="v1:test_tool_id_1:78714d6c-613c-5cf4-bf25-7037569941f9",
        name="test_name",
        reason="test reason",
    )

    def interrupt_callback(event):
        event.interrupt("test_name", "test reason")

    agent.hooks.add_callback(BeforeToolCallEvent, interrupt_callback)

    tool_uses = [
        {"name": "weather_tool", "toolUseId": "test_tool_id_1", "input": {}},
        {"name": "temperature_tool", "toolUseId": "test_tool_id_2", "input": {}},
    ]

    structured_output_context = StructuredOutputContext(None)
    stream = executor._execute(
        agent, tool_uses, tool_results, cycle_trace, cycle_span, invocation_state, structured_output_context
    )

    tru_events = await alist(stream)
    exp_events = [ToolInterruptEvent(tool_uses[0], [interrupt])]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = []
    assert tru_results == exp_results


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
