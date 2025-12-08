"""Tests for structured output integration in the event loop."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from strands.event_loop.event_loop import event_loop_cycle, recurse_event_loop
from strands.telemetry.metrics import EventLoopMetrics
from strands.tools.registry import ToolRegistry
from strands.tools.structured_output._structured_output_context import StructuredOutputContext
from strands.types._events import EventLoopStopEvent, StructuredOutputEvent


class UserModel(BaseModel):
    """Test model for structured output."""

    name: str
    age: int
    email: str


class ProductModel(BaseModel):
    """Another test model."""

    title: str
    price: float
    in_stock: bool


@pytest.fixture
def mock_agent():
    """Create a mock agent with required attributes."""
    agent = Mock(name="agent")
    agent.model = Mock()
    agent.system_prompt = "Test system prompt"
    agent.messages = []
    agent.tool_registry = ToolRegistry()
    agent.event_loop_metrics = EventLoopMetrics()
    agent.hooks = Mock()
    agent.hooks.invoke_callbacks_async = AsyncMock()
    agent.trace_span = None
    agent.trace_attributes = {}
    agent.tool_executor = Mock()
    agent._append_message = AsyncMock()

    # Set up _interrupt_state properly
    agent._interrupt_state = Mock()
    agent._interrupt_state.activated = False
    agent._interrupt_state.context = {}

    return agent


@pytest.fixture
def structured_output_context():
    """Create a structured output context with a test model."""
    return StructuredOutputContext(structured_output_model=UserModel)


@pytest.fixture
def agenerator():
    """Helper to create async generators."""

    def _agenerator(items):
        async def gen():
            for item in items:
                yield item

        return gen()

    return _agenerator


@pytest.fixture
def alist():
    """Helper to consume async generators."""

    async def _alist(async_gen):
        items = []
        async for item in async_gen:
            items.append(item)
        return items

    return _alist


@pytest.mark.asyncio
async def test_event_loop_cycle_with_structured_output_context(mock_agent, agenerator, alist):
    """Test event_loop_cycle with structured output context passed but not enabled."""
    # Create a context that's not enabled (no model)
    structured_output_context = StructuredOutputContext()

    # Setup model to return a text response
    mock_agent.model.stream.return_value = agenerator(
        [
            {"contentBlockDelta": {"delta": {"text": "Here is the user data"}}},
            {"contentBlockStop": {}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
    )

    # Run event loop cycle with structured output context
    stream = event_loop_cycle(
        agent=mock_agent,
        invocation_state={},
        structured_output_context=structured_output_context,
    )
    events = await alist(stream)

    # Should have received events
    assert len(events) > 0

    # The context should be passed through but not enabled
    assert not structured_output_context.is_enabled


@pytest.mark.asyncio
async def test_event_loop_forces_structured_output_on_end_turn(
    mock_agent, structured_output_context, agenerator, alist
):
    """Test that event loop forces structured output tool when model returns end_turn."""
    # First call returns end_turn without using structured output tool
    mock_agent.model.stream.side_effect = [
        agenerator(
            [
                {"contentBlockDelta": {"delta": {"text": "Here is the user info"}}},
                {"contentBlockStop": {}},
                {"messageStop": {"stopReason": "end_turn"}},
            ]
        ),
        # Second call (forced) uses the structured output tool
        agenerator(
            [
                {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "toolUseId": "t1",
                                "name": "UserModel",
                            }
                        }
                    }
                },
                {
                    "contentBlockDelta": {
                        "delta": {"toolUse": {"input": '{"name": "John", "age": 30, "email": "john@example.com"}'}}
                    }
                },
                {"contentBlockStop": {}},
                {"messageStop": {"stopReason": "tool_use"}},
            ]
        ),
    ]

    # Mock tool executor to handle the structured output tool
    mock_agent.tool_executor._execute = Mock(
        return_value=agenerator(
            [
                # Tool execution events would go here
            ]
        )
    )

    # Mock recurse_event_loop to return final result
    with patch("strands.event_loop.event_loop.recurse_event_loop") as mock_recurse:
        # Create a mock EventLoopStopEvent with the expected structure
        mock_stop_event = Mock()
        mock_stop_event.stop = (
            "end_turn",
            {"role": "assistant", "content": [{"text": "Done"}]},
            mock_agent.event_loop_metrics,
            {},
            None,
            UserModel(name="John", age=30, email="john@example.com"),
        )
        mock_stop_event.__getitem__ = lambda self, key: {"stop": self.stop}[key]

        mock_recurse.return_value = agenerator([mock_stop_event])

        stream = event_loop_cycle(
            agent=mock_agent,
            invocation_state={},
            structured_output_context=structured_output_context,
        )
        await alist(stream)

        # Should have appended a message to force structured output
        mock_agent._append_message.assert_called_once()
        args = mock_agent._append_message.call_args[0][0]
        assert args["role"] == "user"

        # Should have called recurse_event_loop with the context
        mock_recurse.assert_called_once()
        call_kwargs = mock_recurse.call_args[1]
        assert call_kwargs["structured_output_context"] == structured_output_context


@pytest.mark.asyncio
async def test_structured_output_tool_execution_extracts_result(
    mock_agent, structured_output_context, agenerator, alist
):
    """Test that structured output result is extracted from tool execution."""
    # Model uses the structured output tool
    mock_agent.model.stream.return_value = agenerator(
        [
            {
                "contentBlockStart": {
                    "start": {
                        "toolUse": {
                            "toolUseId": "t1",
                            "name": "UserModel",
                        }
                    }
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"toolUse": {"input": '{"name": "Alice", "age": 25, "email": "alice@test.com"}'}}
                }
            },
            {"contentBlockStop": {}},
            {"messageStop": {"stopReason": "tool_use"}},
        ]
    )

    # Mock the tool executor to return an async generator
    mock_agent.tool_executor._execute = Mock(return_value=agenerator([]))

    # Mock extract_result to return a model instance
    test_result = UserModel(name="Alice", age=25, email="alice@test.com")
    structured_output_context.extract_result = Mock(return_value=test_result)

    stream = event_loop_cycle(
        agent=mock_agent,
        invocation_state={},
        structured_output_context=structured_output_context,
    )
    events = await alist(stream)

    # Should yield StructuredOutputEvent
    structured_output_events = [e for e in events if isinstance(e, StructuredOutputEvent)]
    assert len(structured_output_events) == 1
    assert structured_output_events[0]["structured_output"] == test_result

    # Extract_result should have been called
    structured_output_context.extract_result.assert_called_once()


@pytest.mark.asyncio
async def test_structured_output_context_not_enabled(mock_agent, agenerator, alist):
    """Test event loop with structured output context that's not enabled."""
    # Create a context that's not enabled (no model)
    structured_output_context = StructuredOutputContext()
    assert not structured_output_context.is_enabled

    # Model returns end_turn
    mock_agent.model.stream.return_value = agenerator(
        [
            {"contentBlockDelta": {"delta": {"text": "Regular response"}}},
            {"contentBlockStop": {}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
    )

    stream = event_loop_cycle(
        agent=mock_agent,
        invocation_state={},
        structured_output_context=structured_output_context,
    )
    events = await alist(stream)

    # Should complete normally without forcing structured output
    stop_events = [e for e in events if isinstance(e, EventLoopStopEvent)]
    assert len(stop_events) == 1
    assert stop_events[0]["stop"][-1] is None


@pytest.mark.asyncio
async def test_structured_output_forced_mode(mock_agent, agenerator, alist):
    """Test event loop with structured output in forced mode."""
    # Create context in forced mode
    structured_output_context = StructuredOutputContext(structured_output_model=ProductModel)
    structured_output_context.set_forced_mode(tool_choice={"tool": {"name": "ProductModel"}})

    # Model should be called with only the structured output tool spec
    mock_agent.model.stream.return_value = agenerator(
        [
            {
                "contentBlockStart": {
                    "start": {
                        "toolUse": {
                            "toolUseId": "t1",
                            "name": "ProductModel",
                        }
                    }
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"toolUse": {"input": '{"title": "Book", "price": 19.99, "in_stock": true}'}}
                }
            },
            {"contentBlockStop": {}},
            {"messageStop": {"stopReason": "tool_use"}},
        ]
    )

    # Mock tool executor
    mock_agent.tool_executor._execute = Mock(return_value=agenerator([]))

    # Mock extract_result
    test_result = ProductModel(title="Book", price=19.99, in_stock=True)
    structured_output_context.extract_result = Mock(return_value=test_result)

    stream = event_loop_cycle(
        agent=mock_agent,
        invocation_state={},
        structured_output_context=structured_output_context,
    )
    await alist(stream)

    # Verify model.stream was called with the forced tool spec
    mock_agent.model.stream.assert_called_once()
    call_args = mock_agent.model.stream.call_args

    # The model.stream method signature (from streaming.py) is:
    # model.stream(messages, tool_specs, system_prompt, tool_choice=tool_choice)
    tool_specs = call_args.args[1] if len(call_args.args) > 1 else None

    # In forced mode, only the structured output tool spec should be passed
    assert tool_specs is not None, "Expected tool_specs to be provided"
    assert isinstance(tool_specs, list), f"Expected tool_specs to be a list, got {type(tool_specs)}"
    assert len(tool_specs) == 1
    assert tool_specs[0]["name"] == "ProductModel"


@pytest.mark.asyncio
async def test_recurse_event_loop_with_structured_output(mock_agent, structured_output_context, agenerator, alist):
    """Test recurse_event_loop preserves structured output context."""
    invocation_state = {
        "event_loop_cycle_trace": Mock(),
        "request_state": {},
    }

    # Mock event_loop_cycle to verify it receives the context
    with patch("strands.event_loop.event_loop.event_loop_cycle") as mock_cycle:
        # Create a mock EventLoopStopEvent with the expected structure
        mock_stop_event = Mock(spec=EventLoopStopEvent)
        mock_stop_event.stop = (
            "end_turn",
            {"role": "assistant", "content": [{"text": "Done"}]},
            mock_agent.event_loop_metrics,
            {},
            None,
            UserModel(name="Test", age=20, email="test@example.com"),
        )
        mock_stop_event.__getitem__ = lambda self, key: {"stop": self.stop}[key]

        mock_cycle.return_value = agenerator([mock_stop_event])

        stream = recurse_event_loop(
            agent=mock_agent,
            invocation_state=invocation_state,
            structured_output_context=structured_output_context,
        )
        events = await alist(stream)

        # Verify event_loop_cycle was called with the context
        mock_cycle.assert_called_once()
        call_kwargs = mock_cycle.call_args[1]
        assert call_kwargs["structured_output_context"] == structured_output_context

        # Verify the result includes structured output
        stop_events = [
            e for e in events if isinstance(e, EventLoopStopEvent) or (hasattr(e, "stop") and hasattr(e, "__getitem__"))
        ]
        assert len(stop_events) == 1
        stop_event = stop_events[0]
        if hasattr(stop_event, "__getitem__"):
            assert stop_event["stop"][5].name == "Test"
        else:
            assert stop_event.stop[5].name == "Test"


@pytest.mark.asyncio
async def test_structured_output_stops_loop_after_extraction(mock_agent, structured_output_context, agenerator, alist):
    """Test that loop stops after structured output is extracted."""
    # Model uses the structured output tool
    mock_agent.model.stream.return_value = agenerator(
        [
            {
                "contentBlockStart": {
                    "start": {
                        "toolUse": {
                            "toolUseId": "t1",
                            "name": "UserModel",
                        }
                    }
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"toolUse": {"input": '{"name": "Bob", "age": 35, "email": "bob@test.com"}'}}
                }
            },
            {"contentBlockStop": {}},
            {"messageStop": {"stopReason": "tool_use"}},
        ]
    )

    # Mock tool executor
    mock_agent.tool_executor._execute = Mock(return_value=agenerator([]))

    # Mock extract_result to return a result and set stop_loop
    test_result = UserModel(name="Bob", age=35, email="bob@test.com")

    def mock_extract(tool_uses):
        structured_output_context.stop_loop = True
        return test_result

    structured_output_context.extract_result = Mock(side_effect=mock_extract)

    stream = event_loop_cycle(
        agent=mock_agent,
        invocation_state={},
        structured_output_context=structured_output_context,
    )
    events = await alist(stream)

    # Should have a StructuredOutputEvent with the result
    structured_output_events = [e for e in events if isinstance(e, StructuredOutputEvent)]
    assert len(structured_output_events) == 1
    assert structured_output_events[0]["structured_output"] == test_result

    # Verify stop_loop was set
    assert structured_output_context.stop_loop

    # Extract_result should have been called
    structured_output_context.extract_result.assert_called_once()
