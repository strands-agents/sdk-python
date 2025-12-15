"""Tests for Agent structured output functionality."""

from unittest import mock
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from strands import Agent
from strands.telemetry.metrics import EventLoopMetrics
from strands.tools.structured_output._structured_output_context import StructuredOutputContext
from strands.tools.structured_output.structured_output_tool import StructuredOutputTool
from strands.types._events import EventLoopStopEvent
from tests.fixtures.mocked_model_provider import MockedModelProvider


class UserModel(BaseModel):
    """Test user model for structured output."""

    name: str
    age: int
    email: str


class ProductModel(BaseModel):
    """Test product model for structured output."""

    title: str
    price: float
    description: str | None = None


@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = Mock()

    async def mock_stream(*args, **kwargs):
        yield {"contentBlockDelta": {"delta": {"text": "test response"}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}

    model.stream.side_effect = lambda *args, **kwargs: mock_stream(*args, **kwargs)
    return model


@pytest.fixture
def mock_metrics():
    return mock.Mock(spec=EventLoopMetrics)


@pytest.fixture
def user_model():
    """Return the test user model class."""
    return UserModel


@pytest.fixture
def product_model():
    """Return the test product model class."""
    return ProductModel


class TestAgentStructuredOutputInit:
    """Test Agent initialization with structured output model."""

    def test_agent_init_with_structured_output_model(self, user_model):
        """Test that Agent can be initialized with a structured_output_model."""
        agent = Agent(structured_output_model=user_model)

        assert agent._default_structured_output_model == user_model
        assert agent.model is not None

    def test_agent_init_without_structured_output_model(self):
        """Test that Agent can be initialized without structured_output_model."""
        agent = Agent()

        assert agent._default_structured_output_model is None
        assert agent.model is not None


class TestAgentStructuredOutputInvocation:
    """Test Agent invocation with structured output."""

    @patch("strands.agent.agent.event_loop_cycle")
    def test_agent_call_with_structured_output_model(self, mock_event_loop, user_model, mock_model, mock_metrics):
        """Test Agent.__call__ with structured_output_model parameter."""

        async def mock_cycle(*args, **kwargs):
            structured_output_context = kwargs.get("structured_output_context")
            assert structured_output_context is not None
            assert structured_output_context.structured_output_model == user_model

            # Return a successful result
            test_user = UserModel(name="John", age=30, email="john@example.com")
            yield EventLoopStopEvent(
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "Response"}]},
                metrics=mock_metrics,
                request_state={},
                structured_output=test_user,
            )

        mock_event_loop.side_effect = mock_cycle

        # Create agent and call with structured_output_model
        agent = Agent(model=mock_model)
        agent("Extract user info", structured_output_model=user_model)

        # Verify event_loop_cycle was called with correct context
        mock_event_loop.assert_called_once()
        call_kwargs = mock_event_loop.call_args[1]
        assert "structured_output_context" in call_kwargs

    @patch("strands.agent.agent.event_loop_cycle")
    def test_agent_call_with_default_structured_output_model(
        self, mock_event_loop, product_model, mock_model, mock_metrics
    ):
        """Test Agent.__call__ uses default structured_output_model when not specified."""

        # Setup mock event loop
        pm = ProductModel(title="Widget", price=9.99)

        async def mock_cycle(*args, **kwargs):
            structured_output_context = kwargs.get("structured_output_context")
            assert structured_output_context is not None
            assert structured_output_context.structured_output_model == product_model

            yield EventLoopStopEvent(
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "Response"}]},
                metrics=mock_metrics,
                request_state={},
                structured_output=pm,
            )

        mock_event_loop.side_effect = mock_cycle

        # Create agent with default structured_output_model
        agent = Agent(model=mock_model, structured_output_model=product_model)
        result = agent("Get product info")

        # Verify result uses default model
        assert result.structured_output is pm

    @patch("strands.agent.agent.event_loop_cycle")
    def test_agent_call_override_default_structured_output_model(
        self, mock_event_loop, user_model, product_model, mock_model, mock_metrics
    ):
        """Test that invocation-level structured_output_model overrides default."""

        # Setup mock event loop
        um = UserModel(name="Jane", age=25, email="jane@example.com")

        async def mock_cycle(*args, **kwargs):
            structured_output_context = kwargs.get("structured_output_context")
            # Should use user_model, not the default product_model
            assert structured_output_context.structured_output_model == user_model

            yield EventLoopStopEvent(
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "Response"}]},
                metrics=mock_metrics,
                request_state={},
                structured_output=um,
            )

        mock_event_loop.side_effect = mock_cycle

        # Create agent with default product_model, but override with user_model
        agent = Agent(model=mock_model, structured_output_model=product_model)
        result = agent("Get user info", structured_output_model=user_model)

        # Verify result uses override model
        assert result.structured_output is um

    @pytest.mark.asyncio
    @patch("strands.agent.agent.event_loop_cycle")
    async def test_agent_invoke_async_with_structured_output(
        self, mock_event_loop, user_model, mock_model, mock_metrics
    ):
        """Test Agent.invoke_async with structured_output_model."""

        # Setup mock event loop
        um = UserModel(name="Alice", age=28, email="alice@example.com")

        async def mock_cycle(*args, **kwargs):
            structured_output_context = kwargs.get("structured_output_context")
            assert structured_output_context is not None
            assert structured_output_context.structured_output_model == user_model

            yield EventLoopStopEvent(
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "Response"}]},
                metrics=mock_metrics,
                request_state={},
                structured_output=um,
            )

        mock_event_loop.side_effect = mock_cycle

        # Create agent and call async
        agent = Agent(model=mock_model)
        result = await agent.invoke_async("Get user", structured_output_model=user_model)

        # Verify result
        assert result.structured_output is um

    @pytest.mark.asyncio
    @patch("strands.agent.agent.event_loop_cycle")
    async def test_agent_stream_async_with_structured_output(
        self, mock_event_loop, product_model, mock_model, mock_metrics
    ):
        """Test Agent.stream_async with structured_output_model."""

        # Setup mock event loop
        pm = ProductModel(title="Gadget", price=19.99, description="Cool gadget")

        async def mock_cycle(*args, **kwargs):
            structured_output_context = kwargs.get("structured_output_context")
            assert structured_output_context is not None
            assert structured_output_context.structured_output_model == product_model

            yield EventLoopStopEvent(
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "Response"}]},
                metrics=mock_metrics,
                request_state={},
                structured_output=pm,
            )

        mock_event_loop.side_effect = mock_cycle

        # Create agent and stream async
        agent = Agent(model=mock_model)
        events = []
        async for event in agent.stream_async("Get product", structured_output_model=product_model):
            events.append(event)

        # Verify we got result event
        assert len(events) > 0
        result_event = events[-1]
        assert "result" in result_event
        result = result_event["result"]
        assert result.structured_output is pm


class TestAgentStructuredOutputContext:
    """Test StructuredOutputContext integration with Agent."""

    @patch("strands.agent.agent.event_loop_cycle")
    def test_structured_output_context_created_with_model(self, mock_event_loop, user_model, mock_model, mock_metrics):
        """Test that StructuredOutputContext is created when structured_output_model is provided."""
        context = None

        async def mock_cycle(*args, **kwargs):
            nonlocal context
            context = kwargs.get("structured_output_context")
            yield EventLoopStopEvent(
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "Response"}]},
                metrics=mock_metrics,
                request_state={},
            )

        mock_event_loop.side_effect = mock_cycle

        agent = Agent(model=mock_model)
        agent("Test", structured_output_model=user_model)

        # Verify context was created and passed
        assert context is not None
        assert isinstance(context, StructuredOutputContext)
        assert context.structured_output_model == user_model
        assert context.is_enabled is True

    @patch("strands.agent.agent.event_loop_cycle")
    def test_structured_output_context_none_without_model(self, mock_event_loop, mock_model, mock_metrics):
        """Test that StructuredOutputContext is created with None when no model provided."""
        context = None

        async def mock_cycle(*args, **kwargs):
            nonlocal context
            context = kwargs.get("structured_output_context")
            yield EventLoopStopEvent(
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "Response"}]},
                metrics=mock_metrics,
                request_state={},
            )

        mock_event_loop.side_effect = mock_cycle

        agent = Agent(model=mock_model)
        agent("Test")  # No structured_output_model

        # Verify context was created but disabled
        assert context is not None
        assert isinstance(context, StructuredOutputContext)
        assert context.structured_output_model is None
        assert context.is_enabled is False

    @patch("strands.tools.registry.ToolRegistry.register_dynamic_tool")
    @patch("strands.agent.agent.event_loop_cycle")
    def test_structured_output_tool_registered_dynamically(
        self, mock_event_loop, mock_register, user_model, mock_model, mock_metrics
    ):
        """Test that StructuredOutputTool is registered dynamically when structured output is used."""
        captured_tool = None

        def capture_tool(tool):
            nonlocal captured_tool
            captured_tool = tool

        mock_register.side_effect = capture_tool

        async def mock_cycle(*args, **kwargs):
            yield EventLoopStopEvent(
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "Response"}]},
                metrics=mock_metrics,
                request_state={},
            )

        mock_event_loop.side_effect = mock_cycle

        agent = Agent(model=mock_model)
        agent("Test", structured_output_model=user_model)

        # Verify tool was registered
        mock_register.assert_called_once()
        assert captured_tool is not None
        assert isinstance(captured_tool, StructuredOutputTool)
        assert captured_tool.structured_output_model == user_model


class TestAgentStructuredOutputEdgeCases:
    """Test edge cases for structured output in Agent."""

    @patch("strands.agent.agent.event_loop_cycle")
    def test_agent_with_no_structured_output(self, mock_event_loop, mock_model, mock_metrics):
        """Test that agent works normally when no structured output is specified."""

        async def mock_cycle(*args, **kwargs):
            structured_output_context = kwargs.get("structured_output_context")
            assert structured_output_context is not None
            assert structured_output_context.structured_output_model is None
            assert structured_output_context.is_enabled is False

            yield EventLoopStopEvent(
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "Normal response"}]},
                metrics=mock_metrics,
                request_state={},
            )

        mock_event_loop.side_effect = mock_cycle

        agent = Agent(model=mock_model)
        result = agent("Normal query")

        # Result should not have structured output
        assert result.structured_output is None
        assert result.message["content"][0]["text"] == "Normal response"

    def test_agent_multiple_structured_output_models(self, user_model, product_model, mock_metrics):
        """Test that agent can switch between different structured output models."""
        model = MockedModelProvider(
            [
                {"role": "assistant", "content": [{"text": "User response"}]},
                {"role": "assistant", "content": [{"text": "Product response"}]},
            ]
        )

        agent = Agent(model=model)

        # First call with user model
        with patch("strands.agent.agent.event_loop_cycle") as mock_event_loop:
            um = UserModel(name="Bob", age=40, email="bob@example.com")

            async def mock_user_cycle(*args, **kwargs):
                ctx = kwargs.get("structured_output_context")
                assert ctx.structured_output_model == user_model
                yield EventLoopStopEvent(
                    stop_reason="end_turn",
                    message={"role": "assistant", "content": [{"text": "User response"}]},
                    metrics=mock_metrics,
                    request_state={},
                    structured_output=um,
                )

            mock_event_loop.side_effect = mock_user_cycle
            result1 = agent("Get user", structured_output_model=user_model)
            assert result1.structured_output is um

        # Second call with product model
        with patch("strands.agent.agent.event_loop_cycle") as mock_event_loop:
            pm = ProductModel(title="Item", price=5.99)

            async def mock_product_cycle(*args, **kwargs):
                ctx = kwargs.get("structured_output_context")
                assert ctx.structured_output_model == product_model
                yield EventLoopStopEvent(
                    stop_reason="end_turn",
                    message={"role": "assistant", "content": [{"text": "Product response"}]},
                    metrics=mock_metrics,
                    request_state={},
                    structured_output=pm,
                )

            mock_event_loop.side_effect = mock_product_cycle
            result2 = agent("Get product", structured_output_model=product_model)
            assert result2.structured_output is pm
