"""Type inference tests for Agent with structured output.

This test file validates that mypy correctly infers types when using
Agent with structured_output_model. These tests are designed to pass
mypy type checking without requiring explicit type casting.

Run mypy on this file to verify type inference:
    mypy tests/strands/agent/test_agent_type_inference.py
"""

from typing import Type
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from strands import Agent
from strands.agent.agent_result import AgentResult
from strands.telemetry.metrics import EventLoopMetrics
from strands.types._events import EventLoopStopEvent


class UserProfile(BaseModel):
    """Test model for user profile data."""

    name: str
    age: int
    email: str


class ProductInfo(BaseModel):
    """Test model for product information."""

    product_id: str
    price: float
    in_stock: bool


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
    return Mock(spec=EventLoopMetrics)


class TestAgentTypeInferenceWithModel:
    """Test type inference when Agent is initialized with structured_output_model."""

    @patch("strands.agent.agent.event_loop_cycle")
    def test_type_inference_with_structured_output_model(self, mock_event_loop, mock_model, mock_metrics):
        """Test that mypy infers AgentResult[UserProfile] when agent has structured_output_model."""
        test_output = UserProfile(name="John Doe", age=30, email="john@example.com")

        async def mock_cycle(*args, **kwargs):
            yield EventLoopStopEvent(
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "Response"}]},
                metrics=mock_metrics,
                request_state={},
                structured_output=test_output,
            )

        mock_event_loop.side_effect = mock_cycle

        # Type inference: agent should be Agent[UserProfile]
        agent = Agent(model=mock_model, structured_output_model=UserProfile)

        # Type inference: result should be AgentResult[UserProfile]
        result = agent("Extract user info")

        # Type inference: structured_output should be UserProfile | None
        assert result.structured_output is not None

        # After None check, mypy should know these fields exist
        # No type: ignore needed after fix
        name: str = result.structured_output.name
        age: int = result.structured_output.age
        email: str = result.structured_output.email

        assert name == "John Doe"
        assert age == 30
        assert email == "john@example.com"

    @pytest.mark.asyncio
    @patch("strands.agent.agent.event_loop_cycle")
    async def test_async_invocation_type_inference(self, mock_event_loop, mock_model, mock_metrics):
        """Test type inference with async invocation."""
        test_output = UserProfile(name="Jane Smith", age=25, email="jane@example.com")

        async def mock_cycle(*args, **kwargs):
            yield EventLoopStopEvent(
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "Response"}]},
                metrics=mock_metrics,
                request_state={},
                structured_output=test_output,
            )

        mock_event_loop.side_effect = mock_cycle

        agent = Agent(model=mock_model, structured_output_model=UserProfile)
        result = await agent.invoke_async("Extract user info")

        assert result.structured_output is not None
        name: str = result.structured_output.name
        assert name == "Jane Smith"

    @patch("strands.agent.agent.event_loop_cycle")
    def test_explicit_type_annotation(self, mock_event_loop, mock_model, mock_metrics):
        """Test that explicit type annotations work correctly."""
        test_output = ProductInfo(product_id="ABC123", price=99.99, in_stock=True)

        async def mock_cycle(*args, **kwargs):
            yield EventLoopStopEvent(
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "Response"}]},
                metrics=mock_metrics,
                request_state={},
                structured_output=test_output,
            )

        mock_event_loop.side_effect = mock_cycle

        agent: Agent[ProductInfo] = Agent(model=mock_model, structured_output_model=ProductInfo)
        result: AgentResult[ProductInfo] = agent("Get product info")

        assert result.structured_output is not None
        product_id: str = result.structured_output.product_id
        price: float = result.structured_output.price
        in_stock: bool = result.structured_output.in_stock

        assert product_id == "ABC123"
        assert price == 99.99
        assert in_stock is True


class TestAgentTypeInferenceWithoutModel:
    """Test backward compatibility when Agent is used without structured_output_model."""

    @patch("strands.agent.agent.event_loop_cycle")
    def test_type_inference_without_model(self, mock_event_loop, mock_model, mock_metrics):
        """Test that Agent without structured_output_model defaults to BaseModel."""
        async def mock_cycle(*args, **kwargs):
            yield EventLoopStopEvent(
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "Response"}]},
                metrics=mock_metrics,
                request_state={},
                structured_output=None,
            )

        mock_event_loop.side_effect = mock_cycle

        # Type inference: agent should be Agent[BaseModel]
        agent = Agent(model=mock_model)

        # Type inference: result should be AgentResult[BaseModel]
        result = agent("Hello")

        # Type inference: structured_output should be BaseModel | None
        output: BaseModel | None = result.structured_output
        assert output is None

    @patch("strands.agent.agent.event_loop_cycle")
    def test_backward_compatibility_with_none_output(self, mock_event_loop, mock_model, mock_metrics):
        """Test that None structured_output works correctly."""
        async def mock_cycle(*args, **kwargs):
            yield EventLoopStopEvent(
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "Response"}]},
                metrics=mock_metrics,
                request_state={},
                structured_output=None,
            )

        mock_event_loop.side_effect = mock_cycle

        agent = Agent(model=mock_model, structured_output_model=UserProfile)
        result = agent("Extract user info")

        # structured_output can be None
        assert result.structured_output is None


class TestGenericFunctionTypePreservation:
    """Test that generic type parameters are preserved through function calls."""

    def test_generic_function_return_type(self, mock_model):
        """Test type preservation through generic function."""

        def create_typed_agent(model_type: Type[UserProfile]) -> Agent[UserProfile]:
            """Create an agent with specific model type."""
            return Agent(model=mock_model, structured_output_model=model_type)

        # Type should be preserved through function
        agent: Agent[UserProfile] = create_typed_agent(UserProfile)
        assert agent._default_structured_output_model == UserProfile

    @patch("strands.agent.agent.event_loop_cycle")
    def test_generic_function_with_invocation(self, mock_event_loop, mock_model, mock_metrics):
        """Test type preservation through generic function with invocation."""
        test_output = UserProfile(name="Test User", age=40, email="test@example.com")

        async def mock_cycle(*args, **kwargs):
            yield EventLoopStopEvent(
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "Response"}]},
                metrics=mock_metrics,
                request_state={},
                structured_output=test_output,
            )

        mock_event_loop.side_effect = mock_cycle

        def create_and_invoke(model_type: Type[UserProfile], prompt: str) -> AgentResult[UserProfile]:
            """Create agent and invoke it."""
            agent = Agent(model=mock_model, structured_output_model=model_type)
            return agent(prompt)

        result = create_and_invoke(UserProfile, "Extract user info")
        assert result.structured_output is not None
        name: str = result.structured_output.name
        assert name == "Test User"


class TestNoneCheckingRequirements:
    """Test that mypy properly requires None checking for structured_output."""

    @patch("strands.agent.agent.event_loop_cycle")
    def test_none_check_required_before_field_access(self, mock_event_loop, mock_model, mock_metrics):
        """Test that None checking is required before accessing fields.

        This test verifies that mypy requires None checking even with proper type inference.
        The structured_output field is Optional (T | None), so None checks are necessary.
        """
        test_output = UserProfile(name="Alice Brown", age=28, email="alice@example.com")

        async def mock_cycle(*args, **kwargs):
            yield EventLoopStopEvent(
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "Response"}]},
                metrics=mock_metrics,
                request_state={},
                structured_output=test_output,
            )

        mock_event_loop.side_effect = mock_cycle

        agent = Agent(model=mock_model, structured_output_model=UserProfile)
        result = agent("Extract user info")

        # Without None check, mypy should error (if strict optional checking is enabled)
        # With None check, field access should work
        if result.structured_output is not None:
            name: str = result.structured_output.name
            age: int = result.structured_output.age
            assert name == "Alice Brown"
            assert age == 28

    @patch("strands.agent.agent.event_loop_cycle")
    def test_none_value_handling(self, mock_event_loop, mock_model, mock_metrics):
        """Test that None values are properly typed."""
        async def mock_cycle(*args, **kwargs):
            yield EventLoopStopEvent(
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "Response"}]},
                metrics=mock_metrics,
                request_state={},
                structured_output=None,
            )

        mock_event_loop.side_effect = mock_cycle

        agent = Agent(model=mock_model, structured_output_model=UserProfile)
        result = agent("Extract user info")

        # Type inference: structured_output is UserProfile | None
        output: UserProfile | None = result.structured_output
        assert output is None

        # Attempting to access fields without None check should be caught by mypy
        # (This would fail at runtime if not checked)
        if output is not None:
            _ = output.name  # This line is never reached but shows proper None checking
