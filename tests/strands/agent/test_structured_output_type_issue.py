"""Reproduction test demonstrating type inference issue with structured output.

This test file demonstrates the current limitation where mypy cannot infer
the specific type of structured_output from AgentResult when an agent is
initialized with structured_output_model.

Expected mypy errors (before fix):
- Line ~40: error: Item "None" has no attribute "name"
- Line ~41: error: Item "None" has no attribute "age"
- Line ~42: error: Item "BaseModel" has no attribute "name"
- Line ~43: error: Item "BaseModel" has no attribute "age"

After implementing generic types, these errors should be resolved without
needing explicit type casting.
"""

from typing import cast
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from strands import Agent
from strands.agent.agent_result import AgentResult
from strands.telemetry.metrics import EventLoopMetrics
from strands.types._events import EventLoopStopEvent


class TestOutputModel(BaseModel):
    """Test model for structured output type inference."""

    name: str
    age: int
    email: str


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


class TestStructuredOutputTypeInference:
    """Test cases demonstrating the type inference issue."""

    @patch("strands.agent.agent.event_loop_cycle")
    def test_current_type_inference_issue(self, mock_event_loop, mock_model, mock_metrics):
        """Demonstrate current type inference limitation requiring explicit casting.

        This test shows that with the current implementation, mypy cannot infer
        the specific type of structured_output, requiring manual type casting.
        """
        # Setup mock event loop to return structured output
        test_output = TestOutputModel(name="John Doe", age=30, email="john@example.com")

        async def mock_cycle(*args, **kwargs):
            yield EventLoopStopEvent(
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "Response"}]},
                metrics=mock_metrics,
                request_state={},
                structured_output=test_output,
            )

        mock_event_loop.side_effect = mock_cycle

        # Create agent with structured_output_model
        agent = Agent(model=mock_model, structured_output_model=TestOutputModel)
        result = agent("Extract user info")

        # Current issue: mypy sees result.structured_output as BaseModel | None
        # This requires explicit casting to access model-specific fields
        assert result.structured_output is not None

        # After fix: mypy should understand the specific type
        # No type: ignore needed
        name = result.structured_output.name
        age = result.structured_output.age

        assert name == "John Doe"
        assert age == 30

    @patch("strands.agent.agent.event_loop_cycle")
    def test_workaround_with_explicit_casting(self, mock_event_loop, mock_model, mock_metrics):
        """Show the current workaround using explicit type casting.

        This is what developers currently need to do to satisfy mypy.
        """
        test_output = TestOutputModel(name="Jane Smith", age=25, email="jane@example.com")

        async def mock_cycle(*args, **kwargs):
            yield EventLoopStopEvent(
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "Response"}]},
                metrics=mock_metrics,
                request_state={},
                structured_output=test_output,
            )

        mock_event_loop.side_effect = mock_cycle

        agent = Agent(model=mock_model, structured_output_model=TestOutputModel)
        result = agent("Extract user info")

        # Current workaround: explicit casting
        if result.structured_output is not None:
            typed_output = cast(TestOutputModel, result.structured_output)
            name = typed_output.name  # mypy is happy with this
            age = typed_output.age

            assert name == "Jane Smith"
            assert age == 25

    @patch("strands.agent.agent.event_loop_cycle")
    def test_desired_type_inference_behavior(self, mock_event_loop, mock_model, mock_metrics):
        """Demonstrate the desired behavior after implementing generic types.

        After the fix, this test should pass mypy without type: ignore comments
        or explicit casting.
        """
        test_output = TestOutputModel(name="Bob Johnson", age=35, email="bob@example.com")

        async def mock_cycle(*args, **kwargs):
            yield EventLoopStopEvent(
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "Response"}]},
                metrics=mock_metrics,
                request_state={},
                structured_output=test_output,
            )

        mock_event_loop.side_effect = mock_cycle

        # Create agent with structured_output_model
        agent = Agent(model=mock_model, structured_output_model=TestOutputModel)
        result = agent("Extract user info")

        # After fix: mypy should understand that result is AgentResult[TestOutputModel]
        # and structured_output is TestOutputModel | None
        assert result.structured_output is not None

        # These should work without type: ignore after the fix
        name = result.structured_output.name
        age = result.structured_output.age
        email = result.structured_output.email

        assert name == "Bob Johnson"
        assert age == 35
        assert email == "bob@example.com"

    @pytest.mark.asyncio
    @patch("strands.agent.agent.event_loop_cycle")
    async def test_async_invocation_type_inference(self, mock_event_loop, mock_model, mock_metrics):
        """Test type inference with async invocation."""
        test_output = TestOutputModel(name="Alice Brown", age=28, email="alice@example.com")

        async def mock_cycle(*args, **kwargs):
            yield EventLoopStopEvent(
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "Response"}]},
                metrics=mock_metrics,
                request_state={},
                structured_output=test_output,
            )

        mock_event_loop.side_effect = mock_cycle

        agent = Agent(model=mock_model, structured_output_model=TestOutputModel)
        result = await agent.invoke_async("Extract user info")

        # After fix: async invocation should also have proper type inference
        assert result.structured_output is not None
        name = result.structured_output.name
        assert name == "Alice Brown"

    def test_agent_result_type_annotation(self):
        """Test that AgentResult type annotation works correctly.

        This demonstrates that even with explicit type annotations,
        the current implementation doesn't provide proper type inference.
        """
        # Create a mock result
        test_output = TestOutputModel(name="Test User", age=40, email="test@example.com")

        result: AgentResult = AgentResult(
            stop_reason="end_turn",
            message={"role": "assistant", "content": [{"text": "Response"}]},
            metrics=EventLoopMetrics(),
            state={},
            structured_output=test_output,
        )

        # After fix: explicit type annotations should work correctly
        assert result.structured_output is not None
        name = result.structured_output.name
        assert name == "Test User"

    @patch("strands.agent.agent.event_loop_cycle")
    def test_none_checking_requirement(self, mock_event_loop, mock_model, mock_metrics):
        """Test that None checking is properly required.

        After the fix, mypy should still require None checking before
        accessing fields on structured_output.
        """
        test_output = TestOutputModel(name="Charlie Davis", age=45, email="charlie@example.com")

        async def mock_cycle(*args, **kwargs):
            yield EventLoopStopEvent(
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"text": "Response"}]},
                metrics=mock_metrics,
                request_state={},
                structured_output=test_output,
            )

        mock_event_loop.side_effect = mock_cycle

        agent = Agent(model=mock_model, structured_output_model=TestOutputModel)
        result = agent("Extract user info")

        # Without None check, mypy should error (even after fix)
        # This is correct behavior - structured_output can be None
        if result.structured_output is not None:
            # After None check, should be able to access fields
            name = result.structured_output.name
            assert name == "Charlie Davis"
