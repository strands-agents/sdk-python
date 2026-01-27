"""
Comprehensive integration tests for structured output passed into the agent functionality.
"""

from pydantic import BaseModel, Field

from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.tools import tool


class MathResult(BaseModel):
    """Math operation result."""

    operation: str = Field(description="the performed operation")
    result: int = Field(description="the result of the operation")


# ========== Tool Definitions ==========


@tool
def calculator(operation: str, a: float, b: float) -> float:
    """Simple calculator tool for testing."""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return b / a if a != 0 else 0
    elif operation == "power":
        return a**b
    else:
        return 0


# ========== Test Classes ==========


class TestBedrockLlamaModelsToolUsageWithStructuredOutput:
    """Test structured output with tool usage."""

    def test_multi_turn_calculator_tool_use_with_structured_output(self):
        """Test tool usage with structured output."""
        model = BedrockModel(
            model_id="us.meta.llama4-maverick-17b-instruct-v1:0",
            region_name="us-east-1",
            max_tokens=2048,
            streaming=False,
        )
        agent = Agent(model=model, tools=[calculator])

        result = agent("Calculate 2 + 2 using the calculator tool", structured_output_model=MathResult)

        assert result.structured_output is not None
        assert isinstance(result.structured_output, MathResult)
        assert result.structured_output.result == 4
        # Check that tool was called
        assert result.metrics.tool_metrics is not None
        assert len(result.metrics.tool_metrics) > 0
        result = agent("What is 5 multiplied by 3? Use the calculator tool.", structured_output_model=MathResult)
        assert result.structured_output is not None
