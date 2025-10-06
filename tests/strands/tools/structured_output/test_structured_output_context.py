"""Tests for StructuredOutputContext class."""

from typing import Optional

from pydantic import BaseModel, Field

from strands.tools.structured_output.structured_output_context import StructuredOutputContext
from strands.tools.structured_output.structured_output_tool import StructuredOutputTool


class SampleModel(BaseModel):
    """Test Pydantic model for testing."""

    name: str = Field(..., description="Name field")
    age: int = Field(..., description="Age field", ge=0)
    email: Optional[str] = Field(None, description="Optional email field")


class AnotherSampleModel(BaseModel):
    """Another test Pydantic model."""

    value: str
    count: int


class TestStructuredOutputContext:
    """Test suite for StructuredOutputContext."""

    def test_initialization_with_structured_output_model(self):
        """Test initialization with a structured output model."""
        context = StructuredOutputContext(structured_output_model=SampleModel)

        assert context.structured_output_model == SampleModel
        assert isinstance(context.structured_output_tool, StructuredOutputTool)
        assert context.expected_tool_name == "SampleModel"
        assert context.MAX_STRUCTURED_OUTPUT_ATTEMPTS == 3
        assert context.attempts == 0
        assert context.results == {}
        assert context.forced_mode is False
        assert context.tool_choice is None
        assert context.stop_loop is False

    def test_initialization_without_structured_output_model(self):
        """Test initialization without a structured output model."""
        context = StructuredOutputContext(structured_output_model=None)

        assert context.structured_output_model is None
        assert context.structured_output_tool is None
        assert context.expected_tool_name is None
        assert context.MAX_STRUCTURED_OUTPUT_ATTEMPTS == 3
        assert context.attempts == 0
        assert context.results == {}
        assert context.forced_mode is False
        assert context.tool_choice is None
        assert context.stop_loop is False

    def test_is_enabled_property(self):
        """Test the is_enabled property."""
        # Test with model
        context_with_model = StructuredOutputContext(structured_output_model=SampleModel)
        assert context_with_model.is_enabled is True

        # Test without model
        context_without_model = StructuredOutputContext(structured_output_model=None)
        assert context_without_model.is_enabled is False

    def test_store_result_and_get_result(self):
        """Test storing and retrieving results."""
        context = StructuredOutputContext(structured_output_model=SampleModel)

        # Create test result
        test_result = SampleModel(name="John Doe", age=30, email="john@example.com")
        tool_use_id = "test_tool_use_123"

        # Store result
        context.store_result(tool_use_id, test_result)
        assert tool_use_id in context.results
        assert context.results[tool_use_id] == test_result

        # Retrieve result
        retrieved_result = context.get_result(tool_use_id)
        assert retrieved_result == test_result

        # Test retrieving non-existent result
        non_existent = context.get_result("non_existent_id")
        assert non_existent is None

    def test_increment_attempts(self):
        """Test increment_attempts method and attempt counting."""
        context = StructuredOutputContext(structured_output_model=SampleModel)

        assert context.attempts == 0

        # First increment
        result1 = context.increment_attempts()
        assert result1 == 1
        assert context.attempts == 1

        # Second increment
        result2 = context.increment_attempts()
        assert result2 == 2
        assert context.attempts == 2

        # Third increment
        result3 = context.increment_attempts()
        assert result3 == 3
        assert context.attempts == 3

    def test_setup_retry(self):
        """Test setup_retry method."""
        context = StructuredOutputContext(structured_output_model=SampleModel)

        assert context.attempts == 0
        assert context.forced_mode is False
        assert context.tool_choice is None

        # Call setup_retry
        context.setup_retry()

        assert context.attempts == 1
        assert context.forced_mode is True
        assert context.tool_choice == {"any": {}}

    def test_can_retry_logic(self):
        """Test can_retry logic with MAX_STRUCTURED_OUTPUT_ATTEMPTS."""
        context = StructuredOutputContext(structured_output_model=SampleModel)

        # Initial state - can retry
        assert context.can_retry() is True

        # After 1 attempt - can retry
        context.increment_attempts()
        assert context.can_retry() is True

        # After 2 attempts - can retry
        context.increment_attempts()
        assert context.can_retry() is True

        # After 3 attempts - cannot retry (reached max)
        context.increment_attempts()
        assert context.can_retry() is False

        # After 4 attempts - still cannot retry
        context.increment_attempts()
        assert context.can_retry() is False

    def test_can_retry_when_disabled(self):
        """Test can_retry when context is not enabled."""
        context = StructuredOutputContext(structured_output_model=None)

        # Should always return False when not enabled
        assert context.can_retry() is False

        context.increment_attempts()
        assert context.can_retry() is False

        context.increment_attempts()
        assert context.can_retry() is False

    def test_set_forced_mode_with_tool_choice(self):
        """Test set_forced_mode with custom tool_choice."""
        context = StructuredOutputContext(structured_output_model=SampleModel)

        custom_tool_choice = {"specific": {"tool": "SampleModel"}}
        context.set_forced_mode(tool_choice=custom_tool_choice)

        assert context.forced_mode is True
        assert context.tool_choice == custom_tool_choice

    def test_set_forced_mode_without_tool_choice(self):
        """Test set_forced_mode without tool_choice (default)."""
        context = StructuredOutputContext(structured_output_model=SampleModel)

        context.set_forced_mode()

        assert context.forced_mode is True
        assert context.tool_choice == {"any": {}}

    def test_set_forced_mode_when_disabled(self):
        """Test set_forced_mode when context is not enabled."""
        context = StructuredOutputContext(structured_output_model=None)

        # Should not change state when not enabled
        context.set_forced_mode(tool_choice={"test": "value"})

        assert context.forced_mode is False
        assert context.tool_choice is None

    def test_has_structured_output_tool(self):
        """Test has_structured_output_tool method."""
        context = StructuredOutputContext(structured_output_model=SampleModel)

        # Create tool uses with the expected tool
        tool_uses_with_output = [
            {"name": "SampleModel", "toolUseId": "123", "input": {}},
            {"name": "OtherTool", "toolUseId": "456", "input": {}},
        ]

        # Should find the structured output tool
        assert context.has_structured_output_tool(tool_uses_with_output) is True

        # Create tool uses without the expected tool
        tool_uses_without_output = [
            {"name": "OtherTool", "toolUseId": "456", "input": {}},
            {"name": "AnotherTool", "toolUseId": "789", "input": {}},
        ]

        # Should not find the structured output tool
        assert context.has_structured_output_tool(tool_uses_without_output) is False

        # Test with empty list
        assert context.has_structured_output_tool([]) is False

    def test_has_structured_output_tool_when_disabled(self):
        """Test has_structured_output_tool when no expected tool name."""
        context = StructuredOutputContext(structured_output_model=None)

        tool_uses = [
            {"name": "SampleModel", "toolUseId": "123", "input": {}},
        ]

        # Should return False when no expected tool name
        assert context.has_structured_output_tool(tool_uses) is False

    def test_get_tool_spec(self):
        """Test get_tool_spec method."""
        context = StructuredOutputContext(structured_output_model=SampleModel)

        tool_spec = context.get_tool_spec()
        assert tool_spec is not None
        assert isinstance(tool_spec, dict)
        assert "name" in tool_spec
        assert tool_spec["name"] == "SampleModel"
        assert "description" in tool_spec
        assert "inputSchema" in tool_spec

    def test_get_tool_spec_when_disabled(self):
        """Test get_tool_spec when no structured output tool."""
        context = StructuredOutputContext(structured_output_model=None)

        tool_spec = context.get_tool_spec()
        assert tool_spec is None

    def test_extract_result(self):
        """Test extract_result method."""
        context = StructuredOutputContext(structured_output_model=SampleModel)

        # Store some results
        result1 = SampleModel(name="Alice", age=25)
        result2 = SampleModel(name="Bob", age=30)
        context.store_result("tool_use_1", result1)
        context.store_result("tool_use_2", result2)

        # Create tool uses with matching tool
        tool_uses = [
            {"name": "SampleModel", "toolUseId": "tool_use_1", "input": {}},
            {"name": "OtherTool", "toolUseId": "tool_use_3", "input": {}},
        ]

        # Extract result should return and remove the first matching result
        extracted = context.extract_result(tool_uses)
        assert extracted == result1
        assert "tool_use_1" not in context.results
        assert "tool_use_2" in context.results  # Other result should remain

    def test_extract_result_no_matching_tool(self):
        """Test extract_result when no matching tool in tool_uses."""
        context = StructuredOutputContext(structured_output_model=SampleModel)

        result = SampleModel(name="Alice", age=25)
        context.store_result("tool_use_1", result)

        # Tool uses without the expected tool name
        tool_uses = [
            {"name": "OtherTool", "toolUseId": "tool_use_1", "input": {}},
        ]

        # Should return None
        extracted = context.extract_result(tool_uses)
        assert extracted is None
        assert "tool_use_1" in context.results  # Result should remain

    def test_extract_result_no_stored_result(self):
        """Test extract_result when no stored result for tool use."""
        context = StructuredOutputContext(structured_output_model=SampleModel)

        # Tool uses with matching tool but no stored result
        tool_uses = [
            {"name": "SampleModel", "toolUseId": "tool_use_1", "input": {}},
        ]

        # Should return None
        extracted = context.extract_result(tool_uses)
        assert extracted is None

    def test_extract_result_multiple_matching_tools(self):
        """Test extract_result with multiple matching tool uses."""
        context = StructuredOutputContext(structured_output_model=SampleModel)

        # Store multiple results
        result1 = SampleModel(name="Alice", age=25)
        result2 = SampleModel(name="Bob", age=30)
        context.store_result("tool_use_1", result1)
        context.store_result("tool_use_2", result2)

        # Multiple matching tool uses
        tool_uses = [
            {"name": "SampleModel", "toolUseId": "tool_use_1", "input": {}},
            {"name": "SampleModel", "toolUseId": "tool_use_2", "input": {}},
        ]

        # Should extract the first matching result
        extracted = context.extract_result(tool_uses)
        assert extracted == result1
        assert "tool_use_1" not in context.results
        assert "tool_use_2" in context.results

        # Extract again for the second result
        extracted2 = context.extract_result(tool_uses)
        assert extracted2 == result2
        assert "tool_use_2" not in context.results
