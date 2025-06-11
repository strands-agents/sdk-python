"""
Tests for the function-based tool decorator pattern.
"""

from typing import Any, Dict, Optional, Union
from unittest.mock import MagicMock

from strands.tools.decorator import tool


def test_basic_tool_creation():
    """Test basic tool decorator functionality."""

    @tool
    def test_tool(param1: str, param2: int) -> str:
        """Test tool function.

        Args:
            param1: First parameter
            param2: Second parameter
        """
        return f"Result: {param1} {param2}"

    # Check TOOL_SPEC was generated correctly
    assert hasattr(test_tool, "TOOL_SPEC")
    spec = test_tool.TOOL_SPEC

    # Check basic spec properties
    assert spec["name"] == "test_tool"
    assert (
        spec["description"]
        == """Test tool function.

Args:
    param1: First parameter
    param2: Second parameter"""
    )

    # Check input schema
    schema = spec["inputSchema"]["json"]
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"param1", "param2"}

    # Check parameter properties
    assert schema["properties"]["param1"]["type"] == "string"
    assert schema["properties"]["param2"]["type"] == "integer"
    assert schema["properties"]["param1"]["description"] == "First parameter"
    assert schema["properties"]["param2"]["description"] == "Second parameter"

    # Test actual usage
    tool_use = {"toolUseId": "test-id", "input": {"param1": "hello", "param2": 42}}
    result = test_tool(tool_use)
    assert result["toolUseId"] == "test-id"
    assert result["status"] == "success"
    assert result["content"][0]["text"] == "Result: hello 42"


def test_tool_with_custom_name_description():
    """Test tool decorator with custom name and description."""

    @tool(name="custom_name", description="Custom description")
    def test_tool(param: str) -> str:
        return f"Result: {param}"

    spec = test_tool.TOOL_SPEC
    assert spec["name"] == "custom_name"
    assert spec["description"] == "Custom description"


def test_tool_with_optional_params():
    """Test tool decorator with optional parameters."""

    @tool
    def test_tool(required: str, optional: Optional[int] = None) -> str:
        """Test with optional param.

        Args:
            required: Required parameter
            optional: Optional parameter
        """
        if optional is None:
            return f"Result: {required}"
        return f"Result: {required} {optional}"

    spec = test_tool.TOOL_SPEC
    schema = spec["inputSchema"]["json"]

    # Only required should be in required list
    assert "required" in schema["required"]
    assert "optional" not in schema["required"]

    # Test with only required param
    tool_use = {"toolUseId": "test-id", "input": {"required": "hello"}}

    result = test_tool(tool_use)
    assert result["status"] == "success"
    assert result["content"][0]["text"] == "Result: hello"

    # Test with both params
    tool_use = {"toolUseId": "test-id", "input": {"required": "hello", "optional": 42}}

    result = test_tool(tool_use)
    assert result["status"] == "success"
    assert result["content"][0]["text"] == "Result: hello 42"


def test_tool_error_handling():
    """Test error handling in tool decorator."""

    @tool
    def test_tool(required: str) -> str:
        """Test tool function."""
        if required == "error":
            raise ValueError("Test error")
        return f"Result: {required}"

    # Test with missing required param
    tool_use = {"toolUseId": "test-id", "input": {}}

    result = test_tool(tool_use)
    assert result["status"] == "error"
    assert "validation error for test_tooltool\nrequired\n" in result["content"][0]["text"].lower(), (
        "Validation error should indicate which argument is missing"
    )

    # Test with exception in tool function
    tool_use = {"toolUseId": "test-id", "input": {"required": "error"}}

    result = test_tool(tool_use)
    assert result["status"] == "error"
    assert "test error" in result["content"][0]["text"].lower(), (
        "Runtime error should contain the original error message"
    )


def test_type_handling():
    """Test handling of basic parameter types."""

    @tool
    def test_tool(
        str_param: str,
        int_param: int,
        float_param: float,
        bool_param: bool,
    ) -> str:
        """Test basic types."""
        return f"Successfully used parameters: {str_param}, {int_param}, {float_param}, {bool_param}"

    spec = test_tool.TOOL_SPEC
    schema = spec["inputSchema"]["json"]
    props = schema["properties"]

    assert props["str_param"]["type"] == "string"
    assert props["int_param"]["type"] == "integer"
    assert props["float_param"]["type"] == "number"
    assert props["bool_param"]["type"] == "boolean"


def test_agent_parameter_passing():
    """Test passing agent parameter to tool function."""
    mock_agent = MagicMock()

    @tool
    def test_tool(param: str, agent=None) -> str:
        """Test tool with agent parameter."""
        if agent:
            return f"Agent: {agent}, Param: {param}"
        return f"Param: {param}"

    tool_use = {"toolUseId": "test-id", "input": {"param": "test"}}

    # Test without agent
    result = test_tool(tool_use)
    assert result["content"][0]["text"] == "Param: test"

    # Test with agent
    result = test_tool(tool_use, agent=mock_agent)
    assert "Agent:" in result["content"][0]["text"]
    assert "test" in result["content"][0]["text"]


def test_tool_decorator_with_different_return_values():
    """Test tool decorator with different return value types."""

    # Test with dict return that follows ToolResult format
    @tool
    def dict_return_tool(param: str) -> dict:
        """Test tool that returns a dict in ToolResult format."""
        return {"status": "success", "content": [{"text": f"Result: {param}"}]}

    # Test with non-dict return
    @tool
    def string_return_tool(param: str) -> str:
        """Test tool that returns a string."""
        return f"Result: {param}"

    # Test with None return
    @tool
    def none_return_tool(param: str) -> None:
        """Test tool that returns None."""
        pass

    # Test the dict return - should preserve dict format but add toolUseId
    tool_use = {"toolUseId": "test-id", "input": {"param": "test"}}
    result = dict_return_tool(tool_use)
    assert result["status"] == "success"
    assert result["content"][0]["text"] == "Result: test"
    assert result["toolUseId"] == "test-id"

    # Test the string return - should wrap in standard format
    result = string_return_tool(tool_use)
    assert result["status"] == "success"
    assert result["content"][0]["text"] == "Result: test"

    # Test None return - should still create valid ToolResult with "None" text
    result = none_return_tool(tool_use)
    assert result["status"] == "success"
    assert result["content"][0]["text"] == "None"


def test_class_method_handling():
    """Test handling of class methods with tool decorator."""

    class TestClass:
        def __init__(self, prefix):
            self.prefix = prefix

        @tool
        def test_method(self, param: str) -> str:
            """Test method.

            Args:
                param: Test parameter
            """
            return f"{self.prefix}: {param}"

    # Create instance and test the method
    instance = TestClass("Test")

    # Check that tool spec exists and doesn't include self
    assert hasattr(instance.test_method, "TOOL_SPEC")
    spec = instance.test_method.TOOL_SPEC
    assert "param" in spec["inputSchema"]["json"]["properties"]
    assert "self" not in spec["inputSchema"]["json"]["properties"]

    # Test regular method call
    result = instance.test_method("value")
    assert result == "Test: value"

    # Test tool-style call
    tool_use = {"toolUseId": "test-id", "input": {"param": "tool-value"}}
    result = instance.test_method(tool_use)
    assert "Test: tool-value" in result["content"][0]["text"]


def test_default_parameter_handling():
    """Test handling of parameters with default values."""

    @tool
    def tool_with_defaults(required: str, optional: str = "default", number: int = 42) -> str:
        """Test tool with multiple default parameters.

        Args:
            required: Required parameter
            optional: Optional with default
            number: Number with default
        """
        return f"{required} {optional} {number}"

    # Check schema has correct required fields
    spec = tool_with_defaults.TOOL_SPEC
    schema = spec["inputSchema"]["json"]
    assert "required" in schema["required"]
    assert "optional" not in schema["required"]
    assert "number" not in schema["required"]

    # Call with just required parameter
    tool_use = {"toolUseId": "test-id", "input": {"required": "hello"}}
    result = tool_with_defaults(tool_use)
    assert result["content"][0]["text"] == "hello default 42"

    # Call with some but not all optional parameters
    tool_use = {"toolUseId": "test-id", "input": {"required": "hello", "number": 100}}
    result = tool_with_defaults(tool_use)
    assert result["content"][0]["text"] == "hello default 100"


def test_empty_tool_use_handling():
    """Test handling of empty tool use dictionaries."""

    @tool
    def test_tool(required: str) -> str:
        """Test with a required parameter."""
        return f"Got: {required}"

    # Test with completely empty tool use
    result = test_tool({})
    assert result["status"] == "error"
    assert "unknown" in result["toolUseId"]

    # Test with missing input
    result = test_tool({"toolUseId": "test-id"})
    assert result["status"] == "error"
    assert "test-id" in result["toolUseId"]


def test_traditional_function_call():
    """Test that decorated functions can still be called normally."""

    @tool
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers.

        Args:
            a: First number
            b: Second number
        """
        return a + b

    # Call the function directly
    result = add_numbers(5, 7)
    assert result == 12

    # Call through tool interface
    tool_use = {"toolUseId": "test-id", "input": {"a": 2, "b": 3}}
    result = add_numbers(tool_use)
    assert result["status"] == "success"
    assert result["content"][0]["text"] == "5"


def test_multiple_default_parameters():
    """Test handling of multiple parameters with default values."""

    @tool
    def multi_default_tool(
        required_param: str,
        optional_str: str = "default_str",
        optional_int: int = 42,
        optional_bool: bool = True,
        optional_float: float = 3.14,
    ) -> str:
        """Tool with multiple default parameters of different types."""
        return f"{required_param}, {optional_str}, {optional_int}, {optional_bool}, {optional_float}"

    # Check the tool spec
    spec = multi_default_tool.TOOL_SPEC
    schema = spec["inputSchema"]["json"]

    # Verify that only required_param is in the required list
    assert len(schema["required"]) == 1
    assert "required_param" in schema["required"]
    assert "optional_str" not in schema["required"]
    assert "optional_int" not in schema["required"]
    assert "optional_bool" not in schema["required"]
    assert "optional_float" not in schema["required"]

    # Test calling with only required parameter
    tool_use = {"toolUseId": "test-id", "input": {"required_param": "hello"}}
    result = multi_default_tool(tool_use)
    assert result["status"] == "success"
    assert "hello, default_str, 42, True, 3.14" in result["content"][0]["text"]

    # Test calling with some optional parameters
    tool_use = {
        "toolUseId": "test-id",
        "input": {"required_param": "hello", "optional_int": 100, "optional_float": 2.718},
    }
    result = multi_default_tool(tool_use)
    assert "hello, default_str, 100, True, 2.718" in result["content"][0]["text"]


def test_return_type_validation():
    """Test that return types are properly handled and validated."""

    # Define tool with explicitly typed return
    @tool
    def int_return_tool(param: str) -> int:
        """Tool that returns an integer.

        Args:
            param: Input parameter
        """
        if param == "valid":
            return 42
        elif param == "invalid_type":
            return "not an int"  # This should work because Python is dynamically typed
        else:
            return None  # This should work but be wrapped correctly

    # Test with return that matches declared type
    tool_use = {"toolUseId": "test-id", "input": {"param": "valid"}}
    result = int_return_tool(tool_use)
    assert result["status"] == "success"
    assert result["content"][0]["text"] == "42"

    # Test with return that doesn't match declared type
    # Note: This should still work because Python doesn't enforce return types at runtime
    # but the function will return a string instead of an int
    tool_use = {"toolUseId": "test-id", "input": {"param": "invalid_type"}}
    result = int_return_tool(tool_use)
    assert result["status"] == "success"
    assert result["content"][0]["text"] == "not an int"

    # Test with None return from a non-None return type
    tool_use = {"toolUseId": "test-id", "input": {"param": "none"}}
    result = int_return_tool(tool_use)
    assert result["status"] == "success"
    assert result["content"][0]["text"] == "None"

    # Define tool with Union return type
    @tool
    def union_return_tool(param: str) -> Union[Dict[str, Any], str, None]:
        """Tool with Union return type.

        Args:
            param: Input parameter
        """
        if param == "dict":
            return {"key": "value"}
        elif param == "str":
            return "string result"
        else:
            return None

    # Test with each possible return type in the Union
    tool_use = {"toolUseId": "test-id", "input": {"param": "dict"}}
    result = union_return_tool(tool_use)
    assert result["status"] == "success"
    assert "{'key': 'value'}" in result["content"][0]["text"] or '{"key": "value"}' in result["content"][0]["text"]

    tool_use = {"toolUseId": "test-id", "input": {"param": "str"}}
    result = union_return_tool(tool_use)
    assert result["status"] == "success"
    assert result["content"][0]["text"] == "string result"

    tool_use = {"toolUseId": "test-id", "input": {"param": "none"}}
    result = union_return_tool(tool_use)
    assert result["status"] == "success"
    assert result["content"][0]["text"] == "None"


def test_tool_with_no_parameters():
    """Test a tool that doesn't require any parameters."""

    @tool
    def no_params_tool() -> str:
        """A tool that doesn't need any parameters."""
        return "Success - no parameters needed"

    # Check schema is still valid even with no parameters
    spec = no_params_tool.TOOL_SPEC
    schema = spec["inputSchema"]["json"]
    assert schema["type"] == "object"
    assert "properties" in schema

    # Test tool use call
    tool_use = {"toolUseId": "test-id", "input": {}}
    result = no_params_tool(tool_use)
    assert result["status"] == "success"
    assert result["content"][0]["text"] == "Success - no parameters needed"

    # Test direct call
    direct_result = no_params_tool()
    assert direct_result == "Success - no parameters needed"


def test_complex_parameter_types():
    """Test handling of complex parameter types like nested dictionaries."""

    @tool
    def complex_type_tool(config: Dict[str, Any]) -> str:
        """Tool with complex parameter type.

        Args:
            config: A complex configuration object
        """
        return f"Got config with {len(config.keys())} keys"

    # Test with a nested dictionary
    nested_dict = {"name": "test", "settings": {"enabled": True, "threshold": 0.5}, "tags": ["important", "test"]}

    # Call via tool use
    tool_use = {"toolUseId": "test-id", "input": {"config": nested_dict}}
    result = complex_type_tool(tool_use)
    assert result["status"] == "success"
    assert "Got config with 3 keys" in result["content"][0]["text"]

    # Direct call
    direct_result = complex_type_tool(nested_dict)
    assert direct_result == "Got config with 3 keys"


def test_custom_tool_result_handling():
    """Test that a function returning a properly formatted tool result dictionary is handled correctly."""

    @tool
    def custom_result_tool(param: str) -> Dict[str, Any]:
        """Tool that returns a custom tool result dictionary.

        Args:
            param: Input parameter
        """
        # Return a dictionary that follows the tool result format including multiple content items
        return {
            "status": "success",
            "content": [{"text": f"First line: {param}"}, {"text": "Second line", "type": "markdown"}],
        }

    # Test via tool use
    tool_use = {"toolUseId": "custom-id", "input": {"param": "test"}}
    result = custom_result_tool(tool_use)

    # The wrapper should preserve our format and just add the toolUseId
    assert result["status"] == "success"
    assert result["toolUseId"] == "custom-id"
    assert len(result["content"]) == 2
    assert result["content"][0]["text"] == "First line: test"
    assert result["content"][1]["text"] == "Second line"
    assert result["content"][1]["type"] == "markdown"


def test_docstring_parsing():
    """Test that function docstring is correctly parsed into tool spec."""

    @tool
    def documented_tool(param1: str, param2: int = 10) -> str:
        """This is the summary line.

        This is a more detailed description that spans
        multiple lines and provides additional context.

        Args:
            param1: Description of first parameter with details
                   that continue on next line
            param2: Description of second parameter (default: 10)
                    with additional info

        Returns:
            A string with the result

        Raises:
            ValueError: If parameters are invalid
        """
        return f"{param1} {param2}"

    spec = documented_tool.TOOL_SPEC

    # Check description captures both summary and details
    assert "This is the summary line" in spec["description"]
    assert "more detailed description" in spec["description"]

    # Check parameter descriptions
    schema = spec["inputSchema"]["json"]
    assert "Description of first parameter" in schema["properties"]["param1"]["description"]
    assert "Description of second parameter" in schema["properties"]["param2"]["description"]

    # Check that default value notes from docstring don't override actual defaults
    assert "param2" not in schema["required"]


def test_detailed_validation_errors():
    """Test detailed error messages for various validation failures."""

    @tool
    def validation_tool(str_param: str, int_param: int, bool_param: bool) -> str:
        """Tool with various parameter types for validation testing.

        Args:
            str_param: String parameter
            int_param: Integer parameter
            bool_param: Boolean parameter
        """
        return f"Valid parameters: String argument{str_param}, Int arguemnt {int_param}, Bool argument {bool_param}"

    # Test wrong type for int
    tool_use = {
        "toolUseId": "test-id",
        "input": {
            "str_param": "hello",
            "int_param": "not an int",  # Wrong type
            "bool_param": True,
        },
    }
    result = validation_tool(tool_use)
    assert result["status"] == "error"
    assert "int_param" in result["content"][0]["text"]

    # Test missing required parameter
    tool_use = {
        "toolUseId": "test-id",
        "input": {
            "str_param": "hello",
            # int_param missing
            "bool_param": True,
        },
    }
    result = validation_tool(tool_use)
    assert result["status"] == "error"
    assert "int_param" in result["content"][0]["text"]


def test_tool_complex_validation_edge_cases():
    """Test validation of complex schema edge cases."""
    from typing import Any, Dict, Union

    # Define a tool with a complex anyOf type that could trigger edge case handling
    @tool
    def edge_case_tool(param: Union[Dict[str, Any], None]) -> str:
        """Tool with complex anyOf structure.

        Args:
            param: A complex parameter that can be None or a dict
        """
        return str(param)

    # Test with None value
    tool_use = {"toolUseId": "test-id", "input": {"param": None}}
    result = edge_case_tool(tool_use)
    assert result["status"] == "success"
    assert result["content"][0]["text"] == "None"

    # Test with empty dict
    tool_use = {"toolUseId": "test-id", "input": {"param": {}}}
    result = edge_case_tool(tool_use)
    assert result["status"] == "success"
    assert result["content"][0]["text"] == "{}"

    # Test with a complex nested dictionary
    nested_dict = {"key1": {"nested": [1, 2, 3]}, "key2": None}
    tool_use = {"toolUseId": "test-id", "input": {"param": nested_dict}}
    result = edge_case_tool(tool_use)
    assert result["status"] == "success"
    assert "key1" in result["content"][0]["text"]
    assert "nested" in result["content"][0]["text"]


def test_tool_method_detection_errors():
    """Test edge cases in method detection logic."""

    # Define a class with a decorated method to test exception handling in method detection
    class TestClass:
        @tool
        def test_method(self, param: str) -> str:
            """Test method that should be called properly despite errors.

            Args:
                param: A test parameter
            """
            return f"Method Got: {param}"

    # Create a mock instance where attribute access will raise exceptions
    class MockInstance:
        @property
        def __class__(self):
            # First access will raise AttributeError to test that branch
            raise AttributeError("Simulated AttributeError")

    class MockInstance2:
        @property
        def __class__(self):
            class MockClass:
                @property
                def test_method(self):
                    # This will raise TypeError when checking for the method name
                    raise TypeError("Simulated TypeError")

            return MockClass()

    # Create instances
    instance = TestClass()
    MockInstance()
    MockInstance2()

    # Test normal method call
    assert instance.test_method("test") == "Method Got: test"

    # Test direct function call
    direct_result = instance.test_method({"toolUseId": "test-id", "input": {"param": "direct"}})
    assert direct_result["status"] == "success"
    assert direct_result["content"][0]["text"] == "Method Got: direct"

    # Create a standalone function to test regular function calls
    @tool
    def standalone_tool(p1: str, p2: str = "default") -> str:
        """Standalone tool for testing.

        Args:
            p1: First parameter
            p2: Second parameter with default
        """
        return f"Standalone: {p1}, {p2}"

    # Test that we can call it directly with multiple parameters
    result = standalone_tool("param1", "param2")
    assert result == "Standalone: param1, param2"

    # And that it works with tool use call too
    tool_use_result = standalone_tool({"toolUseId": "test-id", "input": {"p1": "value1"}})
    assert tool_use_result["status"] == "success"
    assert tool_use_result["content"][0]["text"] == "Standalone: value1, default"


def test_tool_general_exception_handling():
    """Test handling of arbitrary exceptions in tool execution."""

    @tool
    def failing_tool(param: str) -> str:
        """Tool that raises different exception types.

        Args:
            param: Determines which exception to raise
        """
        if param == "value_error":
            raise ValueError("Value error message")
        elif param == "type_error":
            raise TypeError("Type error message")
        elif param == "attribute_error":
            raise AttributeError("Attribute error message")
        elif param == "key_error":
            raise KeyError("key_name")
        return "Success"

    # Test with different error types
    error_types = ["value_error", "type_error", "attribute_error", "key_error"]
    for error_type in error_types:
        tool_use = {"toolUseId": "test-id", "input": {"param": error_type}}
        result = failing_tool(tool_use)
        assert result["status"] == "error"

        error_message = result["content"][0]["text"]

        # Check that error type is included
        if error_type == "value_error":
            assert "Value error message" in error_message
        elif error_type == "type_error":
            assert "TypeError" in error_message
        elif error_type == "attribute_error":
            assert "AttributeError" in error_message
        elif error_type == "key_error":
            assert "KeyError" in error_message
            assert "key_name" in error_message


def test_tool_with_complex_anyof_schema():
    """Test handling of complex anyOf structures in the schema."""
    from typing import Any, Dict, List, Union

    @tool
    def complex_schema_tool(union_param: Union[List[int], Dict[str, Any], str, None]) -> str:
        """Tool with a complex Union type that creates anyOf in schema.

        Args:
            union_param: A parameter that can be list, dict, string or None
        """
        return str(type(union_param).__name__) + ": " + str(union_param)

    # Test with a list
    tool_use = {"toolUseId": "test-id", "input": {"union_param": [1, 2, 3]}}
    result = complex_schema_tool(tool_use)
    assert result["status"] == "success"
    assert "list: [1, 2, 3]" in result["content"][0]["text"]

    # Test with a dict
    tool_use = {"toolUseId": "test-id", "input": {"union_param": {"key": "value"}}}
    result = complex_schema_tool(tool_use)
    assert result["status"] == "success"
    assert "dict:" in result["content"][0]["text"]
    assert "key" in result["content"][0]["text"]

    # Test with a string
    tool_use = {"toolUseId": "test-id", "input": {"union_param": "test_string"}}
    result = complex_schema_tool(tool_use)
    assert result["status"] == "success"
    assert "str: test_string" in result["content"][0]["text"]

    # Test with None
    tool_use = {"toolUseId": "test-id", "input": {"union_param": None}}
    result = complex_schema_tool(tool_use)
    assert result["status"] == "success"
    assert "NoneType: None" in result["content"][0]["text"]
