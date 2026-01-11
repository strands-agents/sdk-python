"""Test for Pydantic 2.12+ compatibility with Literal types and __future__ annotations.

This test ensures that tools using typing.Literal with __future__ annotations
work correctly with Pydantic 2.12+, which changed how string annotations are handled.

Issue: https://github.com/strands-agents/sdk-python/issues/1208
"""

from __future__ import annotations

from typing import Literal

from strands import tool


def test_tool_with_literal_type_from_future_annotations():
    """Test that @tool works with Literal types when using __future__ annotations.
    
    With __future__ annotations enabled, all annotations become strings at runtime.
    Before the fix, param.annotation would return "Literal['option_a', 'option_b']" as a string,
    which Pydantic 2.12+ cannot resolve. The fix uses self.type_hints which properly resolves
    these string annotations to actual Literal types.
    """

    @tool
    def test_literal_tool(option: Literal["option_a", "option_b"]) -> dict:
        """A test tool using Literal types.
        
        Args:
            option: Must be either 'option_a' or 'option_b'
        """
        return {"status": "success", "content": [{"text": f"Selected: {option}"}]}

    # Test that the tool was created successfully
    assert test_literal_tool.tool_name == "test_literal_tool"
    assert test_literal_tool.tool_spec is not None
    assert test_literal_tool.tool_spec["description"] is not None
    
    # Test that the tool spec includes the enum constraint for the Literal type
    tool_spec = test_literal_tool.tool_spec
    assert "inputSchema" in tool_spec
    input_schema = tool_spec["inputSchema"]["json"]
    assert "properties" in input_schema
    assert "option" in input_schema["properties"]
    
    # The Literal type should be converted to an enum in the JSON schema
    option_schema = input_schema["properties"]["option"]
    assert "enum" in option_schema
    assert set(option_schema["enum"]) == {"option_a", "option_b"}


def test_tool_with_multiple_literal_params():
    """Test @tool with multiple parameters including Literal types."""

    @tool
    def multi_param_tool(
        action: Literal["create", "update", "delete"],
        target: Literal["file", "directory"],
        name: str,
    ) -> dict:
        """Perform an action on a target.
        
        Args:
            action: The action to perform
            target: The type of target
            name: The name of the target
        """
        return {"status": "success", "content": [{"text": f"{action} {target} {name}"}]}
    
    # Verify tool creation
    assert multi_param_tool.tool_name == "multi_param_tool"
    
    # Verify the input schema includes enums for both Literal types
    tool_spec = multi_param_tool.tool_spec
    input_schema = tool_spec["inputSchema"]["json"]
    
    # Check action parameter
    assert "action" in input_schema["properties"]
    action_schema = input_schema["properties"]["action"]
    assert "enum" in action_schema
    assert set(action_schema["enum"]) == {"create", "update", "delete"}
    
    # Check target parameter
    assert "target" in input_schema["properties"]
    target_schema = input_schema["properties"]["target"]
    assert "enum" in target_schema
    assert set(target_schema["enum"]) == {"file", "directory"}
    
    # Check name parameter (should not have enum)
    assert "name" in input_schema["properties"]
    name_schema = input_schema["properties"]["name"]
    assert "enum" not in name_schema
    assert name_schema["type"] == "string"


def test_tool_with_literal_invocation():
    """Test that we can actually invoke a tool with Literal types."""

    @tool
    def greet_tool(greeting: Literal["hello", "hi", "hey"], name: str) -> dict:
        """Greet someone.
        
        Args:
            greeting: The greeting to use
            name: The person's name
        """
        return {"status": "success", "content": [{"text": f"{greeting}, {name}!"}]}
    
    # Test invocation with valid input using the tool_func directly
    result = greet_tool._tool_func(greeting="hello", name="World")
    assert result["status"] == "success"
    assert "hello, World!" in result["content"][0]["text"]
    
    # Test with different valid greeting
    result2 = greet_tool._tool_func(greeting="hi", name="There")
    assert result2["status"] == "success"
    assert "hi, There!" in result2["content"][0]["text"]
