from typing import Literal, Optional

import pytest
from pydantic import BaseModel, Field

from strands.tools.structured_output import convert_pydantic_to_tool_spec
from strands.types.tools import ToolSpec


# Basic test model
class User(BaseModel):
    """User model with name and age."""

    name: str = Field(description="The name of the user")
    age: int = Field(description="The age of the user", ge=18, le=100)


# Test model with inheritance and literals
class UserWithPlanet(User):
    """User with planet."""

    planet: Literal["Earth", "Mars"] = Field(description="The planet")


# Test model with multiple same type fields and optional field
class TwoUsersWithPlanet(BaseModel):
    """Two users model with planet."""

    user1: UserWithPlanet = Field(description="The first user")
    user2: Optional[UserWithPlanet] = Field(description="The second user", default=None)


# Test model with list of same type fields
class ListOfUsersWithPlanet(BaseModel):
    """List of users model with planet."""

    users: list[UserWithPlanet] = Field(description="The users", min_length=2, max_length=3)


def test_convert_pydantic_to_tool_spec_basic():
    tool_spec = convert_pydantic_to_tool_spec(User)

    # Check basic structure
    assert tool_spec["name"] == "User"
    assert tool_spec["description"] == "User model with name and age."
    assert "inputSchema" in tool_spec
    assert "json" in tool_spec["inputSchema"]

    # Check schema properties
    schema = tool_spec["inputSchema"]["json"]
    assert schema["type"] == "object"
    assert schema["title"] == "User"

    # Check field properties
    assert "name" in schema["properties"]
    assert schema["properties"]["name"]["description"] == "The name of the user"
    assert schema["properties"]["name"]["type"] == "string"

    assert "age" in schema["properties"]
    assert schema["properties"]["age"]["description"] == "The age of the user"
    assert schema["properties"]["age"]["type"] == "integer"

    # Check required fields
    assert "required" in schema
    assert "name" in schema["required"]
    assert "age" in schema["required"]

    # check validation
    assert schema["properties"]["age"]["minimum"] == 18
    assert schema["properties"]["age"]["maximum"] == 100

    # Verify we can construct a valid ToolSpec
    tool_spec_obj = ToolSpec(**tool_spec)
    assert tool_spec_obj is not None


def test_convert_pydantic_to_tool_spec_complex():
    tool_spec = convert_pydantic_to_tool_spec(ListOfUsersWithPlanet)

    # Assert expected properties are present in the tool spec
    assert tool_spec["name"] == "ListOfUsersWithPlanet"
    assert tool_spec["description"] == "List of users model with planet."
    assert "inputSchema" in tool_spec
    assert "json" in tool_spec["inputSchema"]

    # Check the schema properties
    schema = tool_spec["inputSchema"]["json"]
    assert schema["type"] == "object"
    assert "users" in schema["properties"]
    assert schema["properties"]["users"]["type"] == "array"
    assert schema["properties"]["users"]["items"]["type"] == "object"
    assert schema["properties"]["users"]["items"]["properties"]["name"]["type"] == "string"
    assert schema["properties"]["users"]["items"]["properties"]["age"]["type"] == "integer"
    assert schema["properties"]["users"]["items"]["properties"]["planet"]["type"] == "string"
    assert schema["properties"]["users"]["items"]["properties"]["planet"]["enum"] == ["Earth", "Mars"]

    # Check the list field properties
    assert schema["properties"]["users"]["minItems"] == 2
    assert schema["properties"]["users"]["maxItems"] == 3

    # Verify the required fields
    assert "required" in schema
    assert "users" in schema["required"]

    # Verify we can construct a valid ToolSpec
    tool_spec_obj = ToolSpec(**tool_spec)
    assert tool_spec_obj is not None


def test_convert_pydantic_to_tool_spec_multiple_same_type():
    tool_spec = convert_pydantic_to_tool_spec(TwoUsersWithPlanet)

    # Verify the schema structure
    assert tool_spec["name"] == "TwoUsersWithPlanet"
    assert "user1" in tool_spec["inputSchema"]["json"]["properties"]
    assert "user2" in tool_spec["inputSchema"]["json"]["properties"]

    # Verify both employment fields have the same structure
    primary = tool_spec["inputSchema"]["json"]["properties"]["user1"]
    secondary = tool_spec["inputSchema"]["json"]["properties"]["user2"]

    assert primary["type"] == "object"
    assert secondary["type"] == ["object", "null"]

    assert "properties" in primary
    assert "name" in primary["properties"]
    assert "age" in primary["properties"]
    assert "planet" in primary["properties"]

    assert "properties" in secondary
    assert "name" in secondary["properties"]
    assert "age" in secondary["properties"]
    assert "planet" in secondary["properties"]


def test_convert_pydantic_with_missing_refs():
    """Test that the tool handles missing $refs gracefully."""
    # This test checks that our error handling for missing $refs works correctly
    # by testing with a model that has circular references

    class NodeWithCircularRef(BaseModel):
        """A node with a circular reference to itself."""

        name: str = Field(description="The name of the node")
        parent: Optional["NodeWithCircularRef"] = Field(None, description="Parent node")
        children: list["NodeWithCircularRef"] = Field(default_factory=list, description="Child nodes")

    # This forward reference normally causes issues with schema generation
    # but our error handling should prevent errors
    with pytest.raises(ValueError, match="Circular reference detected and not supported"):
        convert_pydantic_to_tool_spec(NodeWithCircularRef)


def test_convert_pydantic_with_custom_description():
    """Test that custom descriptions override model docstrings."""

    # Test with custom description
    custom_description = "Custom tool description for user model"
    tool_spec = convert_pydantic_to_tool_spec(User, description=custom_description)

    assert tool_spec["description"] == custom_description


def test_convert_pydantic_with_empty_docstring():
    """Test that empty docstrings use default description."""

    class EmptyDocUser(BaseModel):
        name: str = Field(description="The name of the user")

    tool_spec = convert_pydantic_to_tool_spec(EmptyDocUser)
    assert tool_spec["description"] == "EmptyDocUser structured output tool"
