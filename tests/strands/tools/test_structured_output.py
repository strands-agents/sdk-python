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

    expected_spec = {
        "name": "User",
        "description": "User model with name and age.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "name": {"description": "The name of the user", "title": "Name", "type": "string"},
                    "age": {
                        "description": "The age of the user",
                        "maximum": 100,
                        "minimum": 18,
                        "title": "Age",
                        "type": "integer",
                    },
                },
                "title": "User",
                "description": "User model with name and age.",
                "required": ["name", "age"],
            }
        },
    }

    # Verify we can construct a valid ToolSpec
    tool_spec_obj = ToolSpec(**tool_spec)
    assert tool_spec_obj is not None
    assert tool_spec == expected_spec


def test_convert_pydantic_to_tool_spec_complex():
    tool_spec = convert_pydantic_to_tool_spec(ListOfUsersWithPlanet)

    expected_spec = {
        "name": "ListOfUsersWithPlanet",
        "description": "List of users model with planet.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "users": {
                        "description": "The users",
                        "items": {
                            "description": "User with planet.",
                            "title": "UserWithPlanet",
                            "type": "object",
                            "properties": {
                                "name": {"description": "The name of the user", "title": "Name", "type": "string"},
                                "age": {
                                    "description": "The age of the user",
                                    "maximum": 100,
                                    "minimum": 18,
                                    "title": "Age",
                                    "type": "integer",
                                },
                                "planet": {
                                    "description": "The planet",
                                    "enum": ["Earth", "Mars"],
                                    "title": "Planet",
                                    "type": "string",
                                },
                            },
                            "required": ["name", "age", "planet"],
                        },
                        "maxItems": 3,
                        "minItems": 2,
                        "title": "Users",
                        "type": "array",
                    }
                },
                "title": "ListOfUsersWithPlanet",
                "description": "List of users model with planet.",
                "required": ["users"],
            }
        },
    }

    assert tool_spec == expected_spec

    # Verify we can construct a valid ToolSpec
    tool_spec_obj = ToolSpec(**tool_spec)
    assert tool_spec_obj is not None


def test_convert_pydantic_to_tool_spec_multiple_same_type():
    tool_spec = convert_pydantic_to_tool_spec(TwoUsersWithPlanet)

    expected_spec = {
        "name": "TwoUsersWithPlanet",
        "description": "Two users model with planet.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "user1": {
                        "type": "object",
                        "description": "The first user",
                        "properties": {
                            "name": {"description": "The name of the user", "title": "Name", "type": "string"},
                            "age": {
                                "description": "The age of the user",
                                "maximum": 100,
                                "minimum": 18,
                                "title": "Age",
                                "type": "integer",
                            },
                            "planet": {
                                "description": "The planet",
                                "enum": ["Earth", "Mars"],
                                "title": "Planet",
                                "type": "string",
                            },
                        },
                        "required": ["name", "age", "planet"],
                    },
                    "user2": {
                        "type": ["object", "null"],
                        "description": "The second user",
                        "properties": {
                            "name": {"description": "The name of the user", "title": "Name", "type": "string"},
                            "age": {
                                "description": "The age of the user",
                                "maximum": 100,
                                "minimum": 18,
                                "title": "Age",
                                "type": "integer",
                            },
                            "planet": {
                                "description": "The planet",
                                "enum": ["Earth", "Mars"],
                                "title": "Planet",
                                "type": "string",
                            },
                        },
                        "required": ["name", "age", "planet"],
                    },
                },
                "title": "TwoUsersWithPlanet",
                "description": "Two users model with planet.",
                "required": ["user1"],
            }
        },
    }

    assert tool_spec == expected_spec

    # Verify we can construct a valid ToolSpec
    tool_spec_obj = ToolSpec(**tool_spec)
    assert tool_spec_obj is not None


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


def test_convert_pydantic_with_items_refs():
    """Test that no $refs exist after lists of different components."""

    class Address(BaseModel):
        postal_code: Optional[str] = None

    class Person(BaseModel):
        """Complete person information."""

        list_of_items: list[Address]
        list_of_items_nullable: Optional[list[Address]]
        list_of_item_or_nullable: list[Optional[Address]]

    tool_spec = convert_pydantic_to_tool_spec(Person)

    expected_spec = {
        "description": "Complete person information.",
        "inputSchema": {
            "json": {
                "description": "Complete person information.",
                "properties": {
                    "list_of_item_or_nullable": {
                        "items": {
                            "anyOf": [
                                {
                                    "properties": {"postal_code": {"type": ["string", "null"]}},
                                    "title": "Address",
                                    "type": "object",
                                },
                                {"type": "null"},
                            ]
                        },
                        "title": "List Of Item Or Nullable",
                        "type": "array",
                    },
                    "list_of_items": {
                        "items": {
                            "properties": {"postal_code": {"type": ["string", "null"]}},
                            "title": "Address",
                            "type": "object",
                        },
                        "title": "List Of Items",
                        "type": "array",
                    },
                    "list_of_items_nullable": {
                        "items": {
                            "properties": {"postal_code": {"type": ["string", "null"]}},
                            "title": "Address",
                            "type": "object",
                        },
                        "type": ["array", "null"],
                    },
                },
                "required": ["list_of_items", "list_of_item_or_nullable"],
                "title": "Person",
                "type": "object",
            }
        },
        "name": "Person",
    }
    assert tool_spec == expected_spec


def test_convert_pydantic_with_refs():
    """Test that no $refs exist after processing complex hierarchies."""

    class Address(BaseModel):
        street: str
        city: str
        country: str
        postal_code: Optional[str] = None

    class Contact(BaseModel):
        address: Address

    class Person(BaseModel):
        """Complete person information."""

        contact: Contact = Field(description="Contact methods")

    tool_spec = convert_pydantic_to_tool_spec(Person)

    expected_spec = {
        "description": "Complete person information.",
        "inputSchema": {
            "json": {
                "description": "Complete person information.",
                "properties": {
                    "contact": {
                        "description": "Contact methods",
                        "properties": {
                            "address": {
                                "properties": {
                                    "city": {"title": "City", "type": "string"},
                                    "country": {"title": "Country", "type": "string"},
                                    "postal_code": {"type": ["string", "null"]},
                                    "street": {"title": "Street", "type": "string"},
                                },
                                "required": ["street", "city", "country"],
                                "title": "Address",
                                "type": "object",
                            }
                        },
                        "required": ["address"],
                        "type": "object",
                    }
                },
                "required": ["contact"],
                "title": "Person",
                "type": "object",
            }
        },
        "name": "Person",
    }
    assert tool_spec == expected_spec
