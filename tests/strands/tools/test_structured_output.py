from typing import Literal, Optional

from pydantic import BaseModel, Field

from strands.tools.structured_output import convert_pydantic_to_tool_spec
from strands.types.tools import ToolSpec


def test_convert_pydantic_to_tool_spec_basic():
    class User(BaseModel):
        """A user of the system."""

        name: str = Field(description="The name of the user")
        age: int = Field(description="The age of the user")
        email: str = Field(description="The email of the user", default="")

    basic_user_tool_spec = {
        "name": "User",
        "description": "A user of the system.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "name": {"description": "The name of the user", "title": "Name", "type": "string"},
                    "age": {"description": "The age of the user", "title": "Age", "type": "integer"},
                    "email": {
                        "default": "",
                        "description": "The email of the user",
                        "title": "Email",
                        "type": ["string", "null"],
                    },
                },
                "title": "User",
                "description": "A user of the system.",
                "required": ["name", "age"],
            }
        },
    }
    tool_spec = convert_pydantic_to_tool_spec(User)

    assert tool_spec == basic_user_tool_spec
    assert ToolSpec(**tool_spec) == ToolSpec(**basic_user_tool_spec)


def test_convert_pydantic_to_tool_spec_complex():
    class User(BaseModel):
        """A user of the system."""

        name: str = Field(description="The name of the user")
        age: int = Field(description="The age of the user")
        email: str = Field(description="The email of the user", default="")

    class Employment(BaseModel):
        """An employment of the user."""

        company: str = Field(description="The company of the user")
        title: Literal[
            "CEO",
            "CTO",
            "CFO",
            "CMO",
            "COO",
            "VP",
            "Director",
            "Manager",
            "Other",
        ] = Field(description="The title of the user", default="Other")

    class UserWithEmployment(User):
        """A user of the system with employment."""

        employment: Employment = Field(description="The employment of the user")
        employment_2: Optional[Employment] = Field(description="A part time employment of the user")

    tool_spec = convert_pydantic_to_tool_spec(UserWithEmployment)
    complex_user_tool_spec_json = {
        "name": "UserWithEmployment",
        "description": "A user of the system with employment.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "name": {"description": "The name of the user", "title": "Name", "type": "string"},
                    "age": {"description": "The age of the user", "title": "Age", "type": "integer"},
                    "email": {
                        "default": "",
                        "description": "The email of the user",
                        "title": "Email",
                        "type": ["string", "null"],
                    },
                    "employment": {
                        "type": "object",
                        "description": "The employment of the user",
                        "properties": {
                            "company": {"description": "The company of the user", "title": "Company", "type": "string"},
                            "title": {
                                "default": "Other",
                                "description": "The title of the user",
                                "enum": ["CEO", "CTO", "CFO", "CMO", "COO", "VP", "Director", "Manager", "Other"],
                                "title": "Title",
                                "type": "string",
                            },
                        },
                        "required": ["company"],
                    },
                    "employment_2": {
                        "type": ["object", "null"],
                        "description": "A part time employment of the user",
                        "properties": {
                            "company": {"description": "The company of the user", "title": "Company", "type": "string"},
                            "title": {
                                "default": "Other",
                                "description": "The title of the user",
                                "enum": ["CEO", "CTO", "CFO", "CMO", "COO", "VP", "Director", "Manager", "Other"],
                                "title": "Title",
                                "type": "string",
                            },
                        },
                        "required": ["company"],
                    },
                },
                "title": "UserWithEmployment",
                "description": "A user of the system with employment.",
                "required": ["name", "age", "employment"],
            }
        },
    }

    assert tool_spec == complex_user_tool_spec_json
    assert ToolSpec(**tool_spec) == ToolSpec(**complex_user_tool_spec_json)
