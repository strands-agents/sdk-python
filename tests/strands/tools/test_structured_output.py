import pytest
from pydantic import BaseModel, Field

from strands.tools.structured_output import convert_pydantic_to_bedrock_tool


class User(BaseModel):
    """A user of the system."""

    name: str = Field(description="The name of the user")
    age: int = Field(description="The age of the user")
    email: str = Field(description="The email of the user", default="")


@pytest.fixture
def user_model():
    return User


def test_convert_pydantic_to_bedrock_tool(user_model):
    tool_spec = convert_pydantic_to_bedrock_tool(user_model)

    assert tool_spec is not None
    print(tool_spec)
