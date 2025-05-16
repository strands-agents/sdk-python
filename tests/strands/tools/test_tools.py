import pytest

import strands
from strands.tools.tools import (
    FunctionTool,
    InvalidToolUseNameException,
    PythonAgentTool,
    normalize_schema,
    normalize_tool_spec,
    validate_tool_use,
    validate_tool_use_name,
)
from strands.types.tools import ToolUse


def test_validate_tool_use_name_valid():
    tool = {"name": "valid_tool_name", "toolUseId": "123"}
    # Should not raise an exception
    validate_tool_use_name(tool)


def test_validate_tool_use_name_missing():
    tool = {"toolUseId": "123"}
    with pytest.raises(InvalidToolUseNameException, match="tool name missing"):
        validate_tool_use_name(tool)


def test_validate_tool_use_name_invalid_pattern():
    tool = {"name": "123_invalid", "toolUseId": "123"}
    with pytest.raises(InvalidToolUseNameException, match="invalid tool name pattern"):
        validate_tool_use_name(tool)


def test_validate_tool_use_name_too_long():
    tool = {"name": "a" * 65, "toolUseId": "123"}
    with pytest.raises(InvalidToolUseNameException, match="invalid tool name length"):
        validate_tool_use_name(tool)


def test_validate_tool_use():
    tool = {"name": "valid_tool_name", "toolUseId": "123"}
    # Should not raise an exception
    validate_tool_use(tool)


def test_normalize_schema_basic():
    schema = {"type": "object"}
    normalized = normalize_schema(schema)
    assert normalized["type"] == "object"
    assert "properties" in normalized
    assert normalized["properties"] == {}
    assert "required" in normalized
    assert normalized["required"] == []


def test_normalize_schema_with_properties():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "User name"},
            "age": {"type": "integer", "description": "User age"},
        },
    }
    normalized = normalize_schema(schema)
    assert normalized["type"] == "object"
    assert "properties" in normalized
    assert "name" in normalized["properties"]
    assert normalized["properties"]["name"]["type"] == "string"
    assert normalized["properties"]["name"]["description"] == "User name"
    assert "age" in normalized["properties"]
    assert normalized["properties"]["age"]["type"] == "integer"
    assert normalized["properties"]["age"]["description"] == "User age"


def test_normalize_schema_with_property_removed():
    schema = {
        "type": "object",
        "properties": {"name": "invalid"},
    }
    normalized = normalize_schema(schema)
    assert "name" in normalized["properties"]
    assert normalized["properties"]["name"]["type"] == "string"
    assert normalized["properties"]["name"]["description"] == "Property name"


def test_normalize_schema_with_property_defaults():
    schema = {"properties": {"name": {}}}
    normalized = normalize_schema(schema)
    assert "name" in normalized["properties"]
    assert normalized["properties"]["name"]["type"] == "string"
    assert normalized["properties"]["name"]["description"] == "Property name"


def test_normalize_schema_with_property_enum():
    schema = {"properties": {"color": {"type": "string", "description": "color", "enum": ["red", "green", "blue"]}}}
    normalized = normalize_schema(schema)
    assert "color" in normalized["properties"]
    assert normalized["properties"]["color"]["type"] == "string"
    assert normalized["properties"]["color"]["description"] == "color"
    assert "enum" in normalized["properties"]["color"]
    assert normalized["properties"]["color"]["enum"] == ["red", "green", "blue"]


def test_normalize_schema_with_property_numeric_constraints():
    schema = {
        "properties": {
            "age": {"type": "integer", "description": "age", "minimum": 0, "maximum": 120},
            "score": {"type": "number", "description": "score", "minimum": 0.0, "maximum": 100.0},
        }
    }
    normalized = normalize_schema(schema)
    assert "age" in normalized["properties"]
    assert normalized["properties"]["age"]["type"] == "integer"
    assert normalized["properties"]["age"]["minimum"] == 0
    assert normalized["properties"]["age"]["maximum"] == 120
    assert "score" in normalized["properties"]
    assert normalized["properties"]["score"]["type"] == "number"
    assert normalized["properties"]["score"]["minimum"] == 0.0
    assert normalized["properties"]["score"]["maximum"] == 100.0


def test_normalize_schema_with_required():
    schema = {"type": "object", "required": ["name", "email"]}
    normalized = normalize_schema(schema)
    assert "required" in normalized
    assert normalized["required"] == ["name", "email"]


def test_normalize_tool_spec_with_json_schema():
    tool_spec = {
        "name": "test_tool",
        "description": "A test tool",
        "inputSchema": {"json": {"type": "object", "properties": {"query": {}}, "required": ["query"]}},
    }
    normalized = normalize_tool_spec(tool_spec)
    assert normalized["name"] == "test_tool"
    assert normalized["description"] == "A test tool"
    assert "inputSchema" in normalized
    assert "json" in normalized["inputSchema"]
    assert normalized["inputSchema"]["json"]["type"] == "object"
    assert "query" in normalized["inputSchema"]["json"]["properties"]
    assert normalized["inputSchema"]["json"]["properties"]["query"]["type"] == "string"
    assert normalized["inputSchema"]["json"]["properties"]["query"]["description"] == "Property query"


def test_normalize_tool_spec_with_direct_schema():
    tool_spec = {
        "name": "test_tool",
        "description": "A test tool",
        "inputSchema": {"type": "object", "properties": {"query": {}}, "required": ["query"]},
    }
    normalized = normalize_tool_spec(tool_spec)
    assert normalized["name"] == "test_tool"
    assert normalized["description"] == "A test tool"
    assert "inputSchema" in normalized
    assert "json" in normalized["inputSchema"]
    assert normalized["inputSchema"]["json"]["type"] == "object"
    assert "query" in normalized["inputSchema"]["json"]["properties"]
    assert normalized["inputSchema"]["json"]["required"] == ["query"]


def test_normalize_tool_spec_without_input_schema():
    tool_spec = {"name": "test_tool", "description": "A test tool"}
    normalized = normalize_tool_spec(tool_spec)
    assert normalized["name"] == "test_tool"
    assert normalized["description"] == "A test tool"
    # Should not modify the spec if inputSchema is not present
    assert "inputSchema" not in normalized


def test_normalize_tool_spec_empty_input_schema():
    tool_spec = {
        "name": "test_tool",
        "description": "A test tool",
        "inputSchema": "",
    }
    normalized = normalize_tool_spec(tool_spec)
    assert normalized["name"] == "test_tool"
    assert normalized["description"] == "A test tool"
    # Should not modify the spec if inputSchema is not a dict
    assert normalized["inputSchema"] == ""


def test_validate_tool_use_with_valid_input():
    tool_use: ToolUse = {
        "name": "valid",
        "toolUseId": "123",
        "input": {},
    }
    strands.tools.tools.validate_tool_use(tool_use)


@pytest.mark.parametrize(
    ("tool_use", "expected_error"),
    [
        # Name - Invalid characters
        (
            {
                "name": "1-invalid",
                "toolUseId": "123",
                "input": {},
            },
            strands.tools.InvalidToolUseNameException,
        ),
        # Name - Exceeds max length
        (
            {
                "name": "a" * 65,
                "toolUseId": "123",
                "input": {},
            },
            strands.tools.InvalidToolUseNameException,
        ),
    ],
)
def test_validate_tool_use_invalid(tool_use, expected_error):
    with pytest.raises(expected_error):
        strands.tools.tools.validate_tool_use(tool_use)


@pytest.fixture
def function():
    def identity(a: int) -> int:
        return a

    return identity


@pytest.fixture
def tool_function(function):
    return strands.tools.tool(function)


@pytest.fixture
def tool(tool_function):
    return FunctionTool(tool_function, tool_name="identity")


def test__init__invalid_name():
    def identity(a):
        return a

    identity.TOOL_SPEC = {"name": 0}

    with pytest.raises(ValueError, match="Tool name must be a string"):
        FunctionTool(identity)


def test__init__missing_spec():
    def identity(a):
        return a

    with pytest.raises(ValueError, match="Function identity is not decorated with @tool"):
        FunctionTool(identity)


def test_tool_name(tool):
    tru_name = tool.tool_name
    exp_name = "identity"

    assert tru_name == exp_name


def test_tool_spec(tool):
    exp_spec = {
        "name": "identity",
        "description": "identity",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "a": {
                        "description": "Parameter a",
                        "type": "integer",
                    },
                },
                "required": ["a"],
            }
        },
    }

    tru_spec = tool.tool_spec
    assert tru_spec == exp_spec


def test_tool_type(tool):
    tru_type = tool.tool_type
    exp_type = "function"

    assert tru_type == exp_type


def test_supports_hot_reload(tool):
    assert tool.supports_hot_reload


def test_original_function(tool, function):
    tru_name = tool.original_function.__name__
    exp_name = function.__name__

    assert tru_name == exp_name


def test_original_function_not_decorated():
    def identity(a: int):
        return a

    identity.TOOL_SPEC = {}

    tool = FunctionTool(identity, tool_name="identity")

    tru_name = tool.original_function.__name__
    exp_name = "identity"

    assert tru_name == exp_name


def test_get_display_properties(tool):
    tru_properties = tool.get_display_properties()
    exp_properties = {
        "Function": "identity",
        "Name": "identity",
        "Type": "function",
    }

    assert tru_properties == exp_properties


def test_invoke(tool):
    tru_output = tool.invoke({"input": {"a": 2}})
    exp_output = {"toolUseId": "unknown", "status": "success", "content": [{"text": "2"}]}

    assert tru_output == exp_output


def test_invoke_with_agent():
    @strands.tools.tool
    def identity(a: int, agent: dict = None):
        return a, agent

    tool = FunctionTool(identity, tool_name="identity")

    exp_output = {"toolUseId": "unknown", "status": "success", "content": [{"text": "(2, {'state': 1})"}]}

    tru_output = tool.invoke({"input": {"a": 2}}, agent={"state": 1})

    assert tru_output == exp_output


def test_invoke_exception():
    def identity(a: int):
        return a

    identity.TOOL_SPEC = {}

    tool = FunctionTool(identity, tool_name="identity")

    tru_output = tool.invoke({}, invalid=1)
    exp_output = {
        "toolUseId": "unknown",
        "status": "error",
        "content": [
            {
                "text": (
                    "Error executing function: "
                    "test_invoke_exception.<locals>.identity() "
                    "got an unexpected keyword argument 'invalid'"
                )
            }
        ],
    }

    assert tru_output == exp_output


# Tests from test_python_agent_tool.py
@pytest.fixture
def python_tool():
    def identity(tool_use, a):
        return tool_use, a

    return PythonAgentTool(
        tool_name="identity",
        tool_spec={
            "name": "identity",
            "description": "identity",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer",
                    },
                },
            },
        },
        callback=identity,
    )


def test_python_tool_name(python_tool):
    tru_name = python_tool.tool_name
    exp_name = "identity"

    assert tru_name == exp_name


def test_python_tool_spec(python_tool):
    tru_spec = python_tool.tool_spec
    exp_spec = {
        "name": "identity",
        "description": "identity",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "integer",
                },
            },
        },
    }

    assert tru_spec == exp_spec


def test_python_tool_type(python_tool):
    tru_type = python_tool.tool_type
    exp_type = "python"

    assert tru_type == exp_type


def test_python_invoke(python_tool):
    tru_output = python_tool.invoke({"tool_use": 1}, a=2)
    exp_output = ({"tool_use": 1}, 2)

    assert tru_output == exp_output
