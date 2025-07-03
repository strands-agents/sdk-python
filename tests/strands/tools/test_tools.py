import pytest

import strands
from strands.tools.tools import (
    InvalidToolUseNameException,
    PythonAgentTool,
    normalize_schema,
    normalize_tool_spec,
    validate_tool_use,
    validate_tool_use_name,
)
from strands.types.tools import ToolUse


def test_validate_tool_use_name_valid():
    tool1 = {"name": "valid_tool_name", "toolUseId": "123"}
    # Should not raise an exception
    validate_tool_use_name(tool1)

    tool2 = {"name": "valid-name", "toolUseId": "123"}
    # Should not raise an exception
    validate_tool_use_name(tool2)


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

    expected = {"type": "object", "properties": {}, "required": []}

    assert normalized == expected


def test_normalize_schema_with_properties():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "User name"},
            "age": {"type": "integer", "description": "User age"},
        },
    }
    normalized = normalize_schema(schema)

    expected = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "User name"},
            "age": {"type": "integer", "description": "User age"},
        },
        "required": [],
    }

    assert normalized == expected


def test_normalize_schema_with_property_removed():
    schema = {
        "type": "object",
        "properties": {"name": "invalid"},
    }
    normalized = normalize_schema(schema)

    expected = {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Property name"}},
        "required": [],
    }

    assert normalized == expected


def test_normalize_schema_with_property_defaults():
    schema = {"properties": {"name": {}}}
    normalized = normalize_schema(schema)

    expected = {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Property name"}},
        "required": [],
    }

    assert normalized == expected


def test_normalize_schema_with_property_enum():
    schema = {"properties": {"color": {"type": "string", "description": "color", "enum": ["red", "green", "blue"]}}}
    normalized = normalize_schema(schema)

    expected = {
        "type": "object",
        "properties": {"color": {"type": "string", "description": "color", "enum": ["red", "green", "blue"]}},
        "required": [],
    }

    assert normalized == expected


def test_normalize_schema_with_property_numeric_constraints():
    schema = {
        "properties": {
            "age": {"type": "integer", "description": "age", "minimum": 0, "maximum": 120},
            "score": {"type": "number", "description": "score", "minimum": 0.0, "maximum": 100.0},
        }
    }
    normalized = normalize_schema(schema)

    expected = {
        "type": "object",
        "properties": {
            "age": {"type": "integer", "description": "age", "minimum": 0, "maximum": 120},
            "score": {"type": "number", "description": "score", "minimum": 0.0, "maximum": 100.0},
        },
        "required": [],
    }

    assert normalized == expected


def test_normalize_schema_with_required():
    schema = {"type": "object", "required": ["name", "email"]}
    normalized = normalize_schema(schema)

    expected = {"type": "object", "properties": {}, "required": ["name", "email"]}

    assert normalized == expected


def test_normalize_schema_with_nested_object():
    """Test normalization of schemas with nested objects."""
    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "User name"},
                    "age": {"type": "integer", "description": "User age"},
                },
                "required": ["name"],
            }
        },
        "required": ["user"],
    }

    normalized = normalize_schema(schema)

    expected = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "User name"},
                    "age": {"type": "integer", "description": "User age"},
                },
                "required": ["name"],
            }
        },
        "required": ["user"],
    }

    assert normalized == expected


def test_normalize_schema_with_deeply_nested_objects():
    """Test normalization of deeply nested object structures."""
    schema = {
        "type": "object",
        "properties": {
            "level1": {
                "type": "object",
                "properties": {
                    "level2": {
                        "type": "object",
                        "properties": {
                            "level3": {"type": "object", "properties": {"value": {"type": "string", "const": "fixed"}}}
                        },
                    }
                },
            }
        },
    }

    normalized = normalize_schema(schema)

    expected = {
        "type": "object",
        "properties": {
            "level1": {
                "type": "object",
                "properties": {
                    "level2": {
                        "type": "object",
                        "properties": {
                            "level3": {
                                "type": "object",
                                "properties": {
                                    "value": {"type": "string", "description": "Property value", "const": "fixed"}
                                },
                                "required": [],
                            }
                        },
                        "required": [],
                    }
                },
                "required": [],
            }
        },
        "required": [],
    }

    assert normalized == expected


def test_normalize_schema_with_const_constraint():
    """Test that const constraints are preserved."""
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "const": "ACTIVE"},
            "config": {"type": "object", "properties": {"mode": {"type": "string", "const": "PRODUCTION"}}},
        },
    }

    normalized = normalize_schema(schema)

    expected = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "description": "Property status", "const": "ACTIVE"},
            "config": {
                "type": "object",
                "properties": {"mode": {"type": "string", "description": "Property mode", "const": "PRODUCTION"}},
                "required": [],
            },
        },
        "required": [],
    }

    assert normalized == expected


def test_normalize_schema_with_additional_properties():
    """Test that additionalProperties constraint is preserved."""
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "data": {"type": "object", "properties": {"id": {"type": "string"}}, "additionalProperties": False}
        },
    }

    normalized = normalize_schema(schema)

    expected = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "data": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"id": {"type": "string", "description": "Property id"}},
                "required": [],
            }
        },
        "required": [],
    }

    assert normalized == expected


def test_normalize_tool_spec_with_json_schema():
    tool_spec = {
        "name": "test_tool",
        "description": "A test tool",
        "inputSchema": {"json": {"type": "object", "properties": {"query": {}}, "required": ["query"]}},
    }
    normalized = normalize_tool_spec(tool_spec)

    expected = {
        "name": "test_tool",
        "description": "A test tool",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Property query"}},
                "required": ["query"],
            }
        },
    }

    assert normalized == expected


def test_normalize_tool_spec_with_direct_schema():
    tool_spec = {
        "name": "test_tool",
        "description": "A test tool",
        "inputSchema": {"type": "object", "properties": {"query": {}}, "required": ["query"]},
    }
    normalized = normalize_tool_spec(tool_spec)

    expected = {
        "name": "test_tool",
        "description": "A test tool",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Property query"}},
                "required": ["query"],
            }
        },
    }

    assert normalized == expected


def test_normalize_tool_spec_without_input_schema():
    tool_spec = {"name": "test_tool", "description": "A test tool"}
    normalized = normalize_tool_spec(tool_spec)

    expected = {"name": "test_tool", "description": "A test tool"}

    assert normalized == expected


def test_normalize_tool_spec_empty_input_schema():
    tool_spec = {
        "name": "test_tool",
        "description": "A test tool",
        "inputSchema": "",
    }
    normalized = normalize_tool_spec(tool_spec)

    expected = {"name": "test_tool", "description": "A test tool", "inputSchema": ""}

    assert normalized == expected


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
def tool(function):
    return strands.tools.tool(function)


def test__init__invalid_name():
    with pytest.raises(ValueError, match="Tool name must be a string"):

        @strands.tool(name=0)
        def identity(a):
            return a


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

    tool = strands.tool(func=identity, name="identity")

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

    exp_output = {"toolUseId": "unknown", "status": "success", "content": [{"text": "(2, {'state': 1})"}]}

    tru_output = identity.invoke({"input": {"a": 2}}, agent={"state": 1})

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
