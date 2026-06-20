"""Tests for Bedrock strict_tools validation.

Validates that BedrockModel enforces Bedrock strict-mode constraints at build time:
1. No oneOf in tool schemas
2. Aggregate optional parameters ≤ 24 across all tools

Regression test for https://github.com/strands-agents/harness-sdk/issues/2664
"""
import pytest

from strands.models._strict_schema import (
    _count_optional_params,
    _has_oneof,
    validate_bedrock_strict_constraints,
)


class TestOneOfDetection:
    """Test _has_oneof() correctly detects oneOf in various positions."""

    def test_direct_oneof(self):
        """Direct oneOf at root level."""
        schema = {
            "oneOf": [
                {"type": "string"},
                {"type": "number"},
            ]
        }
        assert _has_oneof(schema) is True

    def test_nested_in_properties(self):
        """oneOf nested inside properties."""
        schema = {
            "type": "object",
            "properties": {
                "field": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "number"},
                    ]
                }
            },
        }
        assert _has_oneof(schema) is True

    def test_nested_in_items(self):
        """oneOf inside array items."""
        schema = {
            "type": "array",
            "items": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "number"},
                ]
            },
        }
        assert _has_oneof(schema) is True

    def test_nested_in_anyof(self):
        """oneOf inside anyOf variant."""
        schema = {
            "anyOf": [
                {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "number"},
                    ]
                }
            ]
        }
        assert _has_oneof(schema) is True

    def test_nested_in_defs(self):
        """oneOf inside $defs."""
        schema = {
            "type": "object",
            "$defs": {
                "MyType": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "number"},
                    ]
                }
            },
        }
        assert _has_oneof(schema) is True

    def test_no_oneof(self):
        """Schema without oneOf."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
            },
        }
        assert _has_oneof(schema) is False

    def test_circular_reference_no_infinite_loop(self):
        """Circular references should not cause infinite recursion."""
        schema = {
            "type": "object",
            "properties": {
                "self": {}
            },
        }
        # Create circular reference
        schema["properties"]["self"] = schema
        # Should complete without stack overflow
        assert _has_oneof(schema) is False


class TestOptionalParamCounting:
    """Test _count_optional_params() correctly counts optional parameters."""

    def test_all_optional(self):
        """All properties are optional (not in required)."""
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "number"},
                "c": {"type": "boolean"},
            },
            "required": [],
        }
        assert _count_optional_params(schema) == 3

    def test_all_required(self):
        """All properties are required."""
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        }
        assert _count_optional_params(schema) == 0

    def test_mixed_optional_required(self):
        """Mix of optional and required."""
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "number"},
                "c": {"type": "boolean"},
            },
            "required": ["a"],
        }
        assert _count_optional_params(schema) == 2

    def test_no_required_field(self):
        """Schema without required field (defaults to all optional)."""
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "number"},
            },
        }
        assert _count_optional_params(schema) == 2

    def test_non_object_type(self):
        """Non-object types have no optional params."""
        schema = {"type": "string"}
        assert _count_optional_params(schema) == 0


class TestBedrockStrictValidation:
    """Test validate_bedrock_strict_constraints() integration."""

    def test_rejects_oneof_with_strict_true(self):
        """Strict mode rejects tool with oneOf."""
        tool_specs = [
            {
                "name": "test_tool",
                "description": "Test",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "oneOf": [
                            {"properties": {"a": {"type": "string"}}},
                            {"properties": {"b": {"type": "number"}}},
                        ],
                    }
                },
            }
        ]

        with pytest.raises(ValueError, match="contains unsupported 'oneOf'"):
            validate_bedrock_strict_constraints(tool_specs, strict_tools=True)

    def test_allows_oneof_with_strict_false(self):
        """Non-strict mode allows oneOf."""
        tool_specs = [
            {
                "name": "test_tool",
                "description": "Test",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "oneOf": [
                            {"properties": {"a": {"type": "string"}}},
                            {"properties": {"b": {"type": "number"}}},
                        ],
                    }
                },
            }
        ]

        # Should not raise
        validate_bedrock_strict_constraints(tool_specs, strict_tools=False)

    def test_rejects_too_many_optional_params(self):
        """Strict mode rejects when aggregate optional params > 24."""
        # Create 3 tools with 10 optional params each (30 total)
        tool_specs = []
        for i in range(3):
            tool_specs.append(
                {
                    "name": f"tool_{i}",
                    "description": f"Tool {i}",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {f"param_{j}": {"type": "string"} for j in range(10)},
                            "required": [],
                        }
                    },
                }
            )

        with pytest.raises(ValueError, match="collectively have 30 optional parameters"):
            validate_bedrock_strict_constraints(tool_specs, strict_tools=True)

    def test_allows_within_optional_limit(self):
        """Strict mode allows exactly 24 optional params."""
        # Create 3 tools with 8 optional params each (24 total)
        tool_specs = []
        for i in range(3):
            tool_specs.append(
                {
                    "name": f"tool_{i}",
                    "description": f"Tool {i}",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {f"param_{j}": {"type": "string"} for j in range(8)},
                            "required": [],
                        }
                    },
                }
            )

        # Should not raise
        validate_bedrock_strict_constraints(tool_specs, strict_tools=True)

    def test_error_message_names_tools(self):
        """Error message lists tools and their optional param counts."""
        tool_specs = [
            {
                "name": "tool_a",
                "description": "A",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {f"param_{j}": {"type": "string"} for j in range(10)},
                        "required": [],
                    }
                },
            },
            {
                "name": "tool_b",
                "description": "B",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {f"param_{j}": {"type": "string"} for j in range(15)},
                        "required": [],
                    }
                },
            },
        ]

        with pytest.raises(ValueError) as exc_info:
            validate_bedrock_strict_constraints(tool_specs, strict_tools=True)

        error_msg = str(exc_info.value)
        assert "tool_a: 10 optional" in error_msg
        assert "tool_b: 15 optional" in error_msg
        assert "collectively have 25 optional parameters" in error_msg

    def test_nested_oneof_detection(self):
        """Detects oneOf buried in nested properties."""
        tool_specs = [
            {
                "name": "nested_tool",
                "description": "Test",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "outer": {
                                "type": "object",
                                "properties": {
                                    "inner": {
                                        "oneOf": [
                                            {"type": "string"},
                                            {"type": "number"},
                                        ]
                                    }
                                },
                            }
                        },
                    }
                },
            }
        ]

        with pytest.raises(ValueError, match="contains unsupported 'oneOf'"):
            validate_bedrock_strict_constraints(tool_specs, strict_tools=True)

    def test_empty_tool_list(self):
        """Empty tool list is valid."""
        validate_bedrock_strict_constraints([], strict_tools=True)

    def test_tools_with_zero_optional_params(self):
        """Tools with all required params contribute zero to limit."""
        tool_specs = [
            {
                "name": "tool_required",
                "description": "All required",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "string"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    }
                },
            }
        ]

        # Should not raise (0 optional params)
        validate_bedrock_strict_constraints(tool_specs, strict_tools=True)

    def test_property_named_oneof_is_not_detected_as_combinator(self):
        """Regression: property literally named 'oneOf' should not be detected as oneOf combinator."""
        schema = {
            "type": "object",
            "properties": {
                "oneOf": {"type": "string"}  # This is a field name, not a combinator
            }
        }
        assert _has_oneof(schema) is False

    def test_nested_property_named_oneof_is_not_detected(self):
        """Regression: nested property named 'oneOf' should not trigger false positive."""
        schema = {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "oneOf": {"type": "string"}  # Nested field named oneOf
                    }
                }
            }
        }
        assert _has_oneof(schema) is False

    def test_real_oneof_in_properties_is_still_detected(self):
        """Regression: real oneOf combinator inside properties should still be detected."""
        schema = {
            "type": "object",
            "properties": {
                "field": {
                    "oneOf": [  # This IS a real combinator
                        {"type": "string"},
                        {"type": "number"}
                    ]
                }
            }
        }
        assert _has_oneof(schema) is True

    def test_count_optional_params_with_invalid_required_list(self):
        """Regression: required listing non-existent properties should not go negative."""
        schema = {
            "type": "object",
            "properties": {
                "x": {"type": "string"},
                "y": {"type": "number"}
            },
            "required": ["x", "y", "g1", "g2", "g3"]  # g1, g2, g3 don't exist
        }
        # Should return 0, not -3
        assert _count_optional_params(schema) == 0

    def test_count_optional_params_normal_case_still_works(self):
        """Regression: ensure normal optional counting still works after fix."""
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "number"},
                "c": {"type": "boolean"}
            },
            "required": ["a"]
        }
        # b and c are optional
        assert _count_optional_params(schema) == 2
