"""Tests for structured output utility functions."""

from typing import List, Optional

import pytest
from pydantic import BaseModel, Field

from strands.tools.structured_output.structured_output_utils import (
    _expand_nested_properties,
    _flatten_schema,
    _process_nested_dict,
    _process_properties,
    _process_property,
    _process_referenced_models,
    _process_schema_object,
    convert_pydantic_to_tool_spec,
)


class SimpleModel(BaseModel):
    """A simple test model."""

    name: str = Field(description="The name field")
    age: int = Field(description="The age field")


class OptionalFieldModel(BaseModel):
    """Model with optional fields."""

    required_field: str = Field(description="A required field")
    optional_field: Optional[str] = Field(None, description="An optional field")
    optional_int: Optional[int] = Field(None, description="An optional integer")


class NestedModel(BaseModel):
    """Model with nested structure."""

    simple: SimpleModel = Field(description="A nested simple model")
    description: str = Field(description="A description")


class DeeplyNestedModel(BaseModel):
    """Model with deep nesting."""

    nested: NestedModel = Field(description="A nested model")
    extra_field: str = Field(description="An extra field")


class ListModel(BaseModel):
    """Model with list fields."""

    items: List[str] = Field(description="A list of strings")
    numbers: List[int] = Field(description="A list of numbers")
    models: List[SimpleModel] = Field(description="A list of models")


class SelfReferencingModel(BaseModel):
    """Model that references itself (for circular ref testing)."""

    name: str = Field(description="Name field")
    child: Optional["SelfReferencingModel"] = Field(None, description="Optional child")

    model_config = {"from_attributes": True}


class ModelWithDocstring(BaseModel):
    """This is a model with a docstring description."""

    field1: str = Field(description="First field")
    field2: int = Field(description="Second field")


class ModelWithoutDocstring(BaseModel):
    field1: str = Field(description="First field")


class TestConvertPydanticToToolSpec:
    """Test convert_pydantic_to_tool_spec function."""

    def test_simple_model_conversion(self):
        """Test converting a simple Pydantic model to tool spec."""
        result = convert_pydantic_to_tool_spec(SimpleModel)

        assert result["name"] == "SimpleModel"
        assert result["description"] == "A simple test model."
        assert "inputSchema" in result
        assert "json" in result["inputSchema"]

        schema = result["inputSchema"]["json"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["name"]["description"] == "The name field"
        assert schema["properties"]["age"]["type"] == "integer"
        assert schema["properties"]["age"]["description"] == "The age field"
        assert "required" in schema
        assert set(schema["required"]) == {"name", "age"}

    def test_model_with_custom_description(self):
        """Test providing a custom description overrides docstring."""
        custom_desc = "Custom tool description"
        result = convert_pydantic_to_tool_spec(SimpleModel, description=custom_desc)

        assert result["description"] == custom_desc

    def test_model_without_docstring(self):
        """Test model without docstring gets default description."""
        result = convert_pydantic_to_tool_spec(ModelWithoutDocstring)

        assert result["description"] == "ModelWithoutDocstring structured output tool"

    def test_optional_fields_handling(self):
        """Test that optional fields are handled correctly."""
        result = convert_pydantic_to_tool_spec(OptionalFieldModel)

        schema = result["inputSchema"]["json"]
        assert "required" in schema
        assert "required_field" in schema["required"]
        assert "optional_field" not in schema["required"]
        assert "optional_int" not in schema["required"]

        # Check optional fields have null type
        assert "null" in schema["properties"]["optional_field"]["type"]
        assert "null" in schema["properties"]["optional_int"]["type"]
        assert "null" not in str(schema["properties"]["required_field"]["type"])

    def test_nested_model_conversion(self):
        """Test converting a model with nested Pydantic models."""
        result = convert_pydantic_to_tool_spec(NestedModel)

        schema = result["inputSchema"]["json"]
        assert "simple" in schema["properties"]

        # Check nested model is expanded
        simple_prop = schema["properties"]["simple"]
        assert simple_prop["type"] == "object"
        assert "properties" in simple_prop
        assert "name" in simple_prop["properties"]
        assert "age" in simple_prop["properties"]
        assert simple_prop["properties"]["name"]["type"] == "string"
        assert simple_prop["properties"]["age"]["type"] == "integer"

    def test_deeply_nested_model(self):
        """Test converting a model with multiple levels of nesting."""
        result = convert_pydantic_to_tool_spec(DeeplyNestedModel)

        schema = result["inputSchema"]["json"]
        nested_prop = schema["properties"]["nested"]
        assert nested_prop["type"] == "object"
        assert "properties" in nested_prop
        assert "simple" in nested_prop["properties"]
        assert "description" in nested_prop["properties"]

        # Check deeper nesting
        simple_prop = nested_prop["properties"]["simple"]
        assert "properties" in simple_prop
        assert "name" in simple_prop["properties"]
        assert "age" in simple_prop["properties"]

    def test_list_model_conversion(self):
        """Test converting a model with list fields."""
        result = convert_pydantic_to_tool_spec(ListModel)

        schema = result["inputSchema"]["json"]
        assert "items" in schema["properties"]
        assert "numbers" in schema["properties"]
        assert "models" in schema["properties"]

        # Check list types
        assert schema["properties"]["items"]["type"] == "array"
        assert schema["properties"]["items"]["items"]["type"] == "string"
        assert schema["properties"]["numbers"]["type"] == "array"
        assert schema["properties"]["numbers"]["items"]["type"] == "integer"
        assert schema["properties"]["models"]["type"] == "array"
        # The models list should have object items
        assert "type" in schema["properties"]["models"]["items"]


class TestFlattenSchema:
    """Test _flatten_schema function."""

    def test_flatten_simple_schema(self):
        """Test flattening a simple schema."""
        schema = {
            "type": "object",
            "title": "TestModel",
            "description": "Test description",
            "properties": {
                "field1": {"type": "string", "description": "Field 1"},
                "field2": {"type": "integer", "description": "Field 2"},
            },
            "required": ["field1"],
        }

        result = _flatten_schema(schema)

        assert result["type"] == "object"
        assert result["title"] == "TestModel"
        assert result["description"] == "Test description"
        assert "field1" in result["properties"]
        assert "field2" in result["properties"]
        assert result["required"] == ["field1"]

    def test_flatten_schema_with_refs(self):
        """Test flattening a schema with $ref references."""
        schema = {
            "type": "object",
            "properties": {"nested": {"$ref": "#/$defs/NestedModel"}},
            "$defs": {"NestedModel": {"type": "object", "properties": {"name": {"type": "string"}}}},
            "required": ["nested"],
        }

        result = _flatten_schema(schema)

        assert "nested" in result["properties"]
        assert result["properties"]["nested"]["type"] == "object"
        assert "properties" in result["properties"]["nested"]
        assert "name" in result["properties"]["nested"]["properties"]

    def test_flatten_schema_with_anyof(self):
        """Test flattening a schema with anyOf (optional fields)."""
        schema = {
            "type": "object",
            "properties": {"optional_field": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
            "required": [],
        }

        result = _flatten_schema(schema)

        assert "optional_field" in result["properties"]
        field = result["properties"]["optional_field"]
        assert "null" in field["type"]

    def test_flatten_schema_without_required(self):
        """Test flattening a schema without required field."""
        schema = {"type": "object", "properties": {"field1": {"type": "string"}}}

        result = _flatten_schema(schema)

        assert "properties" in result
        assert "field1" in result["properties"]
        # No required field should be added if all fields are optional
        assert "required" not in result or result["required"] == []

    def test_flatten_schema_circular_ref_error(self):
        """Test that circular references without properties raise an error."""
        schema = {"$ref": "#/$defs/Circular", "$defs": {"Circular": {"$ref": "#/$defs/Circular"}}}

        with pytest.raises(ValueError, match="Circular reference detected"):
            _flatten_schema(schema)


class TestProcessProperty:
    """Test _process_property function."""

    def test_process_simple_property(self):
        """Test processing a simple property."""
        prop = {"type": "string", "description": "A string field"}
        defs = {}

        result = _process_property(prop, defs, is_required=True)

        assert result["type"] == "string"
        assert result["description"] == "A string field"

    def test_process_optional_property(self):
        """Test processing an optional property."""
        prop = {"type": "string", "description": "An optional string"}
        defs = {}

        result = _process_property(prop, defs, is_required=False)

        assert result["type"] == ["string", "null"]
        assert result["description"] == "An optional string"

    def test_process_property_with_anyof(self):
        """Test processing a property with anyOf."""
        prop = {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "Optional field"}
        defs = {}

        result = _process_property(prop, defs)

        assert "null" in result["type"]
        assert result["description"] == "Optional field"

    def test_process_property_with_ref(self):
        """Test processing a property with $ref."""
        prop = {"$ref": "#/$defs/RefModel"}
        defs = {"RefModel": {"type": "object", "properties": {"name": {"type": "string"}}}}

        result = _process_property(prop, defs)

        assert result["type"] == "object"
        assert "properties" in result
        assert "name" in result["properties"]

    def test_process_property_with_missing_ref(self):
        """Test processing a property with missing $ref raises error."""
        prop = {"$ref": "#/$defs/MissingModel"}
        defs = {}

        with pytest.raises(ValueError, match="Missing reference: MissingModel"):
            _process_property(prop, defs)


class TestProcessSchemaObject:
    """Test _process_schema_object function."""

    def test_process_simple_schema_object(self):
        """Test processing a simple schema object."""
        schema_obj = {
            "type": "object",
            "properties": {"field1": {"type": "string"}, "field2": {"type": "integer"}},
            "required": ["field1"],
        }
        defs = {}

        result = _process_schema_object(schema_obj, defs)

        assert result["type"] == "object"
        assert "properties" in result
        assert "field1" in result["properties"]
        assert "field2" in result["properties"]
        assert result["required"] == ["field1"]

    def test_process_schema_object_with_nested_refs(self):
        """Test processing a schema object with nested references."""
        schema_obj = {"type": "object", "properties": {"nested": {"$ref": "#/$defs/Nested"}}, "required": ["nested"]}
        defs = {"Nested": {"type": "object", "properties": {"name": {"type": "string"}}}}

        result = _process_schema_object(schema_obj, defs)

        assert "nested" in result["properties"]
        nested = result["properties"]["nested"]
        assert nested["type"] == "object"
        assert "properties" in nested
        assert "name" in nested["properties"]


class TestProcessNestedDict:
    """Test _process_nested_dict function."""

    def test_process_simple_dict(self):
        """Test processing a simple dictionary."""
        d = {"type": "string", "description": "A string"}
        defs = {}

        result = _process_nested_dict(d, defs)

        assert result["type"] == "string"
        assert result["description"] == "A string"

    def test_process_dict_with_ref(self):
        """Test processing a dictionary with $ref."""
        d = {"$ref": "#/$defs/Model"}
        defs = {"Model": {"type": "object", "properties": {"field": {"type": "string"}}}}

        result = _process_nested_dict(d, defs)

        assert result["type"] == "object"
        assert "properties" in result
        assert "field" in result["properties"]

    def test_process_dict_with_nested_dicts(self):
        """Test processing a dictionary with nested dictionaries."""
        d = {
            "type": "object",
            "properties": {"nested": {"type": "object", "properties": {"field": {"type": "string"}}}},
        }
        defs = {}

        result = _process_nested_dict(d, defs)

        assert result["type"] == "object"
        assert "properties" in result
        assert "nested" in result["properties"]
        assert result["properties"]["nested"]["properties"]["field"]["type"] == "string"

    def test_process_dict_with_list(self):
        """Test processing a dictionary with list values."""
        d = {"type": "array", "items": {"type": "string"}, "enum": ["value1", "value2", "value3"]}
        defs = {}

        result = _process_nested_dict(d, defs)

        assert result["type"] == "array"
        assert result["items"]["type"] == "string"
        assert result["enum"] == ["value1", "value2", "value3"]


class TestExpandNestedProperties:
    """Test _expand_nested_properties function."""

    def test_expand_nested_model_properties(self):
        """Test expanding nested model properties."""
        schema = NestedModel.model_json_schema()
        _expand_nested_properties(schema, NestedModel)

        # Check that the simple field has been expanded
        assert "simple" in schema["properties"]
        simple_prop = schema["properties"]["simple"]
        assert "properties" in simple_prop
        assert "name" in simple_prop["properties"]
        assert "age" in simple_prop["properties"]

    def test_expand_optional_nested_model(self):
        """Test expanding optional nested model properties."""

        class ModelWithOptionalNested(BaseModel):
            nested: Optional[SimpleModel] = Field(None, description="Optional nested")

        schema = ModelWithOptionalNested.model_json_schema()
        _expand_nested_properties(schema, ModelWithOptionalNested)

        nested_prop = schema["properties"]["nested"]
        # For Optional fields, the schema may use anyOf structure
        if "anyOf" in nested_prop:
            # Check that one option is the model and one is null
            assert any(opt.get("type") == "null" for opt in nested_prop["anyOf"])
            assert any("$ref" in opt or "properties" in opt for opt in nested_prop["anyOf"])
        else:
            # Or it might be expanded with type including null
            assert "type" in nested_prop
            assert "null" in nested_prop["type"]
            assert "properties" in nested_prop


class TestProcessReferencedModels:
    """Test _process_referenced_models function."""

    def test_process_referenced_models_with_docstring(self):
        """Test processing referenced models adds docstrings."""
        schema = NestedModel.model_json_schema()
        _process_referenced_models(schema, NestedModel)

        if "$defs" in schema and "SimpleModel" in schema["$defs"]:
            simple_def = schema["$defs"]["SimpleModel"]
            # Should have the docstring from SimpleModel
            assert simple_def.get("description") == "A simple test model."

    def test_process_referenced_models_nested_properties(self):
        """Test processing adds descriptions to nested properties."""
        schema = NestedModel.model_json_schema()
        _process_referenced_models(schema, NestedModel)

        if "$defs" in schema and "SimpleModel" in schema["$defs"]:
            simple_def = schema["$defs"]["SimpleModel"]
            if "properties" in simple_def:
                # Properties should have their field descriptions
                assert simple_def["properties"]["name"].get("description") == "The name field"
                assert simple_def["properties"]["age"].get("description") == "The age field"


class TestProcessProperties:
    """Test _process_properties function."""

    def test_process_properties_adds_descriptions(self):
        """Test that _process_properties adds field descriptions."""
        schema_def = {"properties": {"name": {}, "age": {}}}

        _process_properties(schema_def, SimpleModel)

        assert schema_def["properties"]["name"]["description"] == "The name field"
        assert schema_def["properties"]["age"]["description"] == "The age field"

    def test_process_properties_preserves_existing_descriptions(self):
        """Test that existing descriptions are preserved."""
        schema_def = {"properties": {"name": {"description": "Existing description"}, "age": {}}}

        _process_properties(schema_def, SimpleModel)

        # Should preserve existing description
        assert schema_def["properties"]["name"]["description"] == "Existing description"
        # Should add missing description
        assert schema_def["properties"]["age"]["description"] == "The age field"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_model(self):
        """Test converting an empty Pydantic model."""

        class EmptyModel(BaseModel):
            """An empty model."""

            pass

        result = convert_pydantic_to_tool_spec(EmptyModel)

        assert result["name"] == "EmptyModel"
        assert result["description"] == "An empty model."
        schema = result["inputSchema"]["json"]
        assert schema["type"] == "object"
        assert schema["properties"] == {}

    def test_complex_nested_optional(self):
        """Test complex nested optional structures."""

        class ComplexModel(BaseModel):
            """Complex model with various optional nested structures."""

            required_nested: SimpleModel
            optional_nested: Optional[SimpleModel] = None
            optional_list: Optional[List[SimpleModel]] = None

        result = convert_pydantic_to_tool_spec(ComplexModel)

        schema = result["inputSchema"]["json"]
        assert "required_nested" in schema["required"]
        assert "optional_nested" not in schema.get("required", [])
        assert "optional_list" not in schema.get("required", [])

        # Check optional nested has null type
        optional_prop = schema["properties"]["optional_nested"]
        assert "null" in optional_prop["type"]

    def test_model_with_default_values(self):
        """Test model with default values."""

        class ModelWithDefaults(BaseModel):
            """Model with default values."""

            name: str = Field(default="default_name", description="Name with default")
            count: int = Field(default=0, description="Count with default")

        result = convert_pydantic_to_tool_spec(ModelWithDefaults)

        schema = result["inputSchema"]["json"]
        # Fields with defaults should not be in required
        assert "required" not in schema or len(schema["required"]) == 0
