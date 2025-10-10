"""Tests for SchemaRegistry."""

import json
import tempfile
from pathlib import Path
from typing import Optional

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from strands.experimental.config_loader.agent.schema_registry import SchemaRegistry


class UserModel(BaseModel):
    """Test Pydantic model for registry tests."""

    name: str
    age: Optional[int] = None
    email: Optional[str] = None


class TestSchemaRegistry:
    """Test cases for SchemaRegistry."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = SchemaRegistry()

    def test_register_schema_with_dict(self):
        """Test registering schema with dictionary."""
        schema_dict = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }

        self.registry.register_schema("User", schema_dict)

        # Should be able to retrieve the schema
        user_model = self.registry.get_schema("User")
        assert issubclass(user_model, BaseModel)

        # Test the generated model
        user = user_model(name="John", age=30)
        assert user.name == "John"
        assert user.age == 30

    def test_register_schema_with_pydantic_class(self):
        """Test registering schema with existing Pydantic class."""
        self.registry.register_schema("TestUser", UserModel)

        # Should be able to retrieve the schema
        retrieved_model = self.registry.get_schema("TestUser")
        assert retrieved_model is UserModel

        # Test the model
        user = retrieved_model(name="Jane")
        assert user.name == "Jane"
        assert user.age is None

    def test_register_schema_with_class_path(self):
        """Test registering schema with Python class path."""
        class_path = "tests.strands.experimental.config_loader.agent.test_schema_registry.UserModel"

        self.registry.register_schema("UserFromPath", class_path)

        # Should be able to retrieve the schema
        retrieved_model = self.registry.get_schema("UserFromPath")
        assert retrieved_model is UserModel

    def test_register_from_config_inline_schema(self):
        """Test registering schema from inline configuration."""
        config = {
            "name": "Customer",
            "description": "Customer information",
            "schema": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                },
                "required": ["customer_id", "name"],
            },
        }

        self.registry.register_from_config(config)

        # Should be able to retrieve and use the schema
        customer_model = self.registry.get_schema("Customer")
        customer = customer_model(customer_id="CUST-123", name="John Doe", email="john@example.com")

        assert customer.customer_id == "CUST-123"
        assert customer.name == "John Doe"
        assert customer.email == "john@example.com"

    def test_register_from_config_python_class(self):
        """Test registering schema from Python class configuration."""
        config = {
            "name": "UserModel",
            "description": "User model from existing class",
            "python_class": "tests.strands.experimental.config_loader.agent.test_schema_registry.UserModel",
        }

        self.registry.register_from_config(config)

        # Should be able to retrieve the schema
        retrieved_model = self.registry.get_schema("UserModel")
        assert retrieved_model is UserModel

    def test_register_from_config_external_json_file(self):
        """Test registering schema from external JSON file."""
        schema_dict = {
            "type": "object",
            "properties": {
                "product_id": {"type": "string"},
                "name": {"type": "string"},
                "price": {"type": "number", "minimum": 0},
            },
            "required": ["product_id", "name", "price"],
        }

        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(schema_dict, f)
            temp_file = f.name

        try:
            config = {"name": "Product", "description": "Product from JSON file", "schema_file": temp_file}

            self.registry.register_from_config(config)

            # Should be able to retrieve and use the schema
            product_model = self.registry.get_schema("Product")
            product = product_model(product_id="PROD-123", name="Widget", price=19.99)

            assert product.product_id == "PROD-123"
            assert product.name == "Widget"
            assert product.price == 19.99

        finally:
            Path(temp_file).unlink()

    def test_register_from_config_external_yaml_file(self):
        """Test registering schema from external YAML file."""
        schema_dict = {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "customer_name": {"type": "string"},
                "total": {"type": "number", "minimum": 0},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "quantity": {"type": "integer", "minimum": 1}},
                        "required": ["name", "quantity"],
                    },
                },
            },
            "required": ["order_id", "customer_name", "total"],
        }

        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(schema_dict, f)
            temp_file = f.name

        try:
            config = {"name": "Order", "description": "Order from YAML file", "schema_file": temp_file}

            self.registry.register_from_config(config)

            # Should be able to retrieve and use the schema
            order_model = self.registry.get_schema("Order")
            order = order_model(
                order_id="ORD-123",
                customer_name="Jane Doe",
                total=99.99,
                items=[{"name": "Widget", "quantity": 2}, {"name": "Gadget", "quantity": 1}],
            )

            assert order.order_id == "ORD-123"
            assert order.customer_name == "Jane Doe"
            assert order.total == 99.99
            assert len(order.items) == 2
            assert order.items[0].name == "Widget"
            assert order.items[0].quantity == 2

        finally:
            Path(temp_file).unlink()

    def test_get_schema_not_found(self):
        """Test error when getting non-existent schema."""
        with pytest.raises(ValueError, match="Schema 'NonExistent' not found in registry"):
            self.registry.get_schema("NonExistent")

    def test_resolve_schema_reference_registry_name(self):
        """Test resolving schema reference by registry name."""
        schema_dict = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}

        self.registry.register_schema("TestSchema", schema_dict)

        # Should resolve to registered schema
        resolved_model = self.registry.resolve_schema_reference("TestSchema")
        assert resolved_model is self.registry.get_schema("TestSchema")

    def test_resolve_schema_reference_python_class(self):
        """Test resolving schema reference by Python class path."""
        class_path = "tests.strands.experimental.config_loader.agent.test_schema_registry.UserModel"

        # Should import and return the class directly
        resolved_model = self.registry.resolve_schema_reference(class_path)
        assert resolved_model is UserModel

    def test_list_schemas(self):
        """Test listing all registered schemas."""
        # Register schemas of different types
        self.registry.register_schema("DictSchema", {"type": "object", "properties": {"name": {"type": "string"}}})

        self.registry.register_schema("ClassSchema", UserModel)

        self.registry.register_from_config(
            {
                "name": "ConfigSchema",
                "python_class": "tests.strands.experimental.config_loader.agent.test_schema_registry.UserModel",
            }
        )

        schemas = self.registry.list_schemas()

        assert "DictSchema" in schemas
        assert "ClassSchema" in schemas
        assert "ConfigSchema" in schemas
        assert schemas["DictSchema"] == "programmatic"
        assert schemas["ClassSchema"] == "programmatic"
        assert schemas["ConfigSchema"] == "python_class"

    def test_clear_registry(self):
        """Test clearing all schemas from registry."""
        self.registry.register_schema("TestSchema", UserModel)
        assert "TestSchema" in self.registry.list_schemas()

        self.registry.clear()
        assert len(self.registry.list_schemas()) == 0

        with pytest.raises(ValueError):
            self.registry.get_schema("TestSchema")

    def test_invalid_config_missing_name(self):
        """Test error handling for config missing name."""
        config = {"description": "Missing name", "schema": {"type": "object", "properties": {}}}

        with pytest.raises(ValueError, match="Schema configuration must include 'name' field"):
            self.registry.register_from_config(config)

    def test_invalid_config_missing_schema_definition(self):
        """Test error handling for config missing schema definition."""
        config = {"name": "InvalidSchema", "description": "Missing schema definition"}

        with pytest.raises(ValueError, match="must specify 'schema', 'python_class', or 'schema_file'"):
            self.registry.register_from_config(config)

    def test_invalid_python_class_path(self):
        """Test error handling for invalid Python class path."""
        with pytest.raises(ValueError, match="Cannot import Pydantic class"):
            self.registry.register_schema("Invalid", "non.existent.module.Class")

    def test_non_pydantic_class_path(self):
        """Test error handling for non-Pydantic class."""
        with pytest.raises(ValueError, match="is not a Pydantic BaseModel"):
            self.registry.register_schema("Invalid", "builtins.str")

    def test_missing_schema_file(self):
        """Test error handling for missing schema file."""
        config = {"name": "MissingFile", "schema_file": "/non/existent/file.json"}

        with pytest.raises(FileNotFoundError, match="Schema file not found"):
            self.registry.register_from_config(config)

    def test_invalid_schema_file_format(self):
        """Test error handling for invalid schema file format."""
        # Create temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("not a schema")
            temp_file = f.name

        try:
            config = {"name": "InvalidFormat", "schema_file": temp_file}

            with pytest.raises(ValueError, match="Unsupported schema file format"):
                self.registry.register_from_config(config)

        finally:
            Path(temp_file).unlink()

    def test_malformed_json_file(self):
        """Test error handling for malformed JSON file."""
        # Create temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            temp_file = f.name

        try:
            config = {"name": "MalformedJSON", "schema_file": temp_file}

            with pytest.raises(ValueError, match="Error parsing schema file"):
                self.registry.register_from_config(config)

        finally:
            Path(temp_file).unlink()

    def test_malformed_yaml_file(self):
        """Test error handling for malformed YAML file."""
        # Create temporary file with invalid YAML
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_file = f.name

        try:
            config = {"name": "MalformedYAML", "schema_file": temp_file}

            with pytest.raises(ValueError, match="Error parsing schema file"):
                self.registry.register_from_config(config)

        finally:
            Path(temp_file).unlink()

    def test_register_invalid_schema_type(self):
        """Test error handling for invalid schema type."""
        with pytest.raises(ValueError, match="Schema must be a dict, BaseModel class, or string class path"):
            self.registry.register_schema("Invalid", 123)

    def test_multiple_schemas_same_name(self):
        """Test that registering multiple schemas with same name overwrites."""
        # Register first schema
        schema1 = {"type": "object", "properties": {"field1": {"type": "string"}}, "required": ["field1"]}
        self.registry.register_schema("TestSchema", schema1)

        model1 = self.registry.get_schema("TestSchema")
        instance1 = model1(field1="test")
        assert instance1.field1 == "test"

        # Register second schema with same name
        schema2 = {"type": "object", "properties": {"field2": {"type": "integer"}}, "required": ["field2"]}
        self.registry.register_schema("TestSchema", schema2)

        model2 = self.registry.get_schema("TestSchema")
        instance2 = model2(field2=42)
        assert instance2.field2 == 42

        # Should not be able to create instance with old schema
        with pytest.raises((ValidationError, AttributeError, TypeError)):
            model2(field1="test")
