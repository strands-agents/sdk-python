"""Tests for PydanticModelFactory."""

import pytest
from pydantic import ValidationError

from strands.experimental.config_loader.agent.pydantic_factory import PydanticModelFactory


class TestPydanticModelFactory:
    """Test cases for PydanticModelFactory."""

    def test_simple_string_field(self):
        """Test creating model with simple string field."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "User name"}},
            "required": ["name"],
        }

        UserModel = PydanticModelFactory.create_model_from_schema("User", schema)

        # Test valid data
        user = UserModel(name="John")
        assert user.name == "John"

        # Test validation
        with pytest.raises(ValidationError):
            UserModel()  # Missing required name

    def test_integer_field_with_constraints(self):
        """Test creating model with integer field and constraints."""
        schema = {
            "type": "object",
            "properties": {"age": {"type": "integer", "minimum": 0, "maximum": 150, "description": "User age"}},
            "required": ["age"],
        }

        UserModel = PydanticModelFactory.create_model_from_schema("User", schema)

        # Test valid data
        user = UserModel(age=25)
        assert user.age == 25

        # Test constraints
        with pytest.raises(ValidationError):
            UserModel(age=-1)  # Below minimum

        with pytest.raises(ValidationError):
            UserModel(age=200)  # Above maximum

    def test_optional_fields(self):
        """Test creating model with optional fields."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
            "required": ["name"],
        }

        UserModel = PydanticModelFactory.create_model_from_schema("User", schema)

        # Test with required field only
        user1 = UserModel(name="John")
        assert user1.name == "John"
        assert user1.email is None

        # Test with both fields
        user2 = UserModel(name="Jane", email="jane@example.com")
        assert user2.name == "Jane"
        assert user2.email == "jane@example.com"

    def test_enum_field(self):
        """Test creating model with enum field."""
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["active", "inactive", "pending"], "description": "User status"}
            },
            "required": ["status"],
        }

        UserModel = PydanticModelFactory.create_model_from_schema("User", schema)

        # Test valid enum value
        user = UserModel(status="active")
        assert user.status == "active"

        # Test invalid enum value
        with pytest.raises(ValidationError):
            UserModel(status="invalid")

    def test_array_field(self):
        """Test creating model with array field."""
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5,
                    "description": "User tags",
                }
            },
            "required": ["tags"],
        }

        UserModel = PydanticModelFactory.create_model_from_schema("User", schema)

        # Test valid array
        user = UserModel(tags=["developer", "python"])
        assert user.tags == ["developer", "python"]

        # Test empty array (should fail minItems)
        with pytest.raises(ValidationError):
            UserModel(tags=[])

        # Test too many items
        with pytest.raises(ValidationError):
            UserModel(tags=["a", "b", "c", "d", "e", "f"])

    def test_nested_object(self):
        """Test creating model with nested object."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "zipcode": {"type": "string"},
                    },
                    "required": ["street", "city"],
                },
            },
            "required": ["name", "address"],
        }

        UserModel = PydanticModelFactory.create_model_from_schema("User", schema)

        # Test valid nested object
        user = UserModel(name="John", address={"street": "123 Main St", "city": "Anytown", "zipcode": "12345"})
        assert user.name == "John"
        assert user.address.street == "123 Main St"
        assert user.address.city == "Anytown"
        assert user.address.zipcode == "12345"

        # Test missing required nested field
        with pytest.raises(ValidationError):
            UserModel(
                name="John",
                address={"street": "123 Main St"},  # Missing city
            )

    def test_complex_nested_schema(self):
        """Test creating model with complex nested structures."""
        schema = {
            "type": "object",
            "properties": {
                "user_info": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0}},
                    "required": ["name"],
                },
                "preferences": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"category": {"type": "string"}, "value": {"type": "string"}},
                        "required": ["category", "value"],
                    },
                },
            },
            "required": ["user_info"],
        }

        UserModel = PydanticModelFactory.create_model_from_schema("User", schema)

        # Test complex nested data
        user = UserModel(
            user_info={"name": "John", "age": 30},
            preferences=[{"category": "color", "value": "blue"}, {"category": "theme", "value": "dark"}],
        )

        assert user.user_info.name == "John"
        assert user.user_info.age == 30
        assert len(user.preferences) == 2
        assert user.preferences[0].category == "color"
        assert user.preferences[0].value == "blue"

    def test_string_constraints(self):
        """Test string field constraints."""
        schema = {
            "type": "object",
            "properties": {"username": {"type": "string", "minLength": 3, "maxLength": 20, "description": "Username"}},
            "required": ["username"],
        }

        UserModel = PydanticModelFactory.create_model_from_schema("User", schema)

        # Test valid username
        user = UserModel(username="john_doe123")
        assert user.username == "john_doe123"

        # Test too short
        with pytest.raises(ValidationError):
            UserModel(username="ab")

        # Test too long
        with pytest.raises(ValidationError):
            UserModel(username="a" * 25)

    def test_default_values(self):
        """Test fields with default values."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "status": {"type": "string", "default": "active", "enum": ["active", "inactive"]},
                "count": {"type": "integer", "default": 0},
            },
            "required": ["name"],
        }

        UserModel = PydanticModelFactory.create_model_from_schema("User", schema)

        # Test with defaults
        user = UserModel(name="John")
        assert user.name == "John"
        assert user.status == "active"
        assert user.count == 0

        # Test overriding defaults
        user2 = UserModel(name="Jane", status="inactive", count=5)
        assert user2.status == "inactive"
        assert user2.count == 5

    def test_number_field(self):
        """Test number (float) field."""
        schema = {
            "type": "object",
            "properties": {"price": {"type": "number", "minimum": 0.0, "maximum": 1000.0}},
            "required": ["price"],
        }

        ProductModel = PydanticModelFactory.create_model_from_schema("Product", schema)

        # Test valid price
        product = ProductModel(price=19.99)
        assert product.price == 19.99

        # Test constraints
        with pytest.raises(ValidationError):
            ProductModel(price=-1.0)

        with pytest.raises(ValidationError):
            ProductModel(price=1001.0)

    def test_boolean_field(self):
        """Test boolean field."""
        schema = {"type": "object", "properties": {"is_active": {"type": "boolean"}}, "required": ["is_active"]}

        UserModel = PydanticModelFactory.create_model_from_schema("User", schema)

        # Test boolean values
        user1 = UserModel(is_active=True)
        assert user1.is_active is True

        user2 = UserModel(is_active=False)
        assert user2.is_active is False

    def test_invalid_schema_type(self):
        """Test error handling for invalid schema type."""
        schema = {
            "type": "array",  # Not supported as root type
            "items": {"type": "string"},
        }

        with pytest.raises(ValueError, match="Invalid schema for model"):
            PydanticModelFactory.create_model_from_schema("Invalid", schema)

    def test_schema_validation(self):
        """Test schema validation method."""
        # Valid schema
        valid_schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        assert PydanticModelFactory.validate_schema(valid_schema) is True

        # Invalid schema - not a dict
        assert PydanticModelFactory.validate_schema("not a dict") is False

        # Invalid schema - wrong type
        invalid_schema = {"type": "array", "items": {"type": "string"}}
        assert PydanticModelFactory.validate_schema(invalid_schema) is False

        # Invalid schema - missing type in property
        invalid_schema2 = {
            "type": "object",
            "properties": {
                "name": {"description": "Name"}  # Missing type
            },
        }
        assert PydanticModelFactory.validate_schema(invalid_schema2) is False

    def test_get_schema_info(self):
        """Test schema information extraction."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "status": {"type": "string", "enum": ["active", "inactive"]},
                "tags": {"type": "array", "items": {"type": "string"}},
                "address": {"type": "object", "properties": {"street": {"type": "string"}}},
            },
            "required": ["name", "age"],
        }

        info = PydanticModelFactory.get_schema_info(schema)

        assert info["type"] == "object"
        assert info["properties_count"] == 5
        assert info["required_fields"] == ["name", "age"]
        assert info["has_nested_objects"] is True
        assert info["has_arrays"] is True
        assert info["has_enums"] is True

    def test_error_handling_in_field_processing(self):
        """Test error handling during field processing."""
        # Schema with problematic field that should be handled gracefully
        schema = {
            "type": "object",
            "properties": {
                "good_field": {"type": "string"},
                "problematic_field": {"type": "unknown_type"},  # Unknown type
            },
            "required": ["good_field"],
        }

        # Should still create model, using Any for problematic field
        UserModel = PydanticModelFactory.create_model_from_schema("User", schema)

        # Should work with good field
        user = UserModel(good_field="test", problematic_field="anything")
        assert user.good_field == "test"
        assert user.problematic_field == "anything"

    def test_format_constraints(self):
        """Test format constraints like date-time."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "created_at": {"type": "string", "format": "date-time"}},
            "required": ["name"],
        }

        UserModel = PydanticModelFactory.create_model_from_schema("User", schema)

        # Test with valid data
        user = UserModel(name="test user", created_at="2024-01-01T12:00:00Z")

        # Basic validation should work
        assert user.name == "test user"
