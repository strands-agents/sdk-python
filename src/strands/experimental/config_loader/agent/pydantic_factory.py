"""Factory for creating Pydantic models from JSON schema dictionaries."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import BaseModel, Field, conlist, create_model

logger = logging.getLogger(__name__)


class PydanticModelFactory:
    """Factory for creating Pydantic models from JSON schema dictionaries."""

    @staticmethod
    def create_model_from_schema(
        model_name: str, schema: Dict[str, Any], base_class: Type[BaseModel] = BaseModel
    ) -> Type[BaseModel]:
        """Create a Pydantic BaseModel from a JSON schema dictionary.

        Args:
            model_name: Name for the generated model class
            schema: JSON schema dictionary
            base_class: Base class to inherit from (default: BaseModel)

        Returns:
            Generated Pydantic BaseModel class

        Raises:
            ValueError: If schema is invalid or unsupported
        """
        if not isinstance(schema, dict):
            raise ValueError(f"Schema must be a dictionary, got {type(schema)}")

        if schema.get("type") != "object":
            raise ValueError(
                f"Invalid schema for model '{model_name}': root type must be 'object', got {schema.get('type')}"
            )

        properties = schema.get("properties", {})
        required_fields = set(schema.get("required", []))

        if not properties:
            logger.warning("Schema '%s' has no properties defined", model_name)

        # Build field definitions for create_model
        field_definitions: Dict[str, Any] = {}

        for field_name, field_schema in properties.items():
            try:
                is_required = field_name in required_fields
                field_type, field_info = PydanticModelFactory._process_field_schema(
                    field_name, field_schema, is_required, model_name
                )
                field_definitions[field_name] = (field_type, field_info)
            except Exception as e:
                logger.warning("Error processing field '%s' in schema '%s': %s", field_name, model_name, e)
                # Use Any type as fallback
                fallback_type = Optional[Any] if field_name not in required_fields else Any
                field_definitions[field_name] = (
                    fallback_type,
                    Field(description=f"Field processing failed: {e}"),
                )

        # Create the model
        try:
            model_class = create_model(model_name, __base__=base_class, **field_definitions)
            return model_class
        except Exception as e:
            raise ValueError(f"Failed to create model '{model_name}': {e}") from e

    @staticmethod
    def _process_field_schema(
        field_name: str, field_schema: Dict[str, Any], is_required: bool, parent_model_name: str = ""
    ) -> tuple[Type[Any], Any]:
        """Process a single field schema into Pydantic field type and info.

        Args:
            field_name: Name of the field
            field_schema: JSON schema for the field
            is_required: Whether the field is required
            parent_model_name: Name of the parent model for nested object naming

        Returns:
            Tuple of (field_type, field_info)
        """
        field_type = PydanticModelFactory._get_python_type(field_schema, field_name, parent_model_name)

        # Create Field with metadata
        field_kwargs = {}

        if "description" in field_schema:
            field_kwargs["description"] = field_schema["description"]

        if "default" in field_schema:
            field_kwargs["default"] = field_schema["default"]
        elif not is_required:
            field_kwargs["default"] = None

        # Add validation constraints
        if "minimum" in field_schema:
            field_kwargs["ge"] = field_schema["minimum"]
        if "maximum" in field_schema:
            field_kwargs["le"] = field_schema["maximum"]
        if "minLength" in field_schema:
            field_kwargs["min_length"] = field_schema["minLength"]
        if "maxLength" in field_schema:
            field_kwargs["max_length"] = field_schema["maxLength"]
        if "pattern" in field_schema:
            field_kwargs["pattern"] = field_schema["pattern"]

        # Handle array constraints
        if field_schema.get("type") == "array":
            min_items = field_schema.get("minItems")
            max_items = field_schema.get("maxItems")
            if min_items is not None or max_items is not None:
                # Use conlist for array constraints
                item_type = PydanticModelFactory._get_array_item_type(field_schema, field_name, parent_model_name)
                field_type = conlist(item_type, min_length=min_items, max_length=max_items)

        # Handle format constraints
        if "format" in field_schema:
            format_type = field_schema["format"]
            if format_type == "email":
                try:
                    from pydantic import EmailStr

                    field_type = EmailStr
                except ImportError:
                    logger.warning("EmailStr not available, using str for email field '%s'", field_name)
                    field_type = str
            elif format_type == "uri":
                try:
                    from pydantic import HttpUrl

                    field_type = HttpUrl
                except ImportError:
                    logger.warning("HttpUrl not available, using str for uri field '%s'", field_name)
                    field_type = str
            elif format_type == "date-time":
                field_type = datetime

        # Handle optional fields after all type processing
        if not is_required:
            field_type = Optional[field_type]  # type: ignore[assignment]

        field_info = Field(**field_kwargs) if field_kwargs else Field()

        return field_type, field_info

    @staticmethod
    def _get_array_item_type(schema: Dict[str, Any], field_name: str = "", parent_model_name: str = "") -> Type[Any]:
        """Get the item type for an array schema."""
        items_schema = schema.get("items", {})
        if items_schema:
            return PydanticModelFactory._get_python_type(items_schema, field_name, parent_model_name)
        else:
            return Any

    @staticmethod
    def _get_python_type(schema: Dict[str, Any], field_name: str = "", parent_model_name: str = "") -> Type[Any]:
        """Convert JSON schema type to Python type.

        Args:
            schema: JSON schema dictionary
            field_name: Name of the field (for nested object naming)
            parent_model_name: Name of the parent model (for nested object naming)

        Returns:
            Python type corresponding to the schema
        """
        schema_type = schema.get("type")

        if schema_type == "string":
            # Handle enum constraints
            if "enum" in schema:
                enum_values = schema["enum"]
                # Use Literal for string enums to preserve string values
                if len(enum_values) == 1:
                    return Literal[enum_values[0]]  # type: ignore[return-value]
                else:
                    return Literal[tuple(enum_values)]  # type: ignore[return-value]
            return str
        elif schema_type == "integer":
            return int
        elif schema_type == "number":
            return float
        elif schema_type == "boolean":
            return bool
        elif schema_type == "array":
            items_schema = schema.get("items", {})
            if items_schema:
                item_type = PydanticModelFactory._get_python_type(items_schema, field_name, parent_model_name)
                return List[item_type]  # type: ignore[valid-type]
            else:
                return List[Any]
        elif schema_type == "object":
            # For nested objects, create a nested model
            nested_model_name = (
                f"{parent_model_name}{field_name.title()}"
                if parent_model_name and field_name
                else f"NestedObject{field_name.title()}"
            )
            return PydanticModelFactory.create_model_from_schema(nested_model_name, schema)
        elif schema_type is None and "anyOf" in schema:
            # Handle anyOf by creating Union types
            types = []
            for sub_schema in schema["anyOf"]:
                sub_type = PydanticModelFactory._get_python_type(sub_schema, field_name, parent_model_name)
                types.append(sub_type)
            if len(types) == 1:
                return types[0]
            elif len(types) == 2 and type(None) in types:
                # This is Optional[T]
                non_none_type = next(t for t in types if t is not type(None))
                return Optional[non_none_type]  # type: ignore[return-value]
            else:
                return Union[tuple(types)]  # type: ignore[return-value]
        else:
            logger.warning("Unknown schema type '%s', using Any", schema_type)
            return Any

    @staticmethod
    def validate_schema(schema: Any) -> bool:
        """Validate if a schema is valid for model creation.

        Args:
            schema: Schema to validate

        Returns:
            True if schema is valid, False otherwise
        """
        try:
            if not isinstance(schema, dict):
                return False

            if schema.get("type") != "object":
                return False

            # Check properties have valid types
            properties = schema.get("properties", {})
            for _, prop_schema in properties.items():
                if not isinstance(prop_schema, dict):
                    return False
                if "type" not in prop_schema:
                    return False

            return True
        except Exception:
            return False

    @staticmethod
    def get_schema_info(schema: Dict[str, Any]) -> Dict[str, Any]:
        """Get schema information from a JSON schema dictionary.

        Args:
            schema: JSON schema dictionary

        Returns:
            Dictionary containing schema information
        """
        try:
            properties = schema.get("properties", {})
            required_fields = schema.get("required", [])

            # Analyze schema features
            has_nested_objects = any(prop.get("type") == "object" for prop in properties.values())
            has_arrays = any(prop.get("type") == "array" for prop in properties.values())
            has_enums = any("enum" in prop for prop in properties.values())

            return {
                "type": schema.get("type", "unknown"),
                "properties_count": len(properties),
                "required_fields": required_fields,
                "has_nested_objects": has_nested_objects,
                "has_arrays": has_arrays,
                "has_enums": has_enums,
            }
        except Exception as e:
            logger.error("Failed to get schema info: %s", e)
            return {
                "type": "unknown",
                "properties_count": 0,
                "required_fields": [],
                "has_nested_objects": False,
                "has_arrays": False,
                "has_enums": False,
                "error": str(e),
            }

    @staticmethod
    def get_model_schema_info(model_class: Type[BaseModel]) -> Dict[str, Any]:
        """Get schema information from a Pydantic model.

        Args:
            model_class: Pydantic model class

        Returns:
            Dictionary containing schema information
        """
        try:
            schema = model_class.model_json_schema()
            return {
                "name": model_class.__name__,
                "schema": schema,
                "fields": list(schema.get("properties", {}).keys()),
                "required": schema.get("required", []),
            }
        except Exception as e:
            logger.error("Failed to get schema info for model '%s': %s", model_class.__name__, e)
            return {
                "name": model_class.__name__,
                "schema": {},
                "fields": [],
                "required": [],
                "error": str(e),
            }
