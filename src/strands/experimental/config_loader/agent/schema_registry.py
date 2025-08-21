"""Schema registry for managing structured output schemas with multiple definition methods."""

import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Type, Union

import yaml
from pydantic import BaseModel

from .pydantic_factory import PydanticModelFactory

logger = logging.getLogger(__name__)


class SchemaRegistry:
    """Registry for managing structured output schemas with multiple definition methods."""

    def __init__(self) -> None:
        """Initialize the schema registry."""
        self._schemas: Dict[str, Type[BaseModel]] = {}
        self._schema_configs: Dict[str, Dict[str, Any]] = {}

    def register_schema(self, name: str, schema: Union[Dict[str, Any], Type[BaseModel], str]) -> None:
        """Register a schema by name with support for multiple input types.

        Args:
            name: Schema name for reference
            schema: Can be:
                - Dict: JSON schema dictionary
                - Type[BaseModel]: Existing Pydantic model class
                - str: Python class path (e.g., "myapp.models.UserProfile")

        Raises:
            ValueError: If schema format is invalid or unsupported
        """
        if isinstance(schema, dict):
            # JSON schema dictionary
            model_class = PydanticModelFactory.create_model_from_schema(name, schema)
            self._schemas[name] = model_class
        elif isinstance(schema, type) and issubclass(schema, BaseModel):
            # Existing Pydantic model class
            self._schemas[name] = schema
        elif isinstance(schema, str):
            # Python class path
            model_class = self._import_python_class(schema)
            self._schemas[name] = model_class
        else:
            raise ValueError(f"Schema must be a dict, BaseModel class, or string class path, got {type(schema)}")

        logger.debug("Registered schema '%s' of type %s", name, type(schema).__name__)

    def register_from_config(self, schema_config: Dict[str, Any]) -> None:
        """Register schema from configuration dictionary.

        Supports:
        - Inline schema definition
        - Python class reference
        - External schema file

        Args:
            schema_config: Schema configuration dictionary with 'name' and one of:
                          'schema', 'python_class', or 'schema_file'

        Raises:
            ValueError: If configuration is invalid or missing required fields
            FileNotFoundError: If external schema file is not found
        """
        name = schema_config.get("name")
        if not name:
            raise ValueError("Schema configuration must include 'name' field")

        # Store the original config for reference
        self._schema_configs[name] = schema_config

        if "schema" in schema_config:
            # Inline schema definition
            schema_dict = schema_config["schema"]
            model_class = PydanticModelFactory.create_model_from_schema(name, schema_dict)
            self._schemas[name] = model_class

        elif "python_class" in schema_config:
            # Python class reference
            class_path = schema_config["python_class"]
            model_class = self._import_python_class(class_path)
            self._schemas[name] = model_class

        elif "schema_file" in schema_config:
            # External schema file
            file_path = schema_config["schema_file"]
            schema_dict = self._load_schema_from_file(file_path)
            model_class = PydanticModelFactory.create_model_from_schema(name, schema_dict)
            self._schemas[name] = model_class

        else:
            raise ValueError(f"Schema '{name}' must specify 'schema', 'python_class', or 'schema_file'")

        logger.info("Registered schema '%s' from configuration", name)

    def get_schema(self, name: str) -> Type[BaseModel]:
        """Get a registered schema by name.

        Args:
            name: Schema name

        Returns:
            Pydantic BaseModel class

        Raises:
            ValueError: If schema is not found in registry
        """
        if name not in self._schemas:
            available_schemas = list(self._schemas.keys())
            raise ValueError(f"Schema '{name}' not found in registry. Available schemas: {available_schemas}")
        return self._schemas[name]

    def resolve_schema_reference(self, reference: str) -> Type[BaseModel]:
        """Resolve a schema reference to a Pydantic model.

        Args:
            reference: Can be:
                - Schema name in registry (e.g., "UserProfile")
                - Direct Python class path (e.g., "myapp.models.UserProfile")

        Returns:
            Pydantic BaseModel class

        Raises:
            ValueError: If reference cannot be resolved
        """
        # Check if it's a Python class reference (contains dots)
        if "." in reference:
            return self._import_python_class(reference)

        # Otherwise, look up in schema registry
        return self.get_schema(reference)

    def list_schemas(self) -> Dict[str, str]:
        """List all registered schemas with their types.

        Returns:
            Dictionary mapping schema names to their source types
        """
        result = {}
        for name, _model_class in self._schemas.items():
            if name in self._schema_configs:
                config = self._schema_configs[name]
                if "schema" in config:
                    result[name] = "inline"
                elif "python_class" in config:
                    result[name] = "python_class"
                elif "schema_file" in config:
                    result[name] = "external_file"
                else:
                    result[name] = "unknown"
            else:
                result[name] = "programmatic"
        return result

    def clear(self) -> None:
        """Clear all registered schemas."""
        self._schemas.clear()
        self._schema_configs.clear()
        logger.debug("Cleared all schemas from registry")

    def _import_python_class(self, class_path: str) -> Type[BaseModel]:
        """Import a Pydantic class from module.Class string.

        Args:
            class_path: Full Python class path (e.g., "myapp.models.UserProfile")

        Returns:
            Pydantic BaseModel class

        Raises:
            ValueError: If class cannot be imported or is not a BaseModel
        """
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)

            # Validate it's a Pydantic model
            if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
                raise ValueError(f"{class_path} is not a Pydantic BaseModel")

            return cls

        except (ImportError, AttributeError, ValueError) as e:
            raise ValueError(f"Cannot import Pydantic class {class_path}: {e}") from e

    def _load_schema_from_file(self, file_path: str) -> Dict[str, Any]:
        """Load schema from JSON or YAML file.

        Args:
            file_path: Path to schema file

        Returns:
            Schema dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {file_path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                if path.suffix.lower() in [".yaml", ".yml"]:
                    data = yaml.safe_load(f)
                    if not isinstance(data, dict):
                        raise ValueError(f"Schema file must contain a dictionary, got {type(data)}")
                    return data
                elif path.suffix.lower() == ".json":
                    data = json.load(f)
                    if not isinstance(data, dict):
                        raise ValueError(f"Schema file must contain a dictionary, got {type(data)}")
                    return data
                else:
                    raise ValueError(f"Unsupported schema file format: {path.suffix}")
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ValueError(f"Error parsing schema file {file_path}: {e}") from e
