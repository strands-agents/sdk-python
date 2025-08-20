"""Error handling classes for structured output configuration."""

from typing import Any, Dict, List, Optional


class StructuredOutputError(Exception):
    """Base exception for structured output errors."""

    def __init__(self, message: str, schema_name: Optional[str] = None, agent_name: Optional[str] = None):
        """Initialize the error.

        Args:
            message: Error message
            schema_name: Name of the schema that caused the error
            agent_name: Name of the agent that caused the error
        """
        super().__init__(message)
        self.schema_name = schema_name
        self.agent_name = agent_name


class SchemaValidationError(StructuredOutputError):
    """Raised when schema validation fails."""

    def __init__(self, message: str, schema_name: Optional[str] = None, validation_errors: Optional[List[Any]] = None):
        """Initialize the error.

        Args:
            message: Error message
            schema_name: Name of the schema that failed validation
            validation_errors: List of specific validation errors
        """
        super().__init__(message, schema_name)
        self.validation_errors = validation_errors or []


class ModelCreationError(StructuredOutputError):
    """Raised when Pydantic model creation fails."""

    def __init__(self, message: str, schema_name: Optional[str] = None, schema_dict: Optional[Dict[Any, Any]] = None):
        """Initialize the error.

        Args:
            message: Error message
            schema_name: Name of the schema that failed to create
            schema_dict: The schema dictionary that caused the error
        """
        super().__init__(message, schema_name)
        self.schema_dict = schema_dict


class OutputValidationError(StructuredOutputError):
    """Raised when model output validation fails."""

    def __init__(self, message: str, schema_name: Optional[str] = None, output_data: Optional[Dict[Any, Any]] = None):
        """Initialize the error.

        Args:
            message: Error message
            schema_name: Name of the schema that failed validation
            output_data: The output data that failed validation
        """
        super().__init__(message, schema_name)
        self.output_data = output_data


class SchemaRegistryError(StructuredOutputError):
    """Raised when schema registry operations fail."""

    pass


class SchemaImportError(StructuredOutputError):
    """Raised when importing Python classes for schemas fails."""

    def __init__(self, message: str, class_path: Optional[str] = None):
        """Initialize the error.

        Args:
            message: Error message
            class_path: The Python class path that failed to import
        """
        super().__init__(message)
        self.class_path = class_path
