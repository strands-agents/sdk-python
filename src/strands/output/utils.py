"""Output utility functions."""

from typing import Type, Union, Optional
from pydantic import BaseModel

from .base import OutputMode, OutputSchema
from .modes import ToolMode


def resolve_output_schema(
    structured_output_type: Optional[Union[Type[BaseModel], OutputSchema]] = None,
    output_mode: Optional[OutputMode] = None,
) -> Optional[OutputSchema]:
    """Resolve output type and mode into OutputSchema.

    Args:
        structured_output_type: Output type specification
        output_mode: Output mode (defaults to ToolMode if not specified)

    Returns:
        Resolved OutputSchema or None if no output type specified
    """
    if not structured_output_type:
        return None

    if isinstance(structured_output_type, OutputSchema):
        return structured_output_type

    if output_mode is None:
        output_mode = ToolMode()

    return OutputSchema(structured_output_type, output_mode)
