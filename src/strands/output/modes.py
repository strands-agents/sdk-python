"""Concrete output mode implementations."""

from typing import Any, Type, Optional, TYPE_CHECKING
from pydantic import BaseModel

from .base import OutputMode
from strands.tools.structured_output.structured_output_tool import StructuredOutputTool

if TYPE_CHECKING:
    from strands.models.model import Model
    from strands.types.tools import ToolSpec


class ToolMode(OutputMode):
    """Use function calling for structured output (DEFAULT).

    This is the most reliable approach across all model providers and ensures
    consistent behavior regardless of model capabilities.
    """

    def get_tool_specs(self, structured_output_model: Type[BaseModel]) -> list["ToolSpec"]:
        """Convert Pydantic model to tool specifications."""
        return [StructuredOutputTool(structured_output_model).tool_spec]

    def get_tool_instances(self, structured_output_model: Type[BaseModel]) -> list["StructuredOutputTool"]:
        """Create actual tool instances for structured output.

        Args:
            structured_output_model: The Pydantic model class to create tools for.

        Returns:
            List containing a single StructuredOutputTool instance.
        """
        return [StructuredOutputTool(structured_output_model)]

    def is_supported_by_model(self, model: "Model") -> bool:
        """Tool-based output is supported by all models that support function calling."""
        return True  # All our models support function calling
