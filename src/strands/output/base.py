"""Base classes for output type system."""

from abc import ABC, abstractmethod
from typing import Any, Type, Union, Optional, TYPE_CHECKING
from pydantic import BaseModel
from functools import cached_property


if TYPE_CHECKING:
    from strands.models.model import Model
    from strands.types.tools import ToolSpec


class OutputMode(ABC):
    """Base class for different structured output modes."""

    @abstractmethod
    def get_tool_specs(self, structured_output_type: Type[BaseModel]) -> list["ToolSpec"]:
        """Convert output type to tool specifications.

        Args:
            structured_output_type: Pydantic model type to convert

        Returns:
            List of tool specifications for the output type
        """
        pass

    @abstractmethod
    def is_supported_by_model(self, model: "Model") -> bool:
        """Check if this output mode is supported by the given model.

        Args:
            model: Model instance to check support for

        Returns:
            True if the model supports this output mode
        """
        pass


class OutputSchema:
    """Container for output type information and processing mode."""

    def __init__(
        self,
        type: Type[BaseModel],
        mode: Optional[OutputMode] = None,
    ):
        """Initialize output schema.

        Args:
            type: Pydantic model type for structured output
            mode: Output mode to use (defaults to ToolMode)
        """
        self.type = type
        if mode is None:
            from .modes import ToolMode

            mode = ToolMode()
        self.mode = mode

    @cached_property
    def tool_specs(self) -> list["ToolSpec"]:
        """Get cached tool specifications for this output schema.

        This property computes tool specs once and caches them for reuse,
        avoiding repeated computation for the same output schema type.

        Returns:
            List of tool specifications for the output type
        """
        return self.mode.get_tool_specs(self.type)
