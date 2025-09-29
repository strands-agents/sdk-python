"""Type definitions for output system."""

from typing import Type, Union, Optional
from pydantic import BaseModel

from strands.output.base import OutputMode, OutputSchema

# Type aliases for output type specifications
OutputType = Union[Type[BaseModel], list[Type[BaseModel]], OutputSchema]
OptionalOutputType = Optional[OutputType]

__all__ = [
    "OutputType",
    "OptionalOutputType",
]
