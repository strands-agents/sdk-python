"""Tracing type definitions for the SDK."""

from typing import List, Mapping, Optional, Sequence, Union

AttributeValue = Union[
    str,
    bool,
    float,
    int,
    List[str],
    List[bool],
    List[float],
    List[int],
    Sequence[str],
    Sequence[bool],
    Sequence[int],
    Sequence[float],
]

Attributes = Optional[Mapping[str, AttributeValue]]
