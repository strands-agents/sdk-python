"""Load user-maintained model metadata for input-complexity routing."""

from __future__ import annotations

import json
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, field_validator, model_validator

_MODEL_CATALOG_MAXIMUM_BYTES = 1_000_000
_ModelProfileKey = Annotated[
    str,
    StringConstraints(min_length=1, max_length=500, pattern=r"^\S(?:.*\S)?$"),
]
_ModelProfileText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000)]
_ModelCapability = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]


class _ModelRoutingProfile(BaseModel):
    """Validated model metadata that may be sent to the classifier model."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    description: _ModelProfileText | None = None
    input_cost_per_million_tokens: float | None = Field(default=None, ge=0)
    output_cost_per_million_tokens: float | None = Field(default=None, ge=0)
    relative_latency: Literal["low", "medium", "high"] | None = None
    context_window_tokens: int | None = Field(default=None, gt=0, strict=True)
    capabilities: list[_ModelCapability] = Field(default_factory=list, max_length=20)
    limitations: list[_ModelCapability] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def reject_empty_profile(self) -> _ModelRoutingProfile:
        """Reject catalog entries that provide no routing information."""
        if not self.model_dump(exclude_none=True, exclude_defaults=True):
            raise ValueError("model profile must contain routing metadata")
        return self


class _ModelRoutingCatalog(BaseModel):
    """Versioned on-disk model-routing catalog."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: int = Field(strict=True)
    models: dict[_ModelProfileKey, _ModelRoutingProfile] = Field(max_length=1_000)

    @field_validator("version")
    @classmethod
    def require_supported_version(cls, version: int) -> int:
        """Reject catalog versions that this SDK does not understand."""
        if version != 1:
            raise ValueError("model catalog version must be 1")
        return version


def load_model_catalog(path: str | PathLike[str]) -> Mapping[str, _ModelRoutingProfile]:
    """Load and validate a model-routing catalog once.

    Args:
        path: Path to a UTF-8 JSON catalog.

    Returns:
        An immutable mapping of profile keys to validated routing profiles.

    Raises:
        OSError: If the catalog cannot be read.
        ValueError: If the file is too large or does not match the supported schema.
    """
    catalog_path = Path(path)
    if catalog_path.stat().st_size > _MODEL_CATALOG_MAXIMUM_BYTES:
        raise ValueError("model catalog must not exceed 1000000 bytes")

    catalog_bytes = catalog_path.read_bytes()
    if len(catalog_bytes) > _MODEL_CATALOG_MAXIMUM_BYTES:
        raise ValueError("model catalog must not exceed 1000000 bytes")

    try:
        catalog_data = json.loads(catalog_bytes.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
        catalog = _ModelRoutingCatalog.model_validate(catalog_data)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError) as error:
        raise ValueError(f"invalid model catalog: {error}") from error

    return MappingProxyType(dict(catalog.models))


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting ambiguous duplicate keys."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
