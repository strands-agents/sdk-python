"""User-maintained model metadata for classifier-driven routing."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Mapping
from os import PathLike
from pathlib import Path
from types import MappingProxyType

import yaml
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver

__all__ = ["ModelCatalog"]

_MODEL_CATALOG_MAXIMUM_BYTES = 1_000_000
_MODEL_CATALOG_MAXIMUM_MODELS = 1_000
_MODEL_PROFILE_MAXIMUM_FIELDS = 50
_MODEL_PROFILE_MAXIMUM_SERIALIZED_CHARACTERS = 1_000
_MODEL_IDENTIFIER_MAXIMUM_CHARACTERS = 500
_MODEL_METADATA_STRING_MAXIMUM_CHARACTERS = 100
_STRING_FIELDS = {"litellm_provider", "mode"}
_TOKEN_LIMIT_FIELDS = {"max_tokens", "max_input_tokens", "max_output_tokens", "prompt_cache_min_tokens"}
_COST_FIELDS = {
    "input_cost_per_token",
    "output_cost_per_token",
    "cache_read_input_token_cost",
    "cache_creation_input_token_cost",
}
_TIERED_COST_FIELD = re.compile(
    r"^(?:input_cost_per_token|output_cost_per_token|cache_read_input_token_cost|cache_creation_input_token_cost)"
    r"_above_[1-9]\d*[km]?_tokens$"
)
_SUPPORT_FIELD = re.compile(r"^supports_[a-z][a-z0-9_]{0,99}$")


class ModelCatalog(Mapping[str, Mapping[str, object]]):
    """An immutable snapshot of objective model metadata used for routing.

    Catalog keys are exact model identifiers. Metadata uses LiteLLM-compatible names and
    contains only bounded scalar values; provider credentials and connection settings are
    never accepted as native catalog metadata.
    """

    def __init__(self, models: Mapping[str, Mapping[str, object]] | None = None) -> None:
        """Create a validated catalog from model metadata.

        Args:
            models: Mapping from exact model identifiers to objective metadata.

        Raises:
            TypeError: If the catalog shape or a metadata value has the wrong type.
            ValueError: If an identifier, field, or value is unsupported or out of bounds.
        """
        self._models = _validate_models(models if models is not None else {})

    @classmethod
    def from_file(cls, path: str | PathLike[str]) -> ModelCatalog:
        """Load a versioned native catalog from a JSON or YAML file.

        Args:
            path: Path ending in ``.json``, ``.yaml``, or ``.yml``.

        Returns:
            A validated immutable catalog snapshot.

        Raises:
            OSError: If the catalog cannot be read.
            TypeError: If the catalog shape or a metadata value has the wrong type.
            ValueError: If the file or catalog values are invalid.
        """
        document = _read_document(path)
        if not isinstance(document, Mapping):
            raise TypeError("model catalog root must be a mapping")
        if set(document) != {"version", "models"}:
            raise ValueError("model catalog must contain only version and models")
        version = document["version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise TypeError("model catalog version must be an integer")
        if version != 1:
            raise ValueError("model catalog version must be 1")
        models = document["models"]
        if not isinstance(models, Mapping):
            raise TypeError("model catalog models must be a mapping")
        return cls(models)

    @classmethod
    def from_litellm_config(cls, path: str | PathLike[str]) -> ModelCatalog:
        """Extract safe routing metadata from a LiteLLM proxy JSON or YAML config.

        Provider credentials, endpoints, headers, and unsupported parameters are ignored.
        Extracted profiles are addressable by the LiteLLM alias, proxy-prefixed alias, and
        configured provider model identifier. Entries without objective metadata are skipped.

        Args:
            path: Path to a LiteLLM proxy JSON or YAML configuration.

        Returns:
            A validated immutable catalog snapshot.

        Raises:
            OSError: If the configuration cannot be read.
            TypeError: If required LiteLLM configuration values have the wrong type.
            ValueError: If the configuration does not provide usable model metadata.
        """
        document = _read_document(path)
        if not isinstance(document, Mapping):
            raise TypeError("LiteLLM config root must be a mapping")
        model_list = document.get("model_list")
        if not isinstance(model_list, list):
            raise TypeError("LiteLLM config model_list must be a list")

        models: dict[str, dict[str, object]] = {}
        for model_index, model_entry in enumerate(model_list):
            _add_litellm_model(models, model_entry, model_index)
        if not models:
            raise ValueError("LiteLLM config contains no supported model metadata")
        return cls(models)

    def with_overrides(self, overrides: Mapping[str, Mapping[str, object]]) -> ModelCatalog:
        """Return a new catalog with validated per-model metadata overrides.

        Existing instances remain unchanged so active routing behavior cannot change mid-invocation.

        Args:
            overrides: Metadata to add or replace by exact model identifier.

        Returns:
            A new immutable catalog snapshot.

        Raises:
            TypeError: If the override shape or a metadata value has the wrong type.
            ValueError: If an identifier, field, or value is unsupported or out of bounds.
        """
        if not isinstance(overrides, Mapping):
            raise TypeError("model catalog overrides must be a mapping")
        validated_overrides = _validate_models(overrides)
        merged = {model_id: dict(profile) for model_id, profile in self._models.items()}
        for model_id, profile in validated_overrides.items():
            merged[model_id] = {**merged.get(model_id, {}), **profile}
        return ModelCatalog(merged)

    def as_dict(self) -> dict[str, dict[str, object]]:
        """Return a mutable copy suitable for serialization."""
        return {model_id: dict(profile) for model_id, profile in self._models.items()}

    def __getitem__(self, model_id: str) -> Mapping[str, object]:
        """Return immutable metadata for an exact model identifier."""
        return self._models[model_id]

    def __iter__(self) -> Iterator[str]:
        """Iterate over exact model identifiers in declaration order."""
        return iter(self._models)

    def __len__(self) -> int:
        """Return the number of exact model identifiers in the catalog."""
        return len(self._models)


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    """Construct one YAML mapping while rejecting duplicate or unhashable keys."""
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise ValueError("YAML mapping keys must be scalar values") from error
        if duplicate:
            raise ValueError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _read_document(path: str | PathLike[str]) -> object:
    """Read one bounded UTF-8 JSON or YAML document."""
    document_path = Path(path)
    if document_path.stat().st_size > _MODEL_CATALOG_MAXIMUM_BYTES:
        raise ValueError("model catalog must not exceed 1000000 bytes")
    document_bytes = document_path.read_bytes()
    if len(document_bytes) > _MODEL_CATALOG_MAXIMUM_BYTES:
        raise ValueError("model catalog must not exceed 1000000 bytes")

    try:
        document_text = document_bytes.decode("utf-8")
        if document_path.suffix.lower() == ".json":
            return json.loads(document_text, object_pairs_hook=_reject_duplicate_json_keys)
        if document_path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.load(document_text, Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError, ValueError) as error:
        raise ValueError(f"invalid model catalog: {error}") from error
    raise ValueError("model catalog path must end in .json, .yaml, or .yml")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting ambiguous duplicate keys."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_models(models: object) -> Mapping[str, Mapping[str, object]]:
    """Validate and deeply freeze catalog models."""
    if not isinstance(models, Mapping):
        raise TypeError("model catalog models must be a mapping")
    if len(models) > _MODEL_CATALOG_MAXIMUM_MODELS:
        raise ValueError("model catalog must not contain more than 1000 models")
    validated: dict[str, Mapping[str, object]] = {}
    for model_id, profile in models.items():
        _validate_model_identifier(model_id)
        if not isinstance(profile, Mapping):
            raise TypeError(f"metadata for model {model_id!r} must be a mapping")
        validated[model_id] = MappingProxyType(_validate_profile(model_id, profile))
    return MappingProxyType(validated)


def _validate_model_identifier(model_id: object) -> None:
    """Validate one exact model identifier."""
    if not isinstance(model_id, str):
        raise TypeError("model catalog identifiers must be strings")
    if not model_id or model_id != model_id.strip():
        raise ValueError("model catalog identifiers must be non-empty without surrounding whitespace")
    if len(model_id) > _MODEL_IDENTIFIER_MAXIMUM_CHARACTERS:
        raise ValueError("model catalog identifiers must not exceed 500 characters")


def _validate_profile(model_id: str, profile: Mapping[str, object]) -> dict[str, object]:
    """Validate one objective model metadata profile."""
    if not profile:
        raise ValueError(f"metadata for model {model_id!r} must not be empty")
    if len(profile) > _MODEL_PROFILE_MAXIMUM_FIELDS:
        raise ValueError(f"metadata for model {model_id!r} must not contain more than 50 fields")

    validated: dict[str, object] = {}
    for field_name, value in profile.items():
        if not isinstance(field_name, str):
            raise TypeError(f"metadata field names for model {model_id!r} must be strings")
        validated[field_name] = _validate_metadata_value(model_id, field_name, value)

    serialized = json.dumps(validated, separators=(",", ":"))
    if len(serialized) > _MODEL_PROFILE_MAXIMUM_SERIALIZED_CHARACTERS:
        raise ValueError(f"metadata for model {model_id!r} must not exceed 1000 serialized characters")
    return validated


def _validate_metadata_value(model_id: str, field_name: str, value: object) -> object:
    """Validate one allowlisted scalar metadata value."""
    if field_name in _STRING_FIELDS:
        if not isinstance(value, str):
            raise TypeError(f"metadata field {field_name!r} for model {model_id!r} must be a string")
        if not value or len(value) > _MODEL_METADATA_STRING_MAXIMUM_CHARACTERS:
            raise ValueError(f"metadata field {field_name!r} for model {model_id!r} must contain 1 to 100 characters")
        return value
    if field_name in _TOKEN_LIMIT_FIELDS:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"metadata field {field_name!r} for model {model_id!r} must be an integer")
        if value <= 0:
            raise ValueError(f"metadata field {field_name!r} for model {model_id!r} must be greater than zero")
        return value
    if field_name in _COST_FIELDS or _TIERED_COST_FIELD.fullmatch(field_name):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"metadata field {field_name!r} for model {model_id!r} must be numeric")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"metadata field {field_name!r} for model {model_id!r} must be finite and non-negative")
        return value
    if _SUPPORT_FIELD.fullmatch(field_name):
        if not isinstance(value, bool):
            raise TypeError(f"metadata field {field_name!r} for model {model_id!r} must be boolean")
        return value
    raise ValueError(f"unsupported metadata field {field_name!r} for model {model_id!r}")


def _add_litellm_model(models: dict[str, dict[str, object]], model_entry: object, model_index: int) -> None:
    """Extract one safe profile and its exact identifiers from a LiteLLM config entry."""
    if not isinstance(model_entry, Mapping):
        raise TypeError(f"LiteLLM model_list[{model_index}] must be a mapping")
    model_name = model_entry.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise TypeError(f"LiteLLM model_list[{model_index}].model_name must be a non-empty string")
    litellm_params = model_entry.get("litellm_params")
    if not isinstance(litellm_params, Mapping):
        raise TypeError(f"LiteLLM model_list[{model_index}].litellm_params must be a mapping")
    model_info = model_entry.get("model_info", {})
    if not isinstance(model_info, Mapping):
        raise TypeError(f"LiteLLM model_list[{model_index}].model_info must be a mapping")

    profile = _extract_supported_metadata(model_info)
    profile.update(_extract_supported_metadata(litellm_params))
    if not profile:
        return

    identifiers = [model_name, f"litellm_proxy/{model_name}"]
    provider_model = litellm_params.get("model")
    if isinstance(provider_model, str) and provider_model:
        identifiers.append(provider_model)
    for model_id in dict.fromkeys(identifiers):
        existing = models.get(model_id)
        if existing is not None and existing != profile:
            raise ValueError(f"LiteLLM config defines conflicting metadata for model {model_id!r}")
        models[model_id] = dict(profile)


def _extract_supported_metadata(source: Mapping[object, object]) -> dict[str, object]:
    """Copy only supported objective metadata from a provider configuration mapping."""
    result: dict[str, object] = {}
    for field_name, value in source.items():
        if not isinstance(field_name, str):
            continue
        if (
            field_name in _STRING_FIELDS
            or field_name in _TOKEN_LIMIT_FIELDS
            or field_name in _COST_FIELDS
            or _TIERED_COST_FIELD.fullmatch(field_name)
            or _SUPPORT_FIELD.fullmatch(field_name)
        ):
            result[field_name] = value
    return result
