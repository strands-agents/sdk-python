"""Configuration validation utilities for model providers."""

import warnings
from collections.abc import Mapping
from typing import Any

from typing_extensions import get_type_hints

from ..types.content import ContentBlock
from ..types.tools import ToolChoice


def validate_config_keys(config_dict: Mapping[str, Any], config_class: type) -> None:
    """Validate that config keys match the TypedDict fields.

    Args:
        config_dict: Dictionary of configuration parameters
        config_class: TypedDict class to validate against
    """
    valid_keys = set(get_type_hints(config_class).keys())
    provided_keys = set(config_dict.keys())
    invalid_keys = provided_keys - valid_keys

    if invalid_keys:
        warnings.warn(
            f"Invalid configuration parameters: {sorted(invalid_keys)}."
            f"\nValid parameters are: {sorted(valid_keys)}."
            f"\n"
            f"\nSee https://github.com/strands-agents/sdk-python/issues/815",
            stacklevel=4,
        )


def warn_on_tool_choice_not_supported(tool_choice: ToolChoice | None) -> None:
    """Emits a warning if a tool choice is provided but not supported by the provider.

    Args:
        tool_choice: the tool_choice provided to the provider
    """
    if tool_choice:
        warnings.warn(
            "A ToolChoice was provided to this provider but is not supported and will be ignored",
            stacklevel=4,
        )


def _has_location_source(content: ContentBlock) -> bool:
    """Check if a content block contains a location source.

    Providers need to explicitly define an implementation to support content locations.

    Args:
        content: Content block to check.

    Returns:
        True if the content block contains an location source, False otherwise.
    """
    if "image" in content:
        return "location" in content["image"].get("source", {})
    if "document" in content:
        return "location" in content["document"].get("source", {})
    if "video" in content:
        return "location" in content["video"].get("source", {})
    return False
