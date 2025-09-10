"""Configuration validation utilities for model providers."""

import warnings
from typing import Any, Mapping, Type

from typing_extensions import get_type_hints

from ..types.tools import ToolChoice


def validate_config_keys(config_dict: Mapping[str, Any], config_class: Type) -> None:
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
