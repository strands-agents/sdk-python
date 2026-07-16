"""Experimental features.

This module implements experimental features that are subject to change in future revisions without notice.
"""

from typing import Any

from . import checkpoint, steering, tools

__all__ = ["checkpoint", "config_to_agent", "tools", "steering"]


def __getattr__(name: str) -> Any:
    """Lazy load config_to_agent only when accessed.

    This defers the import of jsonschema until actually needed.
    """
    if name == "config_to_agent":
        from .agent_config import config_to_agent

        return config_to_agent
    raise AttributeError(f"cannot import name '{name}' from '{__name__}' ({__file__})")
