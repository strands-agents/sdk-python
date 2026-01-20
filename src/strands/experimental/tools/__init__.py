"""Experimental tools package."""

from typing import Any

from .tool_provider import ToolProvider

__all__ = ["ToolProvider"]


def __getattr__(name: str) -> Any:
    """Lazy load optional dependencies.

    LangChainTool requires langchain-core which is an optional dependency.
    """
    if name == "LangChainTool":
        from .langchain_tool import LangChainTool

        return LangChainTool

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
