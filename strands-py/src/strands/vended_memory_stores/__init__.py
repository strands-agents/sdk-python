"""Vended memory stores for Strands Agents.

Concrete :class:`~strands.memory.types.MemoryStore` backends shipped with the SDK. A store may be
imported from here or from its subpackage, e.g.
``from strands.vended_memory_stores import BedrockKnowledgeBaseStore``.
"""

import warnings
from typing import Any

__all__ = [
    "BedrockKnowledgeBaseStore",
    "JsonMemoryStore",
]


def __getattr__(name: str) -> Any:
    """Lazy load store implementations only when accessed.

    This defers the import of optional dependencies until actually needed.
    """
    if name == "BedrockKnowledgeBaseStore":
        from .bedrock_knowledge_base import BedrockKnowledgeBaseStore

        return BedrockKnowledgeBaseStore
    if name == "JsonMemoryStore":
        from .local import JsonMemoryStore

        return JsonMemoryStore
    if name == "LocalMemoryStore":
        from .local import JsonMemoryStore

        warnings.warn(
            "LocalMemoryStore has been renamed to JsonMemoryStore. "
            "Use JsonMemoryStore from strands.vended_memory_stores instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return JsonMemoryStore
    raise AttributeError(f"cannot import name '{name}' from '{__name__}' ({__file__})")
