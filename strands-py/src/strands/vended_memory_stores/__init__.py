"""Vended memory stores for Strands Agents.

Concrete :class:`~strands.memory.types.MemoryStore` backends shipped with the SDK. A store may be
imported from here or from its subpackage, e.g.
``from strands.vended_memory_stores import BedrockKnowledgeBaseStore``.
"""

from typing import Any

__all__ = [
    "BedrockKnowledgeBaseStore",
]


def __getattr__(name: str) -> Any:
    """Lazy load store implementations only when accessed.

    This defers the import of optional dependencies until actually needed.
    """
    if name == "BedrockKnowledgeBaseStore":
        from .bedrock_knowledge_base import BedrockKnowledgeBaseStore

        return BedrockKnowledgeBaseStore
    raise AttributeError(f"cannot import name '{name}' from '{__name__}' ({__file__})")
