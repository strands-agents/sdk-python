"""Vended memory stores for Strands Agents.

Concrete :class:`~strands.memory.types.MemoryStore` backends shipped with the SDK. A store may be
imported from here or from its subpackage, e.g.
``from strands.vended_memory_stores import BedrockKnowledgeBaseStore``.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .bedrock_knowledge_base import (
        BedrockKnowledgeBaseAddResult,
        BedrockKnowledgeBaseConfig,
        BedrockKnowledgeBaseS3Config,
        BedrockKnowledgeBaseStore,
        BedrockKnowledgeBaseStoreConfig,
    )

__all__ = [
    "BedrockKnowledgeBaseAddResult",
    "BedrockKnowledgeBaseConfig",
    "BedrockKnowledgeBaseS3Config",
    "BedrockKnowledgeBaseStore",
    "BedrockKnowledgeBaseStoreConfig",
]

_BEDROCK_KNOWLEDGE_BASE_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    """Lazily import store backends only when accessed.

    This defers each store's optional dependencies (e.g. ``boto3`` for the Bedrock Knowledge Base
    store) until one of its names is actually used.
    """
    if name in _BEDROCK_KNOWLEDGE_BASE_EXPORTS:
        from . import bedrock_knowledge_base

        return getattr(bedrock_knowledge_base, name)
    raise AttributeError(f"cannot import name '{name}' from '{__name__}' ({__file__})")
