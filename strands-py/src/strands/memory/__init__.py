"""Memory module for Strands Agents.

This package ports the Strands TypeScript ``memory`` module to Python. It gives
agents cross-session recall and persistence through a ``MemoryManager`` plugin
that manages pluggable memory stores, exposes search/add tools, and runs
automatic background extraction.

The public surface mirrors the TypeScript ``src/memory/index.ts`` exports,
adapted to Python ``snake_case`` naming.
"""

from ..types.exceptions import AggregateMemoryError
from .extraction.model_extractor import ModelExtractor
from .extraction.triggers import IntervalTrigger, InvocationTrigger
from .extraction.types import (
    ExtractionConfig,
    ExtractionResult,
    ExtractionTrigger,
    ExtractionTriggerContext,
    Extractor,
    ExtractorContext,
    MemoryContentBlockType,
    MemoryMessageFilter,
)
from .memory_manager import MemoryManager
from .types import (
    AddMessagesContext,
    MemoryAddOptions,
    MemoryAddToolConfig,
    MemoryEntry,
    MemoryManagerConfig,
    MemorySearchOptions,
    MemoryStore,
    MemoryStoreConfig,
    MemoryToolConfig,
    SearchOptions,
)

__all__ = [
    "AddMessagesContext",
    "AggregateMemoryError",
    "ExtractionConfig",
    "ExtractionResult",
    "ExtractionTrigger",
    "ExtractionTriggerContext",
    "Extractor",
    "ExtractorContext",
    "IntervalTrigger",
    "InvocationTrigger",
    "MemoryAddOptions",
    "MemoryAddToolConfig",
    "MemoryContentBlockType",
    "MemoryEntry",
    "MemoryManager",
    "MemoryManagerConfig",
    "MemoryMessageFilter",
    "MemorySearchOptions",
    "MemoryStore",
    "MemoryStoreConfig",
    "MemoryToolConfig",
    "ModelExtractor",
    "SearchOptions",
]
