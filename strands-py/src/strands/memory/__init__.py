"""Memory module.

This module provides cross-session memory management for agents. The
:class:`MemoryManager` manages one or more :class:`MemoryStore` backends and exposes
``search_memory`` / ``add_memory`` tools for agent-driven recall and persistence,
plus programmatic ``search`` / ``add`` methods.

Example:
    ```python
    from strands import Agent
    from strands.memory import MemoryManager

    agent = Agent(memory_manager=MemoryManager(stores=[my_store], add_tool_config=True))
    ```
"""

from .memory_manager import MemoryManager
from .types import (
    MemoryAddToolConfig,
    MemoryEntry,
    MemoryStore,
    MemoryStoreError,
    MemoryToolConfig,
)

__all__ = [
    "MemoryAddToolConfig",
    "MemoryEntry",
    "MemoryManager",
    "MemoryStore",
    "MemoryStoreError",
    "MemoryToolConfig",
]
