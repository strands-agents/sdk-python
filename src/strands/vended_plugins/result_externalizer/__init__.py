"""Tool result externalization plugin for Strands Agents.

This module provides the ToolResultExternalizer plugin which intercepts oversized
tool results, persists the full content to a storage backend, and replaces the
in-context result with a truncated preview and reference.

Example Usage:
    ```python
    from strands import Agent
    from strands.vended_plugins.result_externalizer import (
        ToolResultExternalizer,
        InMemoryExternalizationStorage,
        FileExternalizationStorage,
    )

    # In-memory storage
    agent = Agent(plugins=[
        ToolResultExternalizer(storage=InMemoryExternalizationStorage())
    ])

    # File storage with custom thresholds
    agent = Agent(plugins=[
        ToolResultExternalizer(
            storage=FileExternalizationStorage("./artifacts"),
            max_result_chars=20_000,
            preview_chars=8_000,
        )
    ])
    ```
"""

from .plugin import ToolResultExternalizer
from .storage import (
    ExternalizationStorage,
    FileExternalizationStorage,
    InMemoryExternalizationStorage,
    S3ExternalizationStorage,
)

__all__ = [
    "ToolResultExternalizer",
    "ExternalizationStorage",
    "FileExternalizationStorage",
    "InMemoryExternalizationStorage",
    "S3ExternalizationStorage",
]
