"""Sandbox-routed file editor tool for viewing, creating, and editing files.

Supports view (with line ranges), create, str_replace, and insert operations.

Example Usage:
    ```python
    from strands import Agent
    from strands.vended_tools import file_editor

    agent = Agent(tools=[file_editor])
    ```
"""

from .file_editor import file_editor, make_file_editor

__all__ = [
    "file_editor",
    "make_file_editor",
]
