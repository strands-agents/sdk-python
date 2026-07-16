"""Read-only file tool.

A thin shim over :func:`~strands.vended_tools.file_editor.make_file_editor`'s
``view`` command with a narrower two-parameter surface (``path`` and
``view_range``). All validation is delegated to ``file_editor``.

Example Usage:
    ```python
    from strands import Agent
    from strands.vended_tools import file_read

    agent = Agent(tools=[file_read])
    ```
"""

from .file_read import DEFAULT_FILE_READ_DESCRIPTION, file_read, make_file_read

__all__ = [
    "DEFAULT_FILE_READ_DESCRIPTION",
    "file_read",
    "make_file_read",
]
