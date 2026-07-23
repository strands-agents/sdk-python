"""Read-only file tool.

A minimal, read-only surface over :func:`~strands.vended_tools.file_editor.make_file_editor`:
the model sees only ``path`` and ``view_range``, and every call is routed through
``file_editor``'s ``view`` command. All input validation (absolute path, ``..``
traversal, size limit, ``view_range`` bounds, sandbox probing) is delegated to
``file_editor``; this tool intentionally adds no new logic and no new checks.

Use this tool when an agent should be able to read files and list directories
but must not be able to create or edit them.

Note on schema strictness: the TypeScript tool uses a ``.strict()`` Zod schema
that rejects any key outside ``{path, view_range}`` with a validation error.
The Python tool relies on Pydantic's default behavior in ``@tool``, which
silently strips unknown keys. Both paths are safe -- the shim hard-codes
``command="view"``, so a smuggled ``command`` or ``file_text`` field cannot
reach ``file_editor``'s write branches -- but the model receives feedback only
in the TypeScript path. This mirrors the ``@tool`` decorator's convention for
all vended Python tools and is intentional.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ...tools.decorator import tool
from ...types.tools import ToolContext
from ..file_editor import make_file_editor

if TYPE_CHECKING:
    from ...sandbox.base import Sandbox
    from ...tools.decorator import DecoratedFunctionTool

DEFAULT_FILE_READ_DESCRIPTION = (
    "Read-only filesystem tool. View a file (with an optional line range) or list a directory. "
    "Paths must be absolute. For creating or editing files, use `file_editor`."
)
"""Description for the file_read tool."""


def make_file_read(
    *,
    sandbox: Sandbox | None = None,
    name: str = "file_read",
    description: str = DEFAULT_FILE_READ_DESCRIPTION,
) -> DecoratedFunctionTool:
    """Create a sandbox-routed, read-only file tool.

    A thin shim over ``file_editor``'s ``view`` command with a narrower input
    schema. If ``sandbox`` is passed, it is bound at creation time. Otherwise
    the underlying ``file_editor`` reads the sandbox from
    ``tool_context.agent.sandbox`` at call time.

    Args:
        sandbox: Sandbox to bind at creation. When ``None``, the agent's
            configured sandbox is used at call time.
        name: Tool name. Defaults to ``"file_read"``.
        description: Tool description shown to the model.

    Returns:
        A decorated tool that reads files and lists directories through the sandbox.
    """
    editor = make_file_editor(sandbox=sandbox)

    @tool(name=name, description=description, context="tool_context")
    async def file_read_tool(
        path: str,
        tool_context: ToolContext,
        view_range: tuple[int, int] | None = None,
    ) -> str:
        """Read a file or list a directory.

        Args:
            path: Absolute path to a file or directory.
            tool_context: Injected by the framework. Not user-facing.
            view_range: Optional line range ``[start, end]`` to view. 1-indexed;
                ``end`` may be ``-1`` for end-of-file. Not allowed when ``path``
                is a directory.
        """
        result = await editor(
            command="view",
            path=path,
            tool_context=tool_context,
            view_range=list(view_range) if view_range is not None else None,
        )
        return cast(str, result)

    return file_read_tool


file_read = make_file_read()
"""Default read-only file tool. Reads the sandbox from the agent's context at call time."""
