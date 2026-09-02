"""Notebook tool for managing persistent text notebooks in agent state.

Notebooks are stored on the agent's :attr:`~strands.Agent.state` under the
``notebooks`` key and persist within the agent session (and, if the caller
supplies a durable :class:`~strands.agent.state.AgentState`, across sessions).
The tool is a thin wrapper over ``agent.state`` — persistence, isolation, and
serialization all follow whatever the caller configured for agent state.

Supported operations: ``create``, ``list``, ``read``, ``write`` (string
replacement or line insertion), and ``clear``.
"""

from __future__ import annotations

from typing import Any, Literal

from ...tools.decorator import tool
from ...types.tools import ToolContext
from .types import (
    DEFAULT_NOTEBOOK_DESCRIPTION,
    DEFAULT_NOTEBOOK_NAME,
    MAX_NOTEBOOK_NAME_LENGTH,
    MAX_NOTEBOOK_SIZE_BYTES,
    MAX_NOTEBOOKS,
    MAX_TOTAL_SIZE_BYTES,
)

_STATE_KEY = "notebooks"

# Modes that alter ``notebooks`` state. Only these persist a write.
_MUTATING_MODES = frozenset({"create", "write", "clear"})

# Modes that can grow the on-disk footprint. Only these need session-cap enforcement.
_GROWING_MODES = frozenset({"create", "write"})


@tool(name="notebook", description=DEFAULT_NOTEBOOK_DESCRIPTION, context="tool_context")
def notebook(
    mode: Literal["create", "list", "read", "write", "clear"],
    tool_context: ToolContext,
    name: str | None = None,
    new_str: str | None = None,
    old_str: str | None = None,
    insert_line: int | str | None = None,
    read_range: list[int] | None = None,
) -> str:
    """Manages text notebooks for note-taking and documentation.

    Supports `create`, `list`, `read`, `write` (replace or insert), and `clear`
    operations. Notebooks persist across invocations within a session, and across
    sessions when the agent has a durable state store.

    Args:
        mode: The operation to perform: `create`, `list`, `read`, `write`, `clear`.
        tool_context: Injected by the framework. Not user-facing.
        name: Name of the notebook to operate on. Defaults to "default".
        new_str: New string for replacement or insertion operations.
        old_str: String to replace in write mode when doing text replacement.
        insert_line: Line number (int) or search text (str) for insertion point in write mode.
            Supports negative indices.
        read_range: Optional parameter of `read` command. Line range to show [start, end].
            Supports negative indices.
    """
    state = tool_context.agent.state

    notebooks_obj: Any = state.get(_STATE_KEY)
    # AgentState.get deep-copies; guard shape strictly to catch corruption from sibling tools.
    if notebooks_obj is None:
        notebooks: dict[str, str] = {}
    elif isinstance(notebooks_obj, dict):
        for k, v in notebooks_obj.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError("Malformed notebooks state: keys and values must be strings")
        notebooks = notebooks_obj
    else:
        raise ValueError("Malformed notebooks state: expected a dict")

    # Ensure default notebook exists in-memory; only persisted if this mode mutates state.
    if not notebooks:
        notebooks[DEFAULT_NOTEBOOK_NAME] = ""

    target = _validate_notebook_name(name if name is not None else DEFAULT_NOTEBOOK_NAME)

    if mode == "create":
        result = _handle_create(notebooks, target, new_str)
    elif mode == "list":
        result = _handle_list(notebooks)
    elif mode == "read":
        result = _handle_read(notebooks, target, read_range)
    elif mode == "write":
        _validate_write_params(old_str, new_str, insert_line)
        result = _handle_write(notebooks, target, old_str, new_str, insert_line)
    elif mode == "clear":
        result = _handle_clear(notebooks, target)
    else:  # pragma: no cover - Literal narrows this at type-check time.
        raise ValueError(f"Unknown mode: {mode}")

    if mode in _GROWING_MODES:
        _enforce_session_caps(notebooks)
    if mode in _MUTATING_MODES:
        state.set(_STATE_KEY, notebooks)

    return result


# ---- Validation ----


def _validate_notebook_name(candidate: str) -> str:
    """Validate a notebook name.

    Args:
        candidate: The proposed notebook name.

    Returns:
        The notebook name, unchanged.

    Raises:
        ValueError: If the name is empty, too long, or contains disallowed characters.
    """
    if not isinstance(candidate, str) or not candidate:
        raise ValueError("Notebook name must be a non-empty string")
    if len(candidate) > MAX_NOTEBOOK_NAME_LENGTH:
        raise ValueError(f"Notebook name exceeds maximum length of {MAX_NOTEBOOK_NAME_LENGTH} characters")
    if candidate != candidate.strip():
        raise ValueError("Notebook name must not have leading or trailing whitespace")
    if "\0" in candidate:
        raise ValueError("Notebook name must not contain NUL bytes")
    if "/" in candidate or "\\" in candidate:
        raise ValueError("Notebook name must not contain path separators")
    if candidate in ("..", "."):
        raise ValueError("Notebook name is not allowed")
    return candidate


def _validate_write_params(old_str: str | None, new_str: str | None, insert_line: int | str | None) -> None:
    """Validate the parameter combination for a write operation.

    Args:
        old_str: The string to replace, if this is a replacement.
        new_str: The replacement or inserted text.
        insert_line: The insertion anchor (line number or search text), if this is an insertion.

    Raises:
        ValueError: If neither valid combination is present.
    """
    has_replacement = old_str is not None and new_str is not None
    has_insertion = insert_line is not None and new_str is not None
    if not (has_replacement or has_insertion):
        raise ValueError(
            "Write operation requires either (old_str + new_str) for replacement "
            "or (insert_line + new_str) for insertion"
        )
    # Reject both anchors — silently preferring one would let a misprompted model corrupt the notebook.
    if old_str is not None and insert_line is not None:
        raise ValueError(
            "Write operation is ambiguous: pass either `old_str` (replace) or `insert_line` (insert), not both"
        )


def _enforce_session_caps(notebooks: dict[str, str]) -> None:
    """Enforce per-session notebook count and size caps.

    Args:
        notebooks: The notebooks map to validate.

    Raises:
        ValueError: If any cap is exceeded.
    """
    if len(notebooks) > MAX_NOTEBOOKS:
        raise ValueError(f"Session notebook count exceeds maximum of {MAX_NOTEBOOKS}")
    total = 0
    for nb_name, content in notebooks.items():
        size = len(content.encode("utf-8"))
        if size > MAX_NOTEBOOK_SIZE_BYTES:
            raise ValueError(
                f"Notebook '{nb_name}' size ({size} bytes) exceeds maximum of {MAX_NOTEBOOK_SIZE_BYTES} bytes"
            )
        total += size
    if total > MAX_TOTAL_SIZE_BYTES:
        raise ValueError(f"Total notebook size ({total} bytes) exceeds session maximum of {MAX_TOTAL_SIZE_BYTES} bytes")


# ---- Handlers ----


def _handle_create(notebooks: dict[str, str], name: str, new_str: str | None) -> str:
    notebooks[name] = new_str if new_str is not None else ""
    suffix = " with specified content" if new_str else " (empty)"
    return f"Created notebook '{name}'{suffix}"


def _handle_list(notebooks: dict[str, str]) -> str:
    lines = []
    for nb_name, content in notebooks.items():
        line_count = len(content.split("\n")) if content else 0
        status = "Empty" if line_count == 0 else f"{line_count} lines"
        lines.append(f"- {nb_name}: {status}")
    return "Available notebooks:\n" + "\n".join(lines)


def _handle_read(notebooks: dict[str, str], name: str, read_range: list[int] | None) -> str:
    if name not in notebooks:
        raise ValueError(f"Notebook '{name}' not found")

    content = notebooks[name]

    if read_range is None:
        return content if content else f"Notebook '{name}' is empty"

    if len(read_range) != 2:
        raise ValueError("`read_range` must be a list of two integers: `[start, end]`")

    lines = content.split("\n")
    start, end = read_range[0], read_range[1]

    if start < 0:
        start = len(lines) + start + 1
    if end < 0:
        end = len(lines) + end + 1

    selected: list[str] = []
    for line_num in range(start, end + 1):
        if 1 <= line_num <= len(lines):
            selected.append(f"{line_num}: {lines[line_num - 1]}")

    return "\n".join(selected) if selected else "No valid lines found in range"


def _handle_write(
    notebooks: dict[str, str],
    name: str,
    old_str: str | None,
    new_str: str | None,
    insert_line: int | str | None,
) -> str:
    if name not in notebooks:
        raise ValueError(f"Notebook '{name}' not found")

    # String replacement mode.
    if old_str is not None and new_str is not None:
        if old_str not in notebooks[name]:
            raise ValueError(f"String '{old_str}' not found in notebook '{name}'")
        notebooks[name] = notebooks[name].replace(old_str, new_str, 1)
        return f"Replaced text in notebook '{name}'"

    # Line insertion mode.
    if insert_line is not None and new_str is not None:
        lines = notebooks[name].split("\n")

        if isinstance(insert_line, str):
            line_num = -1
            for i, line in enumerate(lines):
                if insert_line in line:
                    line_num = i
                    break
            if line_num == -1:
                raise ValueError(f"Text '{insert_line}' not found in notebook '{name}'")
        elif isinstance(insert_line, bool):
            # bool is a subclass of int; reject explicitly to avoid silent coercion.
            raise ValueError("`insert_line` must be an integer or string")
        elif isinstance(insert_line, int):
            if insert_line < 0:
                line_num = len(lines) + insert_line
            else:
                line_num = insert_line - 1
        else:
            raise ValueError("`insert_line` must be an integer or string")

        if line_num < -1 or line_num > len(lines):
            raise ValueError("Line number out of range")

        lines.insert(line_num + 1, new_str)
        notebooks[name] = "\n".join(lines)
        return f"Inserted text at line {line_num + 2} in notebook '{name}'"

    raise ValueError("Invalid write operation")


def _handle_clear(notebooks: dict[str, str], name: str) -> str:
    if name not in notebooks:
        raise ValueError(f"Notebook '{name}' not found")
    notebooks[name] = ""
    return f"Cleared notebook '{name}'"
