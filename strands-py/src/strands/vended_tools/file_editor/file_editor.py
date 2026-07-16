"""Sandbox-routed file editor tool.

Provides
``view`` (with line ranges), ``create``, ``str_replace``, ``insert``,
``pattern_replace``, ``find_line``, and ``undo_edit`` operations, all routed
through a :class:`~strands.sandbox.base.Sandbox`: either one bound at creation
(as the built-in Docker/SSH sandboxes do when vending tools) or the agent's
configured sandbox read from ``tool_context.agent.sandbox`` at call time.
"""

from __future__ import annotations

import asyncio
import os
import posixpath
import re
from collections import OrderedDict
from typing import TYPE_CHECKING, Literal

from ...sandbox.errors import SandboxPathNotFoundError
from ...tools.decorator import tool
from ...types.tools import ToolContext

if TYPE_CHECKING:
    from ...sandbox.base import Sandbox
    from ...tools.decorator import DecoratedFunctionTool

_MB = 1024 * 1024
_SNIPPET_LINES = 4
_DEFAULT_MAX_FILE_SIZE = 1 * _MB
_MAX_DIRECTORY_DEPTH = 2
_MAX_PATTERN_LENGTH = 1000
_MAX_PATTERN_MATCHES = 10_000
_PATTERN_REPLACE_TIMEOUT_SEC = 5.0
_DEFAULT_MAX_UNDO_ENTRIES = 32
_DEFAULT_MAX_UNDO_BYTES = 32 * _MB

DEFAULT_FILE_EDITOR_DESCRIPTION = (
    "Filesystem editor for viewing, creating, and editing files. Supports view (with "
    "line ranges), create, str_replace (exact match; ambiguous matches must opt in via "
    "replace_all), insert, pattern_replace (regex), find_line, and undo_edit. Files "
    "must use absolute paths."
)

_Command = Literal[
    "view",
    "create",
    "str_replace",
    "insert",
    "pattern_replace",
    "find_line",
    "undo_edit",
]


def make_file_editor(
    *,
    sandbox: Sandbox | None = None,
    name: str = "file_editor",
    description: str = DEFAULT_FILE_EDITOR_DESCRIPTION,
    root: str | None = None,
    max_file_size: int = _DEFAULT_MAX_FILE_SIZE,
    max_undo_entries: int = _DEFAULT_MAX_UNDO_ENTRIES,
    max_undo_bytes: int = _DEFAULT_MAX_UNDO_BYTES,
    pattern_replace_timeout: float = _PATTERN_REPLACE_TIMEOUT_SEC,
) -> DecoratedFunctionTool:
    """Create a sandbox-routed file editor tool.

    If a ``sandbox`` is passed, it is bound at creation time. Otherwise the tool
    reads the sandbox from ``tool_context.agent.sandbox`` at call time. Used by
    sandbox implementations in :meth:`~strands.sandbox.base.Sandbox.get_tools`
    and by users who want a customized file editor.

    Args:
        sandbox: Sandbox to bind at creation. When ``None``, the agent's
            configured sandbox is used at call time.
        name: Tool name. Defaults to ``"file_editor"``.
        description: Tool description shown to the model.
        root: Optional absolute directory that confines every operation. String-
            level checks reject non-absolute paths and any ``..`` traversal on
            the raw input; when the resolved target exists on the local host,
            ``os.path.realpath`` is also applied and the resulting path must
            still be inside ``root``. Confinement is enforced against the raw
            (string-level) path input before I/O; concurrent rename attacks
            between the confinement check and the read/write are the sandbox's
            responsibility. When ``root`` is ``None``, only absolute-path and
            ``..``-traversal checks apply; the underlying sandbox's symlink
            policy governs escape.
        max_file_size: Maximum file size (bytes) accepted by view/edit
            commands. Defaults to 1 MB.
        max_undo_entries: Maximum number of distinct paths retained in the
            in-memory undo history. Oldest entry is evicted on overflow.
            Defaults to 32.
        max_undo_bytes: Approximate cap on total bytes of file content held in
            the undo history (UTF-8). Oldest entries are evicted until the cap
            is met. Defaults to 32 MB.
        pattern_replace_timeout: Wall-clock seconds allowed for a single
            ``pattern_replace`` invocation before the tool aborts with a
            "possible catastrophic regex" error. Defaults to 5.0 seconds.

    Returns:
        A decorated tool that performs file operations through the sandbox.
    """
    if root is not None and not posixpath.isabs(root):
        raise ValueError(f"root must be an absolute path, got: {root}")
    normalized_root: str | None = None if root is None else posixpath.normpath(root).rstrip("/") or "/"

    # Bounded LRU: previous file content keyed by path. A second edit overwrites
    # the snapshot for that path, matching the upstream editor's single-step
    # undo semantics; older entries are evicted when either cap is exceeded.
    undo_history: OrderedDict[str, str] = OrderedDict()

    @tool(name=name, description=description, context="tool_context")
    async def file_editor_tool(
        command: _Command,
        path: str,
        tool_context: ToolContext,
        file_text: str | None = None,
        view_range: list[int] | None = None,
        old_str: str | None = None,
        new_str: str | None = None,
        insert_line: int | None = None,
        pattern: str | None = None,
        search_text: str | None = None,
        fuzzy: bool = False,
        replace_all: bool = False,
    ) -> str:
        """Filesystem editor for viewing, creating, and editing files.

        Args:
            command: The operation to perform: ``view``, ``create``,
                ``str_replace``, ``insert``, ``pattern_replace``,
                ``find_line``, or ``undo_edit``.
            path: Absolute path to the file or directory.
            tool_context: Injected by the framework. Not user-facing.
            file_text: Content for new file (required for ``create``).
            view_range: Line range to view ``[start, end]``. 1-indexed. End can
                be ``-1`` for end of file.
            old_str: Exact string to find and replace (required for
                ``str_replace``). Must be unique unless ``replace_all=True``.
            new_str: Replacement string (for ``str_replace``,
                ``pattern_replace``, and ``insert``).
            insert_line: Line number where text should be inserted (0-indexed,
                required for ``insert``).
            pattern: Regex pattern to match (required for ``pattern_replace``).
            search_text: Text to search for (required for ``find_line``).
            fuzzy: Enable whitespace-tolerant matching for ``find_line``.
            replace_all: For ``str_replace`` / ``pattern_replace``, allow
                replacing all occurrences. Defaults to ``False`` (unique match
                required) to prevent silent broad edits.
        """
        active = sandbox if sandbox is not None else tool_context.agent.sandbox
        resolved = _resolve_path(path, normalized_root)

        if command == "view":
            return await _handle_view(active, resolved, view_range, max_file_size)
        if command == "create":
            return await _handle_create(active, resolved, file_text, undo_history, max_file_size)
        if command == "str_replace":
            return await _handle_str_replace(
                active,
                resolved,
                old_str,
                new_str,
                replace_all,
                max_file_size,
                undo_history,
                max_undo_entries,
                max_undo_bytes,
            )
        if command == "insert":
            return await _handle_insert(
                active,
                resolved,
                insert_line,
                new_str,
                max_file_size,
                undo_history,
                max_undo_entries,
                max_undo_bytes,
            )
        if command == "pattern_replace":
            return await _handle_pattern_replace(
                active,
                resolved,
                pattern,
                new_str,
                replace_all,
                max_file_size,
                undo_history,
                max_undo_entries,
                max_undo_bytes,
                pattern_replace_timeout,
            )
        if command == "find_line":
            return await _handle_find_line(active, resolved, search_text, fuzzy, max_file_size)
        if command == "undo_edit":
            return await _handle_undo(active, resolved, undo_history)
        raise ValueError(f"Unknown command: {command}")

    return file_editor_tool


file_editor = make_file_editor()
"""Default sandbox-routed file editor tool. Reads the sandbox from the agent's context at call time."""


# ---- Path resolution and confinement ----


def _resolve_path(file_path: str, root: str | None) -> str:
    """Normalize a path, rejecting traversal and out-of-root inputs.

    This is the single funnel every command routes through — it exists so path
    validation cannot be bypassed by a new command forgetting to call it. When
    ``root`` is set and the target (or its parent, for a not-yet-existing file)
    exists locally, symlinks are also resolved via ``os.path.realpath`` and the
    result must still be inside ``root`` — a symlink inside ``root`` pointing
    at ``/etc/passwd`` is rejected.

    Args:
        file_path: The raw path from the tool input.
        root: Optional absolute directory that confines the result.

    Returns:
        The normalized (trailing-slash-stripped, ``posixpath.normpath``)
        absolute path.

    Raises:
        ValueError: If the path is not absolute, contains a ``..`` segment, or
            resolves (via ``realpath`` when applicable) outside ``root``.
    """
    # Strip trailing slashes on the raw input (compat with pre-existing behavior).
    stripped = re.sub(r"[/\\]+$", "", file_path) or file_path

    if not posixpath.isabs(stripped):
        suggested = posixpath.abspath(stripped)
        raise ValueError(
            f"The path {file_path} is not an absolute path, it should start with `/`. Maybe you meant {suggested}?"
        )

    # Reject '..' segments on the raw input -- normalize() would resolve them
    # away and could silently permit escape past the root.
    if ".." in re.split(r"[/\\]", stripped):
        raise ValueError("Invalid path: path traversal is not allowed")

    normalized = posixpath.normpath(stripped)

    if root is not None:
        # String-level confinement: normalized must equal root or start with root + "/".
        if normalized != root and not normalized.startswith(root.rstrip("/") + "/"):
            raise ValueError(f"Invalid path: {file_path} is outside the configured root {root}")

        # Symlink confinement is a best-effort, local-fs-only layer. If `root`
        # itself does not exist on the local filesystem, the caller is running
        # against a non-local sandbox (Docker, SSH, ...) whose paths do not map
        # to this host — `os.path.realpath` would return the unchanged
        # non-existent path and the check below would reject every operation
        # (`realpath("/workspace-in-container")` != `realpath(root)`). In that
        # case, the string-level check above is the whole policy; the sandbox
        # owns its own symlink handling.
        if not os.path.lexists(root):
            return normalized

        # Resolve the deepest existing ancestor via realpath and confirm it is
        # still inside root. Catches a symlink `inside-root/link -> /etc/passwd`
        # that the string-level check can't see.
        probe = normalized
        while probe and probe != "/" and not os.path.lexists(probe):
            parent = posixpath.dirname(probe)
            if parent == probe:
                break
            probe = parent
        if probe and os.path.lexists(probe):
            real = os.path.realpath(probe)
            root_real = os.path.realpath(root)
            if real != root_real and not real.startswith(root_real.rstrip("/") + "/"):
                raise ValueError(
                    f"Invalid path: {file_path} resolves via symlink to {real}, outside the configured root {root}"
                )

    return normalized


def _apply_view_range(file_content: str, view_range: list[int] | None) -> tuple[str, int]:
    """Slice file content to a 1-indexed [start, end] range (end -1 means end of file).

    Args:
        file_content: The full file content.
        view_range: The [start, end] range, or ``None`` for the whole file.

    Returns:
        A tuple of (visible content, first line number for output numbering).

    Raises:
        ValueError: If the range is out of bounds or malformed.
    """
    if not view_range:
        return file_content, 1
    lines = file_content.split("\n")
    n_lines = len(lines)
    start, end = view_range[0], view_range[1]

    if start < 1 or start > n_lines:
        raise ValueError(
            f"Invalid `view_range`: [{start}, {end}]. Its first element `{start}` should be within the "
            f"range of lines of the file: [1, {n_lines}]"
        )
    if end != -1 and end > n_lines:
        raise ValueError(
            f"Invalid `view_range`: [{start}, {end}]. Its second element `{end}` should be smaller than "
            f"the number of lines in the file: `{n_lines}`"
        )
    if end != -1 and end < start:
        raise ValueError(
            f"Invalid `view_range`: [{start}, {end}]. Its second element `{end}` should be larger or "
            f"equal than its first `{start}`"
        )

    content = "\n".join(lines[start - 1 :]) if end == -1 else "\n".join(lines[start - 1 : end])
    return content, start


def _build_str_replace_result(
    original_content: str,
    old_str: str,
    new_str: str | None,
    file_path: str,
    replace_all: bool,
) -> tuple[str, str, int, int]:
    """Perform ``str_replace`` and return ``(new content, snippet, snippet start, count)``.

    Args:
        original_content: The current file content.
        old_str: The exact string to replace.
        new_str: The replacement string (``None`` deletes the match).
        file_path: The file path, for error messages.
        replace_all: When ``True``, replace every occurrence; otherwise require
            exactly one match.

    Raises:
        ValueError: If ``old_str`` does not appear, or appears more than once
            without ``replace_all``.
    """
    new_str_value = new_str or ""

    occurrences = original_content.count(old_str)
    if occurrences == 0:
        raise ValueError(f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {file_path}.")
    if occurrences > 1 and not replace_all:
        lines = original_content.split("\n")
        line_numbers = [i + 1 for i, line in enumerate(lines) if old_str in line]
        raise ValueError(
            f"No replacement was performed. Multiple occurrences of old_str `{old_str}` in lines "
            f"{line_numbers}. Pass replace_all=True to replace every occurrence, or make old_str unique."
        )

    count = occurrences if replace_all else 1
    new_content = (
        original_content.replace(old_str, new_str_value)
        if replace_all
        else original_content.replace(old_str, new_str_value, 1)
    )
    replacement_line = len(original_content[: original_content.index(old_str)].split("\n")) - 1
    inserted_lines = len(new_str_value.split("\n"))
    original_lines = len(old_str.split("\n"))
    line_difference = inserted_lines - original_lines

    new_lines = new_content.split("\n")
    start_line = max(0, replacement_line - _SNIPPET_LINES)
    end_line = min(len(new_lines), replacement_line + _SNIPPET_LINES + line_difference + 1)
    snippet = "\n".join(new_lines[start_line:end_line])

    return new_content, snippet, start_line, count


def _build_insert_result(original_content: str, insert_line: int, new_str: str) -> tuple[str, str, int]:
    """Insert text at a 0-indexed line and return (new content, snippet, 0-indexed snippet start).

    Args:
        original_content: The current file content.
        insert_line: The 0-indexed line after which to insert.
        new_str: The text to insert.

    Returns:
        A tuple of (new content, snippet around the insertion, 0-indexed snippet start line).

    Raises:
        ValueError: If ``insert_line`` is out of bounds.
    """
    file_text_lines = original_content.split("\n")
    n_lines = len(file_text_lines)

    if insert_line < 0 or insert_line > n_lines:
        raise ValueError(
            f"Invalid `insert_line` parameter: {insert_line}. It should be within the range of lines "
            f"of the file: [0, {n_lines}]"
        )

    new_str_lines = new_str.split("\n")
    new_file_text_lines = (
        new_str_lines
        if original_content == ""
        else [*file_text_lines[:insert_line], *new_str_lines, *file_text_lines[insert_line:]]
    )

    new_content = "\n".join(new_file_text_lines)
    snippet_start_line = max(0, insert_line - _SNIPPET_LINES)
    snippet_end_line = min(len(new_file_text_lines), insert_line + len(new_str_lines) + _SNIPPET_LINES)
    snippet = "\n".join(new_file_text_lines[snippet_start_line:snippet_end_line])

    return new_content, snippet, snippet_start_line


def _make_output(file_content: str, file_descriptor: str, init_line: int = 1) -> str:
    """Format file content with ``cat -n`` style line numbers.

    Args:
        file_content: The content to number.
        file_descriptor: A description of the source (file path or snippet label).
        init_line: The line number of the first line.

    Returns:
        The formatted, line-numbered output.
    """
    expanded_content = file_content.replace("\t", "        ")
    numbered_lines = [f"{index + init_line:>6}  {line}" for index, line in enumerate(expanded_content.split("\n"))]
    return f"Here's the result of running `cat -n` on {file_descriptor}:\n" + "\n".join(numbered_lines) + "\n"


def _find_line_number(content: str, search_text: str, fuzzy: bool) -> int:
    """Return the 0-indexed line number where ``search_text`` first appears, else ``-1``.

    When ``fuzzy=True``, whitespace between words is collapsed and matching is
    case-insensitive.
    """
    lines = content.split("\n")
    if fuzzy:
        tokens = search_text.strip().split()
        if not tokens:
            return -1
        pattern = re.compile(".*".join(re.escape(t) for t in tokens), re.IGNORECASE)
        for i, line in enumerate(lines):
            if pattern.search(line):
                return i
    else:
        for i, line in enumerate(lines):
            if search_text in line:
                return i
    return -1


# ---- Undo history bookkeeping ----


def _store_undo_snapshot(
    undo_history: OrderedDict[str, str],
    file_path: str,
    content: str,
    max_entries: int,
    max_bytes: int,
) -> None:
    """Record a pre-edit snapshot in the LRU undo history, evicting oldest on overflow.

    ``dict`` insertion order gives LRU semantics: a re-inserted key is moved to
    the end, and eviction removes the oldest key when either the entry count or
    the aggregate byte cap is exceeded.
    """
    if file_path in undo_history:
        del undo_history[file_path]
    undo_history[file_path] = content
    total_bytes = sum(len(v.encode("utf-8")) for v in undo_history.values())
    while undo_history and (len(undo_history) > max_entries or total_bytes > max_bytes):
        _, evicted = undo_history.popitem(last=False)
        total_bytes -= len(evicted.encode("utf-8"))


# ---- Sandbox-routed I/O helpers ----


async def _probe_sandbox_path(sandbox: Sandbox, file_path: str) -> tuple[bool, bool]:
    """Probe a path through the sandbox, returning (exists, is_dir).

    Lists the parent directory and looks for the entry. A missing parent or entry
    resolves to ``(False, False)``; permission, transport, and other failures
    propagate so they are not disguised as non-existence.

    Args:
        sandbox: The sandbox to probe through.
        file_path: The path to check.

    Returns:
        A tuple of (exists, is_dir).
    """
    normalized = file_path.replace("\\", "/")
    parent = "/".join(normalized.split("/")[:-1]) or "/"
    name = normalized.split("/")[-1]
    try:
        entry = next((e for e in await sandbox.list_files(parent) if e.name == name), None)
    except SandboxPathNotFoundError:
        return False, False
    if entry is None:
        return False, False
    return True, entry.is_dir or False


async def _read_text_or_reject_binary(sandbox: Sandbox, file_path: str, max_size: int) -> str:
    """Read text through the sandbox, rejecting binary files and oversize inputs.

    Reads raw bytes first so the size cap and encoding detection run before UTF-8
    decoding — a corrupted UTF-8 error message is worse than a clean rejection.
    UTF-16 BOMs are detected up front so a valid UTF-16 file is reported as an
    unsupported encoding rather than misclassified as binary. Otherwise the
    classic NUL-in-first-8-KB heuristic identifies binary content.
    """
    raw = await sandbox.read_file(file_path)
    if len(raw) > max_size:
        raise ValueError(f"File size ({len(raw)} bytes) exceeds maximum allowed size ({max_size} bytes)")
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        raise ValueError(f"Refusing to read non-UTF-8 file (detected UTF-16 BOM): {file_path}")
    if b"\x00" in raw[:8192]:
        raise ValueError(f"Refusing to read binary file: {file_path}")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"Refusing to read non-UTF-8 file: {file_path}") from e


async def _list_directory(sandbox: Sandbox, dir_path: str) -> str:
    """List directory contents up to 2 levels deep through the sandbox, excluding hidden files.

    Args:
        sandbox: The sandbox to list through.
        dir_path: The directory to list.

    Returns:
        A formatted listing of relative paths.
    """
    items: list[str] = []

    async def walk(current_path: str, prefix: str, depth: int) -> None:
        try:
            entries = await sandbox.list_files(current_path)
        except OSError:
            # Ignore permission/path errors and continue.
            return
        for entry in entries:
            if entry.name.startswith("."):
                continue
            relative_path = f"{prefix}/{entry.name}" if prefix else entry.name
            items.append(relative_path)
            if entry.is_dir and depth < _MAX_DIRECTORY_DEPTH:
                await walk(f"{current_path}/{entry.name}", relative_path, depth + 1)

    await walk(dir_path, "", 0)
    result = "\n".join(sorted(items))
    return f"Here's the files and directories up to 2 levels deep in {dir_path}, excluding hidden items:\n{result}\n"


# ---- Sandbox-path handlers ----


async def _handle_view(sandbox: Sandbox, file_path: str, view_range: list[int] | None, max_size: int) -> str:
    """Handle the ``view`` command: render a file with line numbers or list a directory."""
    exists, is_dir = await _probe_sandbox_path(sandbox, file_path)
    if not exists:
        raise ValueError(f"The path {file_path} does not exist. Please provide a valid path.")

    if is_dir:
        if view_range:
            raise ValueError("The `view_range` parameter is not allowed when `path` points to a directory.")
        return await _list_directory(sandbox, file_path)

    file_content = await _read_text_or_reject_binary(sandbox, file_path, max_size)

    content, init_line = _apply_view_range(file_content, view_range)
    return _make_output(content, file_path, init_line)


def _reject_oversize_replacement(text: str | None, max_size: int, label: str = "new_str") -> None:
    """Reject a replacement payload whose UTF-8 encoding would exceed ``max_size``.

    The read side already refuses to load files above the cap; this mirrors it
    on the write side so a model can't ship an unbounded ``file_text`` or
    ``new_str`` through the tool.
    """
    if text is None:
        return
    encoded = len(text.encode("utf-8"))
    if encoded > max_size:
        raise ValueError(f"{label} ({encoded} bytes) exceeds maximum allowed size ({max_size} bytes)")


def _reject_oversize_result(content: str, max_size: int, file_path: str) -> None:
    """Reject a post-edit buffer whose UTF-8 length would exceed ``max_size``.

    Catches the case where the individual ``new_str`` fits but the resulting
    file (after all substitutions) is larger than the read cap — a
    ``replace_all`` that expands every match, for instance.
    """
    encoded = len(content.encode("utf-8"))
    if encoded > max_size:
        raise ValueError(
            f"The edit would produce a {encoded}-byte file at {file_path}, "
            f"exceeding the maximum allowed size of {max_size} bytes."
        )


async def _handle_create(
    sandbox: Sandbox,
    file_path: str,
    file_text: str | None,
    undo_history: OrderedDict[str, str],
    max_size: int,
) -> str:
    """Handle the ``create`` command: write a new file, refusing to overwrite."""
    if file_text is None:
        raise ValueError("Parameter `file_text` is required for command: create")
    _reject_oversize_replacement(file_text, max_size, label="file_text")

    exists, _ = await _probe_sandbox_path(sandbox, file_path)
    if exists:
        raise ValueError(f"File already exists at: {file_path}. Cannot overwrite files using command `create`.")

    await sandbox.write_text(file_path, file_text)
    # ``create`` is intentionally not snapshotted for undo: rolling back a
    # create means deleting the file, which is a different operation from
    # "restore prior content" and is easy for the caller to do themselves.
    undo_history.pop(file_path, None)
    return f"File created successfully at: {file_path}"


async def _handle_str_replace(
    sandbox: Sandbox,
    file_path: str,
    old_str: str | None,
    new_str: str | None,
    replace_all: bool,
    max_size: int,
    undo_history: OrderedDict[str, str],
    max_undo_entries: int,
    max_undo_bytes: int,
) -> str:
    """Handle the ``str_replace`` command: replace ``old_str`` (unique unless ``replace_all``)."""
    if old_str is None:
        raise ValueError("Parameter `old_str` is required for command: str_replace")
    if old_str == "":
        raise ValueError("Parameter `old_str` must not be empty for command: str_replace")
    _reject_oversize_replacement(new_str, max_size)

    exists, is_dir = await _probe_sandbox_path(sandbox, file_path)
    if not exists:
        raise ValueError(f"The path {file_path} does not exist. Please provide a valid path.")
    if is_dir:
        raise ValueError(f"The path {file_path} is a directory and only the `view` command can be used on directories")

    file_content = await _read_text_or_reject_binary(sandbox, file_path, max_size)

    new_content, snippet, start_line, count = _build_str_replace_result(
        file_content, old_str, new_str, file_path, replace_all
    )

    _reject_oversize_result(new_content, max_size, file_path)
    _store_undo_snapshot(undo_history, file_path, file_content, max_undo_entries, max_undo_bytes)
    await sandbox.write_text(file_path, new_content)

    suffix = f" ({count} occurrences replaced)" if replace_all and count > 1 else ""
    return (
        f"The file {file_path} has been edited.{suffix} "
        f"{_make_output(snippet, f'a snippet of {file_path}', start_line + 1)}"
        "Review the changes and make sure they are as expected. Edit the file again if necessary."
    )


async def _handle_insert(
    sandbox: Sandbox,
    file_path: str,
    insert_line: int | None,
    new_str: str | None,
    max_size: int,
    undo_history: OrderedDict[str, str],
    max_undo_entries: int,
    max_undo_bytes: int,
) -> str:
    """Handle the ``insert`` command: insert text at a 0-indexed line."""
    if insert_line is None or new_str is None:
        raise ValueError("Parameters `insert_line` and `new_str` are required for command: insert")
    _reject_oversize_replacement(new_str, max_size)

    exists, is_dir = await _probe_sandbox_path(sandbox, file_path)
    if not exists:
        raise ValueError(f"The path {file_path} does not exist. Please provide a valid path.")
    if is_dir:
        raise ValueError(f"The path {file_path} is a directory and only the `view` command can be used on directories")

    file_text = await _read_text_or_reject_binary(sandbox, file_path, max_size)

    new_content, snippet, start_line = _build_insert_result(file_text, insert_line, new_str)

    _reject_oversize_result(new_content, max_size, file_path)
    _store_undo_snapshot(undo_history, file_path, file_text, max_undo_entries, max_undo_bytes)
    await sandbox.write_text(file_path, new_content)

    return (
        f"The file {file_path} has been edited. "
        f"{_make_output(snippet, 'a snippet of the edited file', start_line + 1)}"
        "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). "
        "Edit the file again if necessary."
    )


async def _handle_pattern_replace(
    sandbox: Sandbox,
    file_path: str,
    pattern: str | None,
    new_str: str | None,
    replace_all: bool,
    max_size: int,
    undo_history: OrderedDict[str, str],
    max_undo_entries: int,
    max_undo_bytes: int,
    timeout_sec: float,
) -> str:
    r"""Handle the ``pattern_replace`` command: regex-based replacement.

    Uses Python ``re`` semantics for both pattern and replacement (backreferences
    like ``\1`` in ``new_str`` are honored — this is the deliberate difference
    from ``str_replace``, whose replacement is literal). Requires ``replace_all``
    for more than one match, same as ``str_replace``.

    The pattern length is capped, match count is capped, and the whole
    match/substitution runs in a worker thread with a wall-clock timeout so a
    catastrophic regex (``(a+)+b`` and friends) cannot hang the event loop.

    Known limit: on timeout, ``asyncio.wait_for`` returns to the caller but the
    underlying worker thread continues running the runaway regex (Python
    threads cannot be cancelled). Concurrent catastrophic patterns will pin CPU
    on the shared default thread pool until they complete.

    Cross-SDK divergence: the TypeScript SDK cannot replicate this timeout — V8
    has no in-exec regex timeout and JavaScript has no way to cancel a running
    ``.exec()`` call. Instead it statically rejects the classic ReDoS shapes
    (``(a+)+``, ``(a*)*``, ...) before compiling. A pattern accepted here under
    ``pattern_replace_timeout`` may be refused by the TypeScript side.
    """
    if pattern is None:
        raise ValueError("Parameter `pattern` is required for command: pattern_replace")
    if pattern == "":
        raise ValueError("Parameter `pattern` must not be empty for command: pattern_replace")
    if new_str is None:
        raise ValueError("Parameter `new_str` is required for command: pattern_replace")
    _reject_oversize_replacement(new_str, max_size)
    if len(pattern) > _MAX_PATTERN_LENGTH:
        raise ValueError(f"Pattern is {len(pattern)} chars, exceeds maximum of {_MAX_PATTERN_LENGTH}.")

    try:
        regex = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern `{pattern}`: {e}") from e

    exists, is_dir = await _probe_sandbox_path(sandbox, file_path)
    if not exists:
        raise ValueError(f"The path {file_path} does not exist. Please provide a valid path.")
    if is_dir:
        raise ValueError(f"The path {file_path} is a directory and only the `view` command can be used on directories")

    file_content = await _read_text_or_reject_binary(sandbox, file_path, max_size)

    def _run_regex() -> tuple[list[re.Match[str]], str, int]:
        # Iterate rather than list(finditer) so a runaway match count can be
        # aborted early. The substitution reuses the counted matches.
        collected: list[re.Match[str]] = []
        for m in regex.finditer(file_content):
            collected.append(m)
            if len(collected) > _MAX_PATTERN_MATCHES:
                raise ValueError(
                    f"Pattern `{pattern}` produced more than {_MAX_PATTERN_MATCHES} matches; refusing to continue."
                )
        if not collected:
            return collected, file_content, 0
        substituted = regex.sub(new_str, file_content, count=0 if replace_all else 1)
        replaced = len(collected) if replace_all else 1
        return collected, substituted, replaced

    try:
        matches, new_content, count = await asyncio.wait_for(asyncio.to_thread(_run_regex), timeout=timeout_sec)
    except asyncio.TimeoutError as e:
        raise ValueError(
            f"Pattern `{pattern}` timed out after {timeout_sec}s; refusing to continue (possible catastrophic regex)."
        ) from e
    except re.error as e:
        # `regex.sub` raises `re.error` at substitution time when `new_str`
        # references a group that does not exist (e.g. `\9`) or contains
        # malformed group syntax. Surface it as the same clean `ValueError` the
        # compile step (above) uses for bad patterns.
        raise ValueError(f"Invalid replacement `{new_str}` for pattern `{pattern}`: {e}") from e

    if not matches:
        raise ValueError(f"No replacement was performed, pattern `{pattern}` did not match in {file_path}.")
    if len(matches) > 1 and not replace_all:
        line_numbers = [file_content.count("\n", 0, m.start()) + 1 for m in matches]
        raise ValueError(
            f"No replacement was performed. Pattern `{pattern}` matched {len(matches)} times "
            f"(lines {line_numbers}). Pass replace_all=True to replace every occurrence, or tighten the pattern."
        )

    _reject_oversize_result(new_content, max_size, file_path)
    _store_undo_snapshot(undo_history, file_path, file_content, max_undo_entries, max_undo_bytes)
    await sandbox.write_text(file_path, new_content)

    replacement_line = file_content.count("\n", 0, matches[0].start())
    new_lines = new_content.split("\n")
    start = max(0, replacement_line - _SNIPPET_LINES)
    end = min(len(new_lines), replacement_line + _SNIPPET_LINES + 1)
    snippet = "\n".join(new_lines[start:end])

    suffix = f" ({count} matches replaced)" if replace_all and count > 1 else ""
    return (
        f"The file {file_path} has been edited via pattern_replace.{suffix} "
        f"{_make_output(snippet, f'a snippet of {file_path}', start + 1)}"
        "Review the changes and make sure they are as expected. Edit the file again if necessary."
    )


async def _handle_find_line(
    sandbox: Sandbox,
    file_path: str,
    search_text: str | None,
    fuzzy: bool,
    max_size: int,
) -> str:
    """Handle the ``find_line`` command: return the 1-indexed line and a context snippet."""
    if search_text is None:
        raise ValueError("Parameter `search_text` is required for command: find_line")

    exists, is_dir = await _probe_sandbox_path(sandbox, file_path)
    if not exists:
        raise ValueError(f"The path {file_path} does not exist. Please provide a valid path.")
    if is_dir:
        raise ValueError(f"The path {file_path} is a directory and only the `view` command can be used on directories")

    file_content = await _read_text_or_reject_binary(sandbox, file_path, max_size)

    line_index = _find_line_number(file_content, search_text, fuzzy)
    if line_index == -1:
        raise ValueError(f"Could not find `{search_text}` in {file_path}.")

    lines = file_content.split("\n")
    start = max(0, line_index - _SNIPPET_LINES)
    end = min(len(lines), line_index + _SNIPPET_LINES + 1)
    snippet = "\n".join(lines[start:end])
    return (
        f"Found `{search_text}` at line {line_index + 1} of {file_path}.\n"
        f"{_make_output(snippet, f'a snippet of {file_path}', start + 1)}"
    )


async def _handle_undo(sandbox: Sandbox, file_path: str, undo_history: OrderedDict[str, str]) -> str:
    """Handle the ``undo_edit`` command: restore the last in-memory snapshot for ``file_path``.

    The snapshot is unconditionally written back through the sandbox: if the
    file was deleted (or moved) outside the tool since the snapshot was
    captured, ``undo_edit`` will re-create it at that path. Undo tracks
    content-per-path, not the file's presence.
    """
    if file_path not in undo_history:
        raise ValueError(f"No undo history available for {file_path} in this session.")

    previous_content = undo_history.pop(file_path)
    await sandbox.write_text(file_path, previous_content)
    return f"Reverted {file_path} to its previous in-memory snapshot."
