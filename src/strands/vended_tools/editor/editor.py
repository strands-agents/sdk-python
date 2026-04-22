"""File editor tool implementation.

Provides view, create, str_replace, insert, and undo_edit operations on files
in the agent's sandbox. The tool delegates all file I/O to the sandbox's
``read_file``, ``write_file``, and ``list_files`` methods.

The tool shape matches Anthropic's ``text_editor`` built-in tool — 5 commands,
7 parameters. This means models trained on Anthropic's tool spec will work
well with this tool out of the box.

Configuration keys (set via ``agent.state.set("strands_editor_tool", {...})``):

- ``max_file_size`` (int): Maximum file size in bytes for read operations.
  Default: 1048576 (1 MB).
- ``require_absolute_paths`` (bool): When True, rejects relative paths and
  paths containing ``..``. When False (the default), paths are passed through
  to the sandbox without filesystem-level validation — the sandbox decides
  what a path means. Default: False.
"""

import logging
from typing import Any, Literal

from ...tools.decorator import tool
from ...types.tools import ToolContext

logger = logging.getLogger(__name__)

#: State key for editor tool configuration in agent.state
STATE_KEY = "strands_editor_tool"

#: State key for undo history (internal)
_UNDO_STATE_KEY = "_strands_editor_undo"

#: Default maximum file size (1 MB)
DEFAULT_MAX_FILE_SIZE = 1_048_576

#: Number of context lines to show around edits
SNIPPET_LINES = 4

#: Maximum directory listing depth
MAX_DIRECTORY_DEPTH = 2


def _get_config(tool_context: ToolContext) -> dict[str, Any]:
    """Read editor tool configuration from agent state."""
    return tool_context.agent.state.get(STATE_KEY) or {}


def _make_output(content: str, descriptor: str, init_line: int = 1) -> str:
    """Format file content with line numbers (cat -n style).

    Args:
        content: The file content to format.
        descriptor: Description of what is being shown (e.g., file path).
        init_line: Starting line number.

    Returns:
        Formatted output with line numbers.
    """
    # Expand tabs to spaces
    content = content.replace("\t", "        ")
    lines = content.split("\n")
    numbered = []
    for i, line in enumerate(lines):
        line_num = i + init_line
        numbered.append(f"{line_num:>6}  {line}")
    return f"Here's the result of running `cat -n` on {descriptor}:\n" + "\n".join(numbered) + "\n"


def _save_undo(tool_context: ToolContext, path: str, content: str) -> None:
    """Save file content for undo.

    Args:
        tool_context: The tool context providing access to agent state.
        path: The file path.
        content: The file content before modification.
    """
    undo_state = tool_context.agent.state.get(_UNDO_STATE_KEY) or {}
    undo_state[path] = content
    tool_context.agent.state.set(_UNDO_STATE_KEY, undo_state)


def _get_undo(tool_context: ToolContext, path: str) -> str | None:
    """Get saved undo content for a file.

    Args:
        tool_context: The tool context providing access to agent state.
        path: The file path.

    Returns:
        The saved content, or None if no undo is available.
    """
    undo_state = tool_context.agent.state.get(_UNDO_STATE_KEY) or {}
    return undo_state.get(path)


@tool(context=True)
async def editor(
    command: Literal["view", "create", "str_replace", "insert", "undo_edit"],
    path: str,
    file_text: str | None = None,
    old_str: str | None = None,
    new_str: str | None = None,
    insert_line: int | None = None,
    view_range: list[int] | None = None,
    tool_context: ToolContext = None,  # type: ignore[assignment]
) -> str:
    """View, create, and edit files in the agent's sandbox.

    Commands:

    - **view**: Display file contents with line numbers, or list directory contents.
      Use ``view_range`` as ``[start_line, end_line]`` (1-indexed, -1 for end of file)
      to view a specific range.
    - **create**: Create a new file with ``file_text`` content. Fails if file exists.
    - **str_replace**: Replace ``old_str`` with ``new_str`` in the file.
      ``old_str`` must match exactly once in the file (uniqueness enforced).
    - **insert**: Insert ``new_str`` at ``insert_line`` (0-indexed line number).
    - **undo_edit**: Revert the last edit to the file at ``path``.

    File operations go through the agent's sandbox. By default, paths are passed
    through to the sandbox as-is — the sandbox decides what a path means. Set
    ``require_absolute_paths: true`` in ``strands_editor_tool`` config to enforce
    absolute paths and block directory traversal.

    Configuration is read from ``agent.state.get("strands_editor_tool")``:

    - ``max_file_size``: Maximum file size in bytes (default: 1 MB).
    - ``require_absolute_paths``: Reject relative paths and ``..`` (default: False).

    Args:
        command: The operation to perform.
        path: Path to the file or directory.
        file_text: Content for new file (required for ``create``).
        old_str: String to find and replace (required for ``str_replace``).
            Must appear exactly once in the file.
        new_str: Replacement string for ``str_replace``, or text to insert for ``insert``.
        insert_line: Line number for insertion (0-indexed, required for ``insert``).
        view_range: Line range for view as ``[start, end]``. 1-indexed.
            Use -1 for end to mean end of file.
        tool_context: Framework-injected tool context.

    Returns:
        Result of the operation — file contents, success message, or error.
    """
    config = _get_config(tool_context)
    sandbox = tool_context.agent.sandbox

    # Path validation is opt-in. By default, paths are passed straight through
    # to the sandbox without filesystem-level validation. This allows sandboxes
    # like S3Sandbox to use relative keys (e.g., "hello.txt") as paths.
    if config.get("require_absolute_paths"):
        import os

        if not os.path.isabs(path):
            suggested = os.path.abspath(path)
            return f"Error: The path {path} is not an absolute path. Maybe you meant {suggested}?"
        if ".." in path:
            return "Error: Path traversal (..) is not allowed."

    try:
        if command == "view":
            return await _handle_view(sandbox, config, path, view_range)
        elif command == "create":
            if file_text is None:
                return "Error: Parameter `file_text` is required for command: create"
            return await _handle_create(sandbox, tool_context, path, file_text)
        elif command == "str_replace":
            if old_str is None:
                return "Error: Parameter `old_str` is required for command: str_replace"
            return await _handle_str_replace(sandbox, tool_context, config, path, old_str, new_str or "")
        elif command == "insert":
            if insert_line is None:
                return "Error: Parameter `insert_line` is required for command: insert"
            if new_str is None:
                return "Error: Parameter `new_str` is required for command: insert"
            return await _handle_insert(sandbox, tool_context, config, path, insert_line, new_str)
        elif command == "undo_edit":
            return await _handle_undo(sandbox, tool_context, path)

        return f"Error: Unknown command: {command}"  # type: ignore[unreachable]
    except NotImplementedError as e:
        return f"Error: Sandbox does not support this operation — {e}"
    except Exception as e:
        return f"Error: {e}"


async def _handle_view(sandbox: Any, config: dict[str, Any], path: str, view_range: list[int] | None) -> str:
    """Handle the view command."""
    # Check if path is a directory
    try:
        entries = await sandbox.list_files(path)
        # It's a directory
        if view_range:
            return "Error: The `view_range` parameter is not allowed when `path` points to a directory."
        items = sorted(f"{e.name}/" if e.is_dir else e.name for e in entries if e.name not in (".", ".."))
        return (
            f"Here's the files and directories up to 2 levels deep in {path}, "
            f"excluding hidden items:\n" + "\n".join(items) + "\n"
        )
    except (FileNotFoundError, OSError):
        pass  # Not a directory, try as file

    # Read file
    max_size = config.get("max_file_size", DEFAULT_MAX_FILE_SIZE)
    try:
        content = (await sandbox.read_file(path)).decode("utf-8")
    except FileNotFoundError:
        return f"Error: The path {path} does not exist. Please provide a valid path."
    except UnicodeDecodeError:
        return f"Error: The file {path} is not a text file (cannot decode as UTF-8)."

    # Check size
    if len(content.encode("utf-8")) > max_size:
        return f"Error: File size exceeds maximum allowed size ({max_size} bytes)."

    if view_range is None:
        return _make_output(content, path)

    # Validate and apply view range
    lines = content.split("\n")
    n_lines = len(lines)

    if len(view_range) != 2:
        return "Error: `view_range` must be a list of two integers [start, end]."

    start, end = view_range[0], view_range[1]

    if start < 1 or start > n_lines:
        return (
            f"Error: Invalid `view_range`: [{start}, {end}]. First element `{start}` should be within [1, {n_lines}]."
        )
    if end != -1 and end > n_lines:
        return f"Error: Invalid `view_range`: [{start}, {end}]. Second element `{end}` should be <= {n_lines}."
    if end != -1 and end < start:
        return f"Error: Invalid `view_range`: [{start}, {end}]. Second element must be >= first element."

    if end == -1:
        selected = lines[start - 1 :]
    else:
        selected = lines[start - 1 : end]

    return _make_output("\n".join(selected), path, init_line=start)


async def _handle_create(sandbox: Any, tool_context: ToolContext, path: str, file_text: str) -> str:
    """Handle the create command."""
    # Check if file already exists
    try:
        await sandbox.read_file(path)
        return f"Error: File already exists at: {path}. Cannot overwrite with `create`. Use `str_replace` to edit."
    except (FileNotFoundError, OSError):
        pass  # File doesn't exist, good

    await sandbox.write_file(path, file_text.encode("utf-8"))
    return f"File created successfully at: {path}"


async def _handle_str_replace(
    sandbox: Any,
    tool_context: ToolContext,
    config: dict[str, Any],
    path: str,
    old_str: str,
    new_str: str,
) -> str:
    """Handle the str_replace command."""
    try:
        content = (await sandbox.read_file(path)).decode("utf-8")
    except FileNotFoundError:
        return f"Error: The path {path} does not exist."

    # Expand tabs for matching
    content = content.replace("\t", "        ")
    expanded_old = old_str.replace("\t", "        ")
    expanded_new = new_str.replace("\t", "        ")

    # Count occurrences — MUST be exactly 1
    count = content.count(expanded_old)

    if count == 0:
        return f"Error: No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}."

    if count > 1:
        # Find line numbers of all occurrences
        lines = content.split("\n")
        line_nums = []
        for i, line in enumerate(lines):
            if expanded_old in line:
                line_nums.append(i + 1)
        # Also check multi-line matches
        if not line_nums:
            # old_str spans multiple lines, find approximate locations
            idx = 0
            while True:
                idx = content.find(expanded_old, idx)
                if idx == -1:
                    break
                line_num = content[:idx].count("\n") + 1
                line_nums.append(line_num)
                idx += 1
        return (
            f"Error: No replacement was performed. Multiple occurrences ({count}) of old_str "
            f"in lines {line_nums}. Please ensure old_str is unique."
        )

    # Save undo state
    _save_undo(tool_context, path, content)

    # Perform replacement
    new_content = content.replace(expanded_old, expanded_new, 1)

    # Write back
    await sandbox.write_file(path, new_content.encode("utf-8"))

    # Generate snippet around the change
    replace_idx = content.find(expanded_old)
    replace_line = content[:replace_idx].count("\n")
    inserted_lines = expanded_new.count("\n") + 1
    original_lines = expanded_old.count("\n") + 1
    line_diff = inserted_lines - original_lines

    new_lines = new_content.split("\n")
    start = max(0, replace_line - SNIPPET_LINES)
    end = min(len(new_lines), replace_line + SNIPPET_LINES + line_diff + 1)
    snippet = "\n".join(new_lines[start:end])

    return (
        f"The file {path} has been edited. "
        + _make_output(snippet, f"a snippet of {path}", init_line=start + 1)
        + "Review the changes and make sure they are as expected. Edit the file again if necessary."
    )


async def _handle_insert(
    sandbox: Any,
    tool_context: ToolContext,
    config: dict[str, Any],
    path: str,
    insert_line: int,
    new_str: str,
) -> str:
    """Handle the insert command."""
    try:
        content = (await sandbox.read_file(path)).decode("utf-8")
    except FileNotFoundError:
        return f"Error: The path {path} does not exist."

    # Expand tabs
    content = content.replace("\t", "        ")
    expanded_new = new_str.replace("\t", "        ")

    lines = content.split("\n")
    n_lines = len(lines)

    if insert_line < 0 or insert_line > n_lines:
        return f"Error: Invalid `insert_line`: {insert_line}. Should be within [0, {n_lines}]."

    # Save undo state
    _save_undo(tool_context, path, content)

    # Insert
    new_str_lines = expanded_new.split("\n")
    if content == "":
        new_lines = new_str_lines
    else:
        new_lines = lines[:insert_line] + new_str_lines + lines[insert_line:]

    new_content = "\n".join(new_lines)
    await sandbox.write_file(path, new_content.encode("utf-8"))

    # Generate snippet
    start = max(0, insert_line - SNIPPET_LINES)
    end = min(len(new_lines), insert_line + len(new_str_lines) + SNIPPET_LINES)
    snippet = "\n".join(new_lines[start:end])

    return (
        f"The file {path} has been edited. "
        + _make_output(snippet, "a snippet of the edited file", init_line=start + 1)
        + "Review the changes and make sure they are as expected. Edit the file again if necessary."
    )


async def _handle_undo(sandbox: Any, tool_context: ToolContext, path: str) -> str:
    """Handle the undo_edit command."""
    previous_content = _get_undo(tool_context, path)
    if previous_content is None:
        return f"Error: No edit history found for {path}."

    # Read current content for future undo
    try:
        current = (await sandbox.read_file(path)).decode("utf-8")
    except FileNotFoundError:
        current = ""

    # Write the previous content back
    await sandbox.write_file(path, previous_content.encode("utf-8"))

    # Save current as new undo (so undo is toggleable)
    _save_undo(tool_context, path, current)

    return f"Successfully reverted last edit to {path}."
