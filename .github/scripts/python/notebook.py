"""Notebook management tool for Strands Agents.

This module provides comprehensive notebook operations for managing text-based notebooks
within agent workflows. Enables persistent note-taking, documentation, and context
preservation across agent sessions.

Key Features:
1. Create and manage multiple named notebooks
2. Write content using string replacement or line insertion
3. Read entire notebooks or specific line ranges
4. List all available notebooks with metadata
5. Clear notebook contents when needed
6. Rich console output with formatted panels and tables
7. Agent state persistence for session continuity

Usage Examples:
```python
from strands import Agent
from tools.notebook import notebook

agent = Agent(tools=[notebook])

# Create a new notebook with initial content
result = agent.tool.notebook(
    mode="create",
    name="research_notes",
    new_str="# Research Notes\n\nKey findings and observations..."
)

# Write to notebook using line insertion
result = agent.tool.notebook(
    mode="write",
    name="research_notes",
    insert_line=-1,  # Append to end
    new_str="- Important discovery about AI behavior patterns"
)

# Read specific lines from notebook
result = agent.tool.notebook(
    mode="read",
    name="research_notes",
    read_range=[1, 5]  # Read first 5 lines
)

# Replace text in notebook
result = agent.tool.notebook(
    mode="write",
    name="research_notes",
    old_str="[ ] Todo item",
    new_str="[x] Completed todo item"
)

# List all notebooks
result = agent.tool.notebook(mode="list")

# Clear notebook contents
result = agent.tool.notebook(mode="clear", name="research_notes")
```
"""

from typing import Any, Literal

from rich import box
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from strands import ToolContext, tool
from strands_tools.utils import console_util


@tool(context=True)
def notebook(
    mode: Literal["create", "list", "read", "write", "clear"],
    name: str = "default",
    read_range: list[int] | None = None,
    old_str: str | None = None,
    new_str: str | None = None,
    insert_line: str | int | None = None,
    tool_context: ToolContext | None = None,
) -> str:
    """
    Notebook tool for managing text notebooks.

    This tool provides a comprehensive interface for creating, reading, writing, listing,
    and deleting text notebooks. Start writing notes in the default notebook which is avaiable
    from the start, or create new notebooks to record notes on additional topics or tasks.

    Command Details:
    --------------
    1. write:
       • Supports two types of write operations:
         - String replacement: Uses old_str and new_str parameters
         - Line insertion: Uses insert_line and new_str parameters

    2. read:
       • Reads contents of a notebook
       • Supports reading specific line numbers with read_range parameter

    3. create:
       • Creates a new notebook with the specified name
       • Optionally initializes with content using new_str parameter
       • Defaults to empty content if new_str not provided

    4. list:
       • Lists all available notebook names
       • Returns comma-separated list of notebook names

    5. clear:
       • Clears the contents of a notebook

    Args:
        mode: The operation to perform: `create`, `list`, `read`, `write`, `clear`.
        name: Name of the notebook to operate on. Defaults to "default".
        read_range: Optional parameter of `view` command. Line range to show [start, end]. Supports negative indices.
        old_str: String to replace in write mode when doing text replacement.
        new_str: New string for replacement or insertion operations.
        insert_line: Line number (int) or search text (str) for insertion point in write mode.
            Supports negative indices.

    Returns:
        Dict containing status and response content in the format:
        {
            "status": "success|error",
            "content": [{"text": "Response message"}]
        }

        Success case: Returns details about the operation performed
        Error case: Returns information about what went wrong

    Examples:
        1. Create a notebook:
           notebook(mode="create", name="notes")

        2. List all notebooks:
           notebook(mode="list")

        3. Read entire notebook:
           notebook(mode="read", name="notes")

        4. Read specific lines:
           notebook(mode="read", name="notes", read_range=[1, 5])

        5. Replace text:
           notebook(mode="write", name="notes", old_str="[] Update the calendar", new_str="[x] Update the calendar")

        6. Insert text after line 5:
           notebook(mode="write", name="notes", insert_line=5, new_str="inserted text")

        7. Insert text at end of notebook:
           notebook(mode="write", name="notes", insert_line=-1, new_str="Appended text")

        7. Insert text after finding a line:
           notebook(mode="write", name="notes", insert_line="def function", new_str="# comment")

        8. Clear notebook:
           notebook(mode="clear", name="notes")
    """
    console = console_util.create()
    if tool_context is None:
        raise ValueError("Tool context is required")
    agent = tool_context.agent

    if agent.state.get("notebooks") is None:
        agent.state.set("notebooks", {"default": ""})

    notebooks: dict[str, Any] = agent.state.get("notebooks")

    if mode == "create":
        notebooks[name] = new_str if new_str else ""
        message = f"Created notebook '{name}'" + (" with specified content" if new_str else " (empty)")
        console.print(
            Panel(
                escape(message + f":\n{new_str}" if new_str else ""),
                title="[bold green]Success",
                border_style="green",
            )
        )
        agent.state.set("notebooks", notebooks)
        return message

    elif mode == "list":
        table = Table(title="📚 Available Notebooks", box=box.DOUBLE)
        table.add_column("Name", style="cyan")
        table.add_column("Lines", style="yellow")
        table.add_column("Status", style="green")

        for nb_name in notebooks.keys():
            line_count = len(notebooks[nb_name].split("\n")) if notebooks[nb_name] else 0
            status = "Empty" if line_count == 0 else "Has content"
            table.add_row(nb_name, str(line_count), status)

        console.print(table)
        return f"Notebooks: {', '.join(notebooks.keys())}"

    elif mode == "read":
        if name not in notebooks:
            error_msg = f"Notebook '{name}' not found"
            console.print(Panel(escape(error_msg), title="[bold red]Error", border_style="red"))
            raise ValueError(error_msg)

        content = notebooks[name]
        if read_range:
            lines = content.split("\n")
            start, end = read_range
            # Handle negative indices
            if start < 0:
                start = len(lines) + start + 1
            if end < 0:
                end = len(lines) + end + 1

            selected_lines = []
            for line_num in range(start, end + 1):
                if 1 <= line_num <= len(lines):
                    selected_lines.append(f"{line_num}: {lines[line_num - 1]}")

            result = "\n".join(selected_lines) if selected_lines else "No valid lines found"
            console.print(
                Panel(
                    escape(result),
                    title=f"[bold green]📖 {name} (lines {start}-{end})",
                    border_style="blue",
                )
            )
            return result

        result = content if content else f"Notebook '{name}' is empty"
        console.print(Panel(escape(result), title=f"[bold green]📖 {name}", border_style="blue"))
        return result

    elif mode == "write":
        if name not in notebooks:
            error_msg = f"Notebook '{name}' not found"
            console.print(Panel(escape(error_msg), title="[bold red]Error", border_style="red"))
            raise ValueError(error_msg)

        # String replacement
        if old_str is not None and new_str is not None:
            if old_str not in notebooks[name]:
                error_msg = f"String '{old_str}' not found in notebook '{name}'"
                console.print(Panel(escape(error_msg), title="[bold red]Error", border_style="red"))
                raise ValueError(error_msg)

            notebooks[name] = notebooks[name].replace(old_str, new_str)
            agent.state.set("notebooks", notebooks)

            # Create git-style diff
            old_lines = old_str.split("\n")
            new_lines = new_str.split("\n")
            diff_lines = []

            for line in old_lines:
                diff_lines.append(f"[red]-{escape(line)}[/red]")
            for line in new_lines:
                diff_lines.append(f"[green]+{escape(line)}[/green]")

            diff_content = "\n".join(diff_lines)
            console.print(Panel(diff_content, title="[bold yellow]📝 Diff", border_style="yellow"))

            message = f"Replaced text in notebook '{name}'"
            console.print(Panel(escape(message), title="[bold green]Success", border_style="green"))
            return message

        # Line insertion
        elif insert_line is not None and new_str is not None:
            lines = notebooks[name].split("\n")

            # Check if string represents a number first
            if isinstance(insert_line, str):
                try:
                    insert_line = int(insert_line)
                except ValueError:
                    pass  # Keep as string for text search

            if isinstance(insert_line, str):
                line_num = -1
                for i, line in enumerate(lines):
                    if insert_line in line:
                        line_num = i
                        break
                if line_num == -1:
                    error_msg = f"Text '{insert_line}' not found in notebook '{name}'"
                    console.print(
                        Panel(
                            escape(error_msg),
                            title="[bold red]Error",
                            border_style="red",
                        )
                    )
                    raise ValueError(error_msg)
            else:
                # Handle negative indices
                if insert_line < 0:
                    line_num = len(lines) + insert_line
                else:
                    line_num = insert_line - 1

            if 0 <= line_num <= len(lines):
                lines.insert(line_num + 1, new_str)
                notebooks[name] = "\n".join(lines)
                agent.state.set("notebooks", notebooks)
                message = f"Inserted text at line {line_num + 2} in notebook '{name}'"
                console.print(
                    Panel(
                        escape(message),
                        title="[bold green]Success",
                        border_style="green",
                    )
                )
                console.print(
                    Panel(
                        escape(notebooks[name]),
                        title=f"[bold blue]📝 {name} Content",
                        border_style="blue",
                    )
                )
                return message
            else:
                error_msg = f"Line number {insert_line} out of range"
                console.print(Panel(escape(error_msg), title="[bold red]Error", border_style="red"))
                raise ValueError(error_msg)

        # No valid operation provided
        else:
            error_msg = "No valid write operation specified"
            console.print(Panel(escape(error_msg), title="[bold red]Error", border_style="red"))
            raise ValueError(error_msg)

    elif mode == "clear":
        if name not in notebooks:
            error_msg = f"Notebook '{name}' not found"
            console.print(Panel(escape(error_msg), title="[bold red]Error", border_style="red"))
            raise ValueError(error_msg)
        notebooks[name] = ""
        agent.state.set("notebooks", notebooks)
        message = f"Cleared notebook '{name}'"
        console.print(Panel(escape(message), title="[bold green]Success", border_style="green"))
        return message
