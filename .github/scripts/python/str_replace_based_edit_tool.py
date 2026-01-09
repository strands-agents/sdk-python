"""Text editor tool for Strands Agents.

A minimal implementation of Claude's text editor tool that supports:
- view: Read file contents or list directory contents
- str_replace: Replace text in files
- create: Create new files
- insert: Insert text at specific line numbers

Based on Claude's text_editor_20250728 specification.
"""

from pathlib import Path
from typing import List, Optional

from rich.markup import escape
from rich.panel import Panel
from strands import tool
from strands_tools.utils import console_util

console = console_util.create()


@tool
def str_replace_based_edit_tool(
    command: str,
    path: str,
    old_str: str | None = None,
    new_str: str | None  = None,
    file_text: str | None = None,
    insert_line: str | None = None,
    view_range: list[int] | None = None,
) -> str:
    """Text editor tool for viewing and modifying files.
    
    Args:
        command: The command to execute ("view", "str_replace", "create", "insert")
        path: Path to the file or directory
        old_str: Text to replace (for str_replace command)
        new_str: Replacement text (for str_replace and insert commands)
        file_text: Content for new file (for create command)
        insert_line: Line number to insert after (for insert command)
        view_range: [start_line, end_line] for viewing specific lines (for view command)
    
    Returns:
        Result of the operation
    """
    try:
        console.print(Panel(f"Command: {command}, Path: {path}", title="[bold blue]Text Editor", border_style="blue"))
        
        if command == "view":
            return _handle_view(path, view_range)
        elif command == "str_replace":
            if old_str is None or new_str is None:
                error_msg = "Error: str_replace requires both old_str and new_str parameters"
                console.print(Panel(escape(error_msg), title="[bold red]Error", border_style="red"))
                return error_msg
            return _handle_str_replace(path, old_str, new_str)
        elif command == "create":
            if file_text is None:
                error_msg = "Error: create requires file_text parameter"
                console.print(Panel(escape(error_msg), title="[bold red]Error", border_style="red"))
                return error_msg
            return _handle_create(path, file_text)
        elif command == "insert":
            if new_str is None or insert_line is None:
                error_msg = "Error: insert requires both new_str and insert_line parameters"
                console.print(Panel(escape(error_msg), title="[bold red]Error", border_style="red"))
                return error_msg
            return _handle_insert(path, new_str, insert_line)
        else:
            error_msg = f"Error: Unknown command '{command}'"
            console.print(Panel(escape(error_msg), title="[bold red]Error", border_style="red"))
            return error_msg
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        console.print(Panel(escape(error_msg), title="[bold red]Error", border_style="red"))
        return error_msg


def _handle_view(path: str, view_range: Optional[List[int]] = None) -> str:
    """Handle view command to read files or list directories."""
    path_obj = Path(path)
    
    if not path_obj.exists():
        return f"Error: Path '{path}' does not exist"
    
    if path_obj.is_dir():
        # List directory contents
        try:
            items = []
            for item in sorted(path_obj.iterdir()):
                if item.is_dir():
                    items.append(f"{item.name}/")
                else:
                    items.append(item.name)
            return "\n".join(items)
        except PermissionError:
            return f"Error: Permission denied accessing directory '{path}'"
    
    elif path_obj.is_file():
        # Read file contents
        try:
            with open(path_obj, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Apply view_range if specified
            if view_range:
                start_line, end_line = view_range
                # Convert to 0-based indexing
                start_idx = max(0, start_line - 1) if start_line > 0 else 0
                end_idx = len(lines) if end_line == -1 else min(len(lines), end_line)
                lines = lines[start_idx:end_idx]
                start_line_num = start_idx + 1
            else:
                start_line_num = 1
            
            # Add line numbers
            numbered_lines = []
            for i, line in enumerate(lines):
                line_num = start_line_num + i
                numbered_lines.append(f"{line_num}: {line.rstrip()}")
            
            return "\n".join(numbered_lines)
        except UnicodeDecodeError:
            return f"Error: Cannot read '{path}' - file appears to be binary"
        except PermissionError:
            return f"Error: Permission denied reading file '{path}'"
    
    else:
        return f"Error: '{path}' is not a regular file or directory"


def _handle_str_replace(path: str, old_str: str, new_str: str) -> str:
    """Handle str_replace command to replace text in a file."""
    path_obj = Path(path)
    
    if not path_obj.exists():
        return f"Error: File '{path}' does not exist"
    
    if not path_obj.is_file():
        return f"Error: '{path}' is not a file"
    
    try:
        # Read file content
        with open(path_obj, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if old_str exists
        if old_str not in content:
            return f"Error: Text '{old_str}' not found in file"
        
        # Count occurrences
        count = content.count(old_str)
        if count > 1:
            return f"Error: Text '{old_str}' appears {count} times in file. Please be more specific."
        
        # Replace text
        new_content = content.replace(old_str, new_str)
        
        # Write back to file
        with open(path_obj, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        success_msg = f"Successfully replaced text in '{path}'"
        console.print(Panel(escape(success_msg), title="[bold green]Success", border_style="green"))
        return success_msg
    
    except UnicodeDecodeError:
        return f"Error: Cannot modify '{path}' - file appears to be binary"
    except PermissionError:
        return f"Error: Permission denied modifying file '{path}'"


def _handle_create(path: str, file_text: str) -> str:
    """Handle create command to create a new file."""
    path_obj = Path(path)
    
    # Create parent directories if they don't exist
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path_obj, 'w', encoding='utf-8') as f:
            f.write(file_text)
        
        success_msg = f"Successfully created file '{path}'"
        console.print(Panel(escape(success_msg), title="[bold green]Success", border_style="green"))
        return success_msg
    
    except PermissionError:
        return f"Error: Permission denied creating file '{path}'"


def _handle_insert(path: str, new_str: str, insert_line: int) -> str:
    """Handle insert command to insert text at a specific line."""
    path_obj = Path(path)
    
    if not path_obj.exists():
        return f"Error: File '{path}' does not exist"
    
    if not path_obj.is_file():
        return f"Error: '{path}' is not a file"
    
    try:
        # Read file lines
        with open(path_obj, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Insert new text
        if insert_line == 0:
            # Insert at beginning
            lines.insert(0, new_str + '\n')
        elif insert_line >= len(lines):
            # Insert at end
            lines.append(new_str + '\n')
        else:
            # Insert after specified line (1-based indexing)
            lines.insert(insert_line, new_str + '\n')
        
        # Write back to file
        with open(path_obj, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        success_msg = f"Successfully inserted text in '{path}' at line {insert_line + 1}"
        console.print(Panel(escape(success_msg), title="[bold green]Success", border_style="green"))
        return success_msg
    
    except UnicodeDecodeError:
        return f"Error: Cannot modify '{path}' - file appears to be binary"
    except PermissionError:
        return f"Error: Permission denied modifying file '{path}'"