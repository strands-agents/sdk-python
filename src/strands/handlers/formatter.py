"""This module provides formatters for tool invocations and results."""

import sys
from datetime import datetime
from typing import Any


class EnhancedToolFormatter:
    """Enhanced formatter for tool invocations with colors and structured output."""

    # ANSI color codes
    COLORS = {
        "purple": "\033[35m",  # Purple/Magenta
        "blue": "\033[34m",  # Blue
        "gray": "\033[90m",  # Gray
        "white": "\033[37m",  # White
        "cyan": "\033[36m",  # Cyan
        "green": "\033[32m",  # Green
        "yellow": "\033[33m",  # Yellow
        "red": "\033[31m",  # Red
    }
    RESET = "\033[0m"

    def __init__(self) -> None:
        """Initialize the formatter."""
        self.use_colors = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

    def display_tool_call(self, tool_name: str, tool_input: dict, agent_name: str = "my_agent") -> None:
        """Display tool call with parameters.

        Args:
            tool_name: Name of the tool being called
            tool_input: Input parameters for the tool
            agent_name: Name of the agent calling the tool
        """
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        # Format tool call header
        if self.use_colors:
            header = (
                f"\n{self.COLORS['purple']}Calling Tool{self.RESET} | "
                f"{self.COLORS['blue']}{agent_name}{self.RESET} | "
                f"{self.COLORS['gray']}{tool_name}{self.RESET} - "
                f"{self.COLORS['white']}{timestamp}{self.RESET}"
            )
        else:
            header = f"\nCalling Tool | {agent_name} | {tool_name} - {timestamp}"

        # Format parameters
        params_str = self._format_args(tool_input)
        if self.use_colors:
            params = f"{self.COLORS['cyan']}Parameters:{self.RESET} {self.COLORS['white']}{params_str}{self.RESET}"
        else:
            params = f"Parameters: {params_str}"

        # Print both lines
        print(header)
        print(params)

    def _format_args(self, args: dict[str, Any] | str | None) -> str:
        """Format tool arguments for display.

        Args:
            args: Tool arguments (can be dict, string, or None)

        Returns:
            Formatted arguments string
        """
        if args is None:
            return "{}"

        # Handle string input (from streaming)
        if isinstance(args, str):
            if not args.strip():
                return "{}"

            # Try to parse as complete JSON
            try:
                import json

                parsed_args = json.loads(args)
                return json.dumps(parsed_args, ensure_ascii=False, separators=(",", ":"))
            except (json.JSONDecodeError, TypeError):
                pass

            # Handle partial JSON by completing braces
            args_clean = args.strip()
            if args_clean.startswith("{") and len(args_clean) > 1:
                open_braces = args_clean.count("{")
                close_braces = args_clean.count("}")
                missing_braces = open_braces - close_braces

                if missing_braces > 0:
                    completed_json = args_clean + "}" * missing_braces
                    try:
                        import json

                        parsed_args = json.loads(completed_json)
                        return json.dumps(parsed_args, ensure_ascii=False, separators=(",", ":"))
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Show partial content if completion fails
                return args_clean if len(args_clean) <= 100 else args_clean[:97] + "..."
            elif args_clean == "{":
                return "[参数加载中...]"
            else:
                return args_clean if len(args_clean) <= 100 else args_clean[:97] + "..."

        # Handle dict input
        if isinstance(args, dict):
            try:
                import json

                return json.dumps(args, ensure_ascii=False, separators=(",", ":"))
            except (TypeError, ValueError):
                return str(args)

    def format_tool_completion(
        self,
        tool_name: str,
        success: bool = True,
        result: Any = None,
        error: str | list | None = None,
        formatted_tool_name: str | None = None,
    ) -> str:
        """Format a tool completion message with boxed result content.

        Args:
            tool_name: Name of the tool that completed
            success: Whether the tool completed successfully
            result: Tool result (optional)
            error: Error message if failed (optional)
            formatted_tool_name: Pre-formatted tool name with context (optional)

        Returns:
            Formatted completion message with boxed content
        """
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        display_name = formatted_tool_name if formatted_tool_name else tool_name

        if success:
            action = "Tool Completed"
            # Extract result content for the box
            result_content = ""
            if result is not None:
                if isinstance(result, list) and result:
                    result_texts = []
                    for item in result:
                        if isinstance(item, dict) and "text" in item:
                            result_texts.append(item["text"])
                        else:
                            result_texts.append(str(item))

                    if result_texts:
                        result_content = "\n".join(result_texts)
                    else:
                        result_content = f"Content items: {len(result)}"
                else:
                    try:
                        import json

                        if hasattr(result, "__dict__"):
                            result_content = json.dumps(result.__dict__, ensure_ascii=False, indent=2)
                        else:
                            result_content = json.dumps(result, ensure_ascii=False, indent=2)
                    except (TypeError, ValueError):
                        result_content = str(result)
            else:
                result_content = "No result content"

            # Truncate result content to 500 characters
            if len(result_content) > 500:
                result_content = result_content[:497] + "..."
        else:
            action = "Tool Failed"
            # Extract error content for the box
            if isinstance(error, list):
                error_texts = []
                for item in error:
                    if isinstance(item, dict) and "text" in item:
                        error_texts.append(item["text"])
                    else:
                        error_texts.append(str(item))
                result_content = "\n".join(error_texts) if error_texts else str(error)
            else:
                result_content = str(error) if error else "Unknown error"

            # Truncate error content to 500 characters
            if len(result_content) > 500:
                result_content = result_content[:497] + "..."

        # Create the boxed result
        box_content = self._create_result_box(result_content, success)

        # Create header line
        status = "success" if success else "failed"

        # Apply colors to header
        if success:
            colored_header = (
                f"{self.COLORS['green']}{action}{self.RESET} | "
                f"{self.COLORS['gray']}{display_name}{self.RESET} - "
                f"{self.COLORS['white']}{timestamp} - status={status}{self.RESET}"
            )
        else:
            colored_header = (
                f"{self.COLORS['red']}{action}{self.RESET} | "
                f"{self.COLORS['gray']}{display_name}{self.RESET} - "
                f"{self.COLORS['white']}{timestamp} - status={status}{self.RESET}"
            )

        return f"{colored_header}\n{box_content}"

    def _create_result_box(self, content: str, success: bool = True) -> str:
        """Create a boxed display for result content.

        Args:
            content: Content to display in the box
            success: Whether this is a success or error result

        Returns:
            Boxed content string
        """
        # Split content into lines
        lines = content.split("\n")

        # Create box borders (only top, left, bottom - no right border)
        top_border = "┌─────────────────────────────────────────────────────────────────────────────"
        bottom_border = "└─────────────────────────────────────────────────────────────────────────────"

        # Create content lines with left border only
        content_lines = []
        for line in lines:
            content_lines.append(f"│ {line}")

        # If no content lines, add an empty line
        if not content_lines:
            content_lines.append("│")

        # Combine all parts
        box_lines = [top_border] + content_lines + [bottom_border]

        if not self.use_colors:
            return "\n".join(box_lines)

        # Apply colors to the box
        color = self.COLORS["green"] if success else self.COLORS["red"]
        colored_lines = []
        for line in box_lines:
            colored_lines.append(f"{color}{line}{self.RESET}")

        return "\n".join(colored_lines)
