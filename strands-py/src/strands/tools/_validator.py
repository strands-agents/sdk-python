"""Tool validation utilities."""

from ..tools.tools import InvalidToolUseNameException, validate_tool_use
from ..types.content import Message
from ..types.tools import ToolResult, ToolUse


def strip_input_parse_errors(message: Message) -> dict[str, str]:
    """Pop the transient ``inputParseError`` markers off a message's tool uses.

    Streaming attaches this marker when a tool's streamed input is not valid JSON. It is an internal,
    within-cycle bridge to tool validation and must not persist. Calling this before the assistant
    message is appended to history keeps the marker out of the session store and out of what is sent
    back to the model, while the returned mapping lets validation still emit the error tool result.

    Args:
        message: The assistant message whose tool uses may carry parse-error markers.

    Returns:
        Mapping of ``toolUseId`` to the parse-error detail for every tool use that carried a marker.
    """
    parse_errors: dict[str, str] = {}
    for content in message.get("content", []):
        if isinstance(content, dict) and "toolUse" in content:
            tool_use = content["toolUse"]
            error = tool_use.pop("inputParseError", None)
            if error:
                parse_errors[tool_use["toolUseId"]] = error
    return parse_errors


def validate_and_prepare_tools(
    message: Message,
    tool_uses: list[ToolUse],
    tool_results: list[ToolResult],
    invalid_tool_use_ids: list[str],
    input_parse_errors: dict[str, str] | None = None,
) -> None:
    """Validate tool uses and prepare them for execution.

    Args:
        message: Current message.
        tool_uses: List to populate with tool uses.
        tool_results: List to populate with tool results for invalid tools.
        invalid_tool_use_ids: List to populate with invalid tool use IDs.
        input_parse_errors: Mapping of ``toolUseId`` to parse-error detail for tool uses whose streamed
            input was not valid JSON, carried out-of-band so the marker never persists in history. Falls
            back to an ``inputParseError`` marker still present on the tool use when not provided.
    """
    input_parse_errors = input_parse_errors or {}

    # Extract tool uses from message
    for content in message["content"]:
        if isinstance(content, dict) and "toolUse" in content:
            tool_uses.append(content["toolUse"])

    # Validate tool uses
    # Avoid modifying original `tool_uses` variable during iteration
    tool_uses_copy = tool_uses.copy()
    for tool in tool_uses_copy:
        # Prefer the out-of-band map (stripped before persistence); fall back to a marker still on the
        # tool use for callers that did not strip it first.
        parse_error = input_parse_errors.get(tool["toolUseId"]) or tool.pop("inputParseError", None)
        if parse_error:
            tool_uses.remove(tool)
            invalid_tool_use_ids.append(tool["toolUseId"])
            tool_uses.append(tool)
            tool_results.append(
                {
                    "toolUseId": tool["toolUseId"],
                    "status": "error",
                    "content": [{"text": f"Error: {parse_error}"}],
                }
            )
            continue

        try:
            validate_tool_use(tool)
        except InvalidToolUseNameException as e:
            # Return invalid name error as ToolResult to the LLM as context
            # The replacement of the tool name to INVALID_TOOL_NAME happens in streaming.py now
            tool_uses.remove(tool)
            invalid_tool_use_ids.append(tool["toolUseId"])
            tool_uses.append(tool)
            tool_results.append(
                {
                    "toolUseId": tool["toolUseId"],
                    "status": "error",
                    "content": [{"text": f"Error: {str(e)}"}],
                }
            )
