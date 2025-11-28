"""Tool to gracefully stop a bidirectional connection."""

from ....tools.decorator import tool


@tool
def stop_connection() -> dict:
    """Stop the bidirectional conversation gracefully.

    Use this tool when the user wants to end the conversation, such as when they say:
    goodbye, bye, end conversation, stop, exit, quit, that's all, or I'm done.

    Returns:
        Success message confirming the conversation will end
    """
    return {
        "status": "success",
        "content": [{"text": "Ending conversation"}],
    }
