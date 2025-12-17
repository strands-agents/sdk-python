from rich.markup import escape
from rich.panel import Panel
from strands import tool
from strands.types.tools import ToolContext
from strands_tools.utils import console_util

@tool(context=True)
def handoff_to_user(message: str, tool_context: ToolContext) -> str:
    """
    Hand off control to the user with a message.

    Args:
        message: The message to give to the user

    Returns:
        The users response after handing back control
    """
    console = console_util.create()
    
    console.print(
        Panel(
            escape(message),
            title="[bold yellow]🤝 Handoff to User",
            border_style="yellow",
        )
    )
    
    request_state = {
        "stop_event_loop": True
    }
    tool_context.invocation_state["request_state"] = request_state

    # Return an empty string as this will break out of the event loop
    return ""