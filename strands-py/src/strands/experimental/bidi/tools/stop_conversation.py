"""Tool to gracefully stop a bidirectional connection.

.. deprecated::
    The ``stop_conversation`` tool is deprecated and will be removed in a future version.
    Use ``strands_tools.stop`` or set ``request_state["stop_event_loop"] = True`` in any custom tool instead.
"""

import warnings

from ....tools.decorator import tool


@tool
def stop_conversation() -> str:
    """Stop the bidirectional conversation gracefully.

    .. deprecated::
        Use ``strands_tools.stop`` or set ``request_state["stop_event_loop"] = True`` in a custom tool instead.

    Use ONLY when user says "stop conversation" exactly.
    Do NOT use for: "stop", "goodbye", "bye", "exit", "quit", "end" or other farewells or phrases.

    Returns:
        Success message confirming the conversation will end.
    """
    warnings.warn(
        "stop_conversation is deprecated and will be removed in a future version. "
        "Use strands_tools.stop or set request_state['stop_event_loop'] = True in any custom tool instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return "Ending conversation"
