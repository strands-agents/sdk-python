"""Built-in tools for bidirectional agents.

Note: To stop a bidirectional conversation, use the standard `stop` tool from strands_tools:

    from strands_tools import stop
    agent = BidiAgent(tools=[stop, ...])

The stop tool sets `request_state["stop_event_loop"] = True`, which signals the
BidiAgent to gracefully close the connection.
"""
