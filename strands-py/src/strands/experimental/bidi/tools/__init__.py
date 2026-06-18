"""Built-in tools for bidirectional agents.

.. deprecated::
    The built-in ``stop_conversation`` tool is deprecated. Use ``strands_tools.stop`` or set
    ``request_state["stop_event_loop"] = True`` in any custom tool instead.

To stop a bidirectional conversation, use the standard ``stop`` tool from strands_tools::

    from strands_tools import stop
    agent = BidiAgent(tools=[stop, ...])

The stop tool sets ``request_state["stop_event_loop"] = True``, which signals the
BidiAgent to gracefully close the connection.
"""

from .stop_conversation import stop_conversation

__all__ = ["stop_conversation"]
