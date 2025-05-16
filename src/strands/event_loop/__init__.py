"""This package provides the core event loop implementation for the agents SDK.

The event loop enables conversational AI agents to process messages, execute tools, and handle errors in a controlled,
iterative manner.
"""

from . import error_handler, event_loop, message_processor

__all__ = ["error_handler", "event_loop", "message_processor"]
