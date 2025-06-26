"""Strands Agent executor for the A2A protocol.

This module provides the StrandsA2AExecutor class, which adapts a Strands Agent
to be used as an executor in the A2A protocol. It handles the execution of agent
requests and the conversion of Strands Agent responses to A2A events.
"""

import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import UnsupportedOperationError
from a2a.utils import new_agent_text_message
from a2a.utils.errors import ServerError

from ...agent.agent import Agent as SAAgent
from ...agent.agent_result import AgentResult as SAAgentResult

log = logging.getLogger(__name__)


class StrandsA2AExecutor(AgentExecutor):
    """Executor that adapts a Strands Agent to the A2A protocol."""

    def __init__(self, agent: SAAgent):
        """Initialize a StrandsA2AExecutor.

        Args:
            agent: The Strands Agent to adapt to the A2A protocol.
        """
        self.agent = agent

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute a request using the Strands Agent and send the response as A2A events.

        This method executes the user's input using the Strands Agent and converts
        the agent's response to A2A events, which are then sent to the event queue.

        Args:
            context: The A2A request context, containing the user's input and other metadata.
            event_queue: The A2A event queue, used to send response events.
        """
        result: SAAgentResult = self.agent(context.get_user_input())
        if result.message and "content" in result.message:
            for content_block in result.message["content"]:
                if "text" in content_block:
                    await event_queue.enqueue_event(new_agent_text_message(content_block["text"]))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel an ongoing execution.

        This method is called when a request is cancelled. Currently, cancellation
        is not supported, so this method raises an UnsupportedOperationError.

        Args:
            context: The A2A request context.
            event_queue: The A2A event queue.

        Raises:
            ServerError: Always raised with an UnsupportedOperationError, as cancellation
                is not currently supported.
        """
        raise ServerError(error=UnsupportedOperationError())
