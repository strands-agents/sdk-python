"""A2A Module."""

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
    """StrandsA2AExecutor."""

    def __init__(self, agent: SAAgent):
        """StrandsA2AExecutor constructor."""
        self.agent = agent

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute."""
        result: SAAgentResult = self.agent(context.get_user_input())
        if result.message and "content" in result.message:
            for content_block in result.message["content"]:
                if "text" in content_block:
                    await event_queue.enqueue_event(new_agent_text_message(content_block["text"]))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel."""
        raise ServerError(error=UnsupportedOperationError())
