"""A2A Module."""

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

from ...agent.agent import Agent as SAAAgent


class HelloWorldAgent:
    """Hello World Agent."""

    async def invoke(self) -> str:
        """Hello World Agent."""
        return "Hello World"


class StrandsA2AExecutor(AgentExecutor):
    """Test AgentProxy Implementation."""

    def __init__(self, agent: SAAAgent):
        """Hello World Agent."""
        self.agent = agent

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Hello World Agent."""
        result = self.agent("tell me something about love")
        await event_queue.enqueue_event(new_agent_text_message(result))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Hello World Agent."""
        raise Exception("cancel not supported")
