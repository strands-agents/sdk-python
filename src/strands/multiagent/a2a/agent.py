"""A2A-compatible wrapper for Strands Agent.

This module provides the A2AAgent class, which adapts a Strands Agent to the A2A protocol,
allowing it to be used in A2A-compatible systems.
"""

import logging

import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCard,
    AgentSkill,
)

from ...agent.agent import Agent as SAAgent
from .executor import StrandsA2AExecutor

logger = logging.getLogger(__name__)


class A2AAgent:
    """A2A-compatible wrapper for Strands Agent.

    This class adapts a Strands Agent to the A2A protocol, allowing it to be used
    in A2A-compatible systems.
    """

    def __init__(
        self,
        agent: SAAgent | None,
        agent_card: AgentCard | None,
        skills: list[AgentSkill] | None,
        version: str = "1.0.0",
        host: str = "localhost",
        port: int = 9000,
    ):
        """Initialize an A2A-compatible agent from a Strands agent.

        TODO: add args

        """
        self.host = host
        self.port = port
        self.agent = agent
        self.agent_card = agent_card
        self.skills = skills
        self.executor = StrandsA2AExecutor(self.agent)
        self.httpx_client = httpx.AsyncClient()
        self.request_handler = DefaultRequestHandler(
            agent_executor=self.executor,
            task_store=InMemoryTaskStore(),
            # push_notifier=a2a_tasks.InMemoryPushNotifier(self.httpx_client),
        )

    def to_starlette(self):
        """TODO."""
        starlette_server = A2AStarletteApplication(agent_card=self.agent_card, http_handler=self.request_handler)
        return starlette_server.build()

    def serve(self):
        """TODO."""
        uvicorn.run(self.to_starlette(), host=self.host, port=self.port)
