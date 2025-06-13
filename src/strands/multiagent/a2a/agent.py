"""A2A-compatible wrapper for Strands Agent.

This module provides the A2AAgent class, which adapts a Strands Agent to the A2A protocol,
allowing it to be used in A2A-compatible systems.
"""

import logging
from typing import Any, Literal

import uvicorn
from a2a.server.apps import A2AFastAPIApplication, A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from fastapi import FastAPI
from starlette.applications import Starlette

from ...agent.agent import Agent as SAAgent
from .executor import StrandsA2AExecutor

log = logging.getLogger(__name__)


class A2AAgent:
    """A2A-compatible wrapper for Strands Agent.

    This class adapts a Strands Agent to the A2A protocol, allowing it to be used
    in A2A-compatible systems.
    """

    def __init__(
        self,
        agent: SAAgent,
        *,
        # AgentCard
        name: str,
        description: str,
        host: str = "localhost",
        port: int = 9000,
        version: str = "0.0.1",
    ):
        """Initialize an A2A-compatible agent from a Strands agent.

        TODO: add args

        """
        self.name = name
        self.description = description
        self.host = host
        self.port = port
        self.http_url = f"http://{self.host}:{self.port}/"
        self.version = version
        self.strands_agent = agent
        self.capabilities = AgentCapabilities()
        self.request_handler = DefaultRequestHandler(
            agent_executor=StrandsA2AExecutor(self.strands_agent),
            task_store=InMemoryTaskStore(),
        )

    @property
    def public_agent_card(self) -> AgentCard:
        """AgentCard."""
        return AgentCard(
            name=self.name,
            description=self.description,
            url=self.http_url,
            version=self.version,
            skills=self.agent_skills,
            defaultInputModes=["text"],
            defaultOutputModes=["text"],
            capabilities=self.capabilities,
        )

    @property
    def agent_skills(self) -> list[AgentSkill]:
        """AgentSkills."""
        return []

    def to_starlette_app(self) -> Starlette:
        """Startlette app."""
        starlette_app = A2AStarletteApplication(agent_card=self.public_agent_card, http_handler=self.request_handler)
        return starlette_app.build()

    def to_fastapi_app(self) -> FastAPI:
        """FastAPI app."""
        fastapi_app = A2AFastAPIApplication(agent_card=self.public_agent_card, http_handler=self.request_handler)
        return fastapi_app.build()

    def serve(self, app_type: Literal["fastapi", "starlette"] = "starlette", **kwargs: Any) -> None:
        """Start the A2A server with the specified application type.

        Args:
            app_type: The type of application to serve, either "fastapi" or "starlette".
            **kwargs: Additional keyword arguments to pass to uvicorn.run.
        """
        try:
            log.info("Starting Strands agent A2A server...")
            if app_type == "fastapi":
                uvicorn.run(self.to_fastapi_app(), host=self.host, port=self.port, **kwargs)
            else:
                uvicorn.run(self.to_starlette_app(), host=self.host, port=self.port, **kwargs)
        except KeyboardInterrupt:
            log.warning("Server shutdown requested (KeyboardInterrupt).")
        finally:
            log.info("Strands agent A2A server has shutdown.")
