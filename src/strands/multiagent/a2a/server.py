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

logger = logging.getLogger(__name__)


class A2AServer:
    """A2A-compatible wrapper for Strands Agent."""

    def __init__(
        self,
        agent: SAAgent,
        *,
        # AgentCard
        host: str = "0.0.0.0",
        port: int = 9000,
        version: str = "0.0.1",
        skills: list[AgentSkill] | None = None,
    ):
        """Initialize an A2A-compatible agent from a Strands agent.

        Args:
            agent: The Strands Agent to wrap with A2A compatibility.
            name: The name of the agent, used in the AgentCard.
            description: A description of the agent's capabilities, used in the AgentCard.
            host: The hostname or IP address to bind the A2A server to. Defaults to "0.0.0.0".
            port: The port to bind the A2A server to. Defaults to 9000.
            version: The version of the agent. Defaults to "0.0.1".
            skills: The list of capabilities or functions the agent can perform.
        """
        self.host = host
        self.port = port
        self.http_url = f"http://{self.host}:{self.port}/"
        self.version = version
        self.strands_agent = agent
        self.name = self.strands_agent.name
        self.description = self.strands_agent.description
        # TODO: enable configurable capabilities and request handler
        self.capabilities = AgentCapabilities()
        self.request_handler = DefaultRequestHandler(
            agent_executor=StrandsA2AExecutor(self.strands_agent),
            task_store=InMemoryTaskStore(),
        )
        self._agent_skills = skills
        logger.info("Strands' integration with A2A is experimental. Be aware of frequent breaking changes.")

    @property
    def public_agent_card(self) -> AgentCard:
        """Get the public AgentCard for this agent.

        The AgentCard contains metadata about the agent, including its name,
        description, URL, version, skills, and capabilities. This information
        is used by other agents and systems to discover and interact with this agent.

        Returns:
            AgentCard: The public agent card containing metadata about this agent.

        Raises:
            ValueError: If name or description is None or empty.
        """
        if not self.name:
            raise ValueError("A2A agent name cannot be None or empty")
        if not self.description:
            raise ValueError("A2A agent description cannot be None or empty")

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

    def _get_skills_from_tools(self) -> list[AgentSkill]:
        """Get the list of skills from Strands agent tools.

        Skills represent specific capabilities that the agent can perform.
        Strands agent tools are adapted to A2A skills.

        Returns:
            list[AgentSkill]: A list of skills this agent provides.
        """
        return [
            AgentSkill(name=config["name"], id=config["name"], description=config["description"], tags=[])
            for config in self.strands_agent.tool_registry.get_all_tools_config().values()
        ]

    @property
    def agent_skills(self) -> list[AgentSkill]:
        """Get the list of skills this agent provides."""
        return self._agent_skills if self._agent_skills is not None else self._get_skills_from_tools()

    @agent_skills.setter
    def agent_skills(self, skills: list[AgentSkill]) -> None:
        """Set the list of skills this agent provides.

        Args:
            skills: A list of AgentSkill objects to set for this agent.
        """
        self._agent_skills = skills

    def to_starlette_app(self) -> Starlette:
        """Create a Starlette application for serving this agent via HTTP.

        This method creates a Starlette application that can be used to serve
        the agent via HTTP using the A2A protocol.

        Returns:
            Starlette: A Starlette application configured to serve this agent.
        """
        return A2AStarletteApplication(agent_card=self.public_agent_card, http_handler=self.request_handler).build()

    def to_fastapi_app(self) -> FastAPI:
        """Create a FastAPI application for serving this agent via HTTP.

        This method creates a FastAPI application that can be used to serve
        the agent via HTTP using the A2A protocol.

        Returns:
            FastAPI: A FastAPI application configured to serve this agent.
        """
        return A2AFastAPIApplication(agent_card=self.public_agent_card, http_handler=self.request_handler).build()

    def serve(self, app_type: Literal["fastapi", "starlette"] = "starlette", **kwargs: Any) -> None:
        """Start the A2A server with the specified application type.

        This method starts an HTTP server that exposes the agent via the A2A protocol.
        The server can be implemented using either FastAPI or Starlette, depending on
        the specified app_type.

        Args:
            app_type: The type of application to serve, either "fastapi" or "starlette".
                Defaults to "starlette".
            **kwargs: Additional keyword arguments to pass to uvicorn.run.
        """
        try:
            logger.info("Starting Strands A2A server...")
            if app_type == "fastapi":
                uvicorn.run(self.to_fastapi_app(), host=self.host, port=self.port, **kwargs)
            else:
                uvicorn.run(self.to_starlette_app(), host=self.host, port=self.port, **kwargs)
        except KeyboardInterrupt:
            logger.warning("Strands A2A server shutdown requested (KeyboardInterrupt).")
        except Exception:
            logger.exception("Strands A2A server encountered exception.")
        finally:
            logger.info("Strands A2A server has shutdown.")
