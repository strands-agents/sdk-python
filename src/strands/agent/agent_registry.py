"""Agent registry.

This module provides a central registry for all agents, supporting registration, discovery, and
session-scoped management.
Mirrors the ToolRegistry pattern, but for agents. Supports mapping subsets of tools to agents, and
integrates with agent state and event loop patterns.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..tools.registry import ToolRegistry
from .agent import Agent

logger = logging.getLogger(__name__)

class AgentRegistry:
    """Central registry for all agents.

    Manages agent registration, discovery, and session-scoped management.
    Supports mapping subsets of tools to agents, and integrates with agent state and event loop patterns.
    Designed to be lightweight, optional, and extensible.
    """
    def __init__(self) -> None:
        """Initialize the agent registry."""
        self.registry: Dict[str, Agent] = {}
        self.session_agents: Dict[str, List[str]] = {}  # session_id -> list of agent names
        self.agent_tool_subsets: Dict[str, ToolRegistry] = {}  # agent_name -> ToolRegistry subset

    def register_agent(
        self,
        agent: Agent,
        name: Optional[str] = None,
        session_id: Optional[str] = None,
        tool_subset: Optional[List[Any]] = None,
    ) -> str:
        """Register an agent, optionally for a session and with a subset of tools.

        Args:
            agent: The Agent instance to register.
            name: Optional name for the agent (defaults to agent.name or generated).
            session_id: Optional session/event loop ID to scope the agent.
            tool_subset: Optional list of tools to restrict this agent to (mirrors ToolRegistry.process_tools).

        Returns:
            The agent's registry name.
        """
        agent_name = name or getattr(agent, 'name', None) or f"agent_{uuid4().hex[:8]}"
        self.registry[agent_name] = agent
        if session_id:
            self.session_agents.setdefault(session_id, []).append(agent_name)
        if tool_subset is not None:
            tool_registry = ToolRegistry()
            tool_registry.process_tools(tool_subset)
            self.agent_tool_subsets[agent_name] = tool_registry
        logger.debug(
            "Registered agent '%s' (session: %s, tool subset: %s)",
            agent_name,
            session_id,
            tool_subset is not None,
        )
        return agent_name

    def get_agent(self, name: str, session_id: Optional[str] = None) -> Optional[Agent]:
        """Retrieve an agent by name, optionally scoped to a session."""
        if session_id:
            if name in self.session_agents.get(session_id, []):
                return self.registry.get(name)
            return None
        return self.registry.get(name)

    def unregister_agent(self, name: str, session_id: Optional[str] = None) -> None:
        """Unregister an agent by name, optionally from a session."""
        if session_id:
            if name in self.session_agents.get(session_id, []):
                self.session_agents[session_id].remove(name)
                if not self.session_agents[session_id]:
                    del self.session_agents[session_id]
        if name in self.registry:
            del self.registry[name]
        if name in self.agent_tool_subsets:
            del self.agent_tool_subsets[name]
        logger.debug(
            "Unregistered agent '%s' (session: %s)",
            name,
            session_id,
        )

    def list_agents(self, session_id: Optional[str] = None) -> List[str]:
        """List all registered agent names, optionally filtered by session."""
        if session_id:
            return list(self.session_agents.get(session_id, []))
        return list(self.registry.keys())

    def get_tool_registry_for_agent(self, name: str) -> Optional[ToolRegistry]:
        """Get the ToolRegistry subset for a specific agent, if set."""
        return self.agent_tool_subsets.get(name)

    def clear(self) -> None:
        """Clear all registered agents and session mappings."""
        self.registry.clear()
        self.session_agents.clear()
        self.agent_tool_subsets.clear()
        logger.debug("Cleared all agents and session mappings from AgentRegistry.")

# Example usage (optional):
# agent_registry = AgentRegistry()
# agent_name = agent_registry.register_agent(agent, tool_subset=[...])
# agent = agent_registry.get_agent(agent_name)
