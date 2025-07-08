"""Agent registry module: provides a global registry for agent classes."""
import logging
import os
import sys
import importlib.util
import inspect
from typing import Any, Dict, Type, Optional

logger = logging.getLogger(__name__)

class AgentRegistry:
    """Global registry for agent classes. Allows registration, lookup, and (future) discovery of agents by name."""
    def __init__(self):
        """Initialize the agent registry."""
        self._agents: Dict[str, Type[Any]] = {}

    def register(self, name: str, agent_cls: Type[Any]) -> None:
        """Register an agent class with a unique name."""
        if name in self._agents:
            raise ValueError(f"Agent '{name}' already registered")
        self._agents[name] = agent_cls
        logger.debug("Registered agent: %s", name)

    def get(self, name: str) -> Optional[Type[Any]]:
        """Retrieve an agent class by name."""
        return self._agents.get(name)

    def all_agents(self) -> Dict[str, Type[Any]]:
        """Return all registered agents."""
        return dict(self._agents)

    def discover(self, directory: str) -> None:
        """Discover and register agent classes from all Python files in the given directory.

        Args:
            directory: Path to the directory to scan for agent classes.
        """
        from .agent import Agent  # Local import to avoid circular import
        logger.info("Discovering agents in directory: %s", directory)
        if not os.path.isdir(directory):
            logger.warning("Directory does not exist: %s", directory)
            return
        sys.path.insert(0, directory)
        for filename in os.listdir(directory):
            if filename.startswith("_") or not filename.endswith(".py"):
                continue
            module_name = filename[:-3]
            file_path = os.path.join(directory, filename)
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, Agent) and obj is not Agent:
                            agent_name = getattr(obj, "agent_name", name)
                            if agent_name not in self._agents:
                                self.register(agent_name, obj)
                                logger.info("Discovered and registered agent: %s (from %s)", agent_name, filename)
            except Exception as e:
                logger.warning("Failed to import agent from %s: %s", filename, e)
        sys.path.pop(0)
