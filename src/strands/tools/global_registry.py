"""Global tool registry module: provides a global registry for all tools in the system."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class GlobalToolRegistry:
    """Global registry for all tools in the system. Allows registration, lookup, and listing of tools by name."""

    def __init__(self):
        """Initialize the global tool registry."""
        self._tools: Dict[str, Any] = {}

    def register(self, name: str, tool: Any) -> None:
        """Register a tool with a unique name."""
        if name in self._tools:
            raise ValueError(f"Tool '{name}' already registered in global registry")
        self._tools[name] = tool
        logger.debug("Registered tool: %s", name)

    def get(self, name: str) -> Optional[Any]:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def all_tools(self) -> Dict[str, Any]:
        """Return all registered tools."""
        return dict(self._tools)
