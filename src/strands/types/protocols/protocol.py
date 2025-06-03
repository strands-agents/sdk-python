"""Protocol server abstraction for agent networking.

This module provides the abstract base class for protocol server implementations,
following the same pattern as the Model abstraction.
"""

import abc
import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...agent.agent import Agent

logger = logging.getLogger(__name__)


class ProtocolServer(abc.ABC):
    """Abstract base class for agent protocol server implementations.
    
    This class defines the interface for all protocol server implementations in the 
    Strands Agents SDK. It provides a standardized way to configure, start, and manage
    different protocol servers (A2A, MCP, GraphQL, gRPC, etc.).
    """
    
    @abc.abstractmethod
    def update_config(self, **server_config: Any) -> None:
        """Update the server configuration with the provided arguments.
        
        Args:
            **server_config: Configuration overrides.
        """
        pass
    
    @abc.abstractmethod 
    def get_config(self) -> Any:
        """Return the server configuration.
        
        Returns:
            The server's configuration.
        """
        pass
    
    @abc.abstractmethod
    def start(self, agent: "Agent") -> None:
        """Start the protocol server for the given agent.
        
        Args:
            agent: The agent to expose via this protocol.
        """
        pass
    
    @abc.abstractmethod
    def stop(self) -> None:
        """Stop the protocol server."""
        pass
    
    @abc.abstractmethod
    def get_endpoint(self) -> str:
        """Get the server's endpoint URL.
        
        Returns:
            The URL where the server is accessible.
        """
        pass
    
    @property
    @abc.abstractmethod
    def protocol_name(self) -> str:
        """The name of the protocol (e.g., 'a2a', 'mcp', 'graphql')."""
        pass
    
    @property
    def is_running(self) -> bool:
        """Whether the server is currently running.
        
        Returns:
            True if the server is running, False otherwise.
        """
        return False
    
    def __enter__(self) -> "ProtocolServer":
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ensure server is stopped."""
        if self.is_running:
            self.stop() 