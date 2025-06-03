"""A2A Protocol Client implementation.

This module provides client functionality for interacting with remote A2A agents.
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, Optional, List
from urllib.parse import urljoin

import httpx
from a2a.types import AgentCard, Message

logger = logging.getLogger(__name__)


class A2AProtocolClient:
    """Client for interacting with A2A protocol servers."""
    
    def __init__(self, base_url: str, auth_token: Optional[str] = None):
        """Initialize A2A client.
        
        Args:
            base_url: Base URL of the remote A2A agent
            auth_token: Optional bearer token for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self._agent_card: Optional[AgentCard] = None
        self._session: Optional[httpx.AsyncClient] = None
        
    @property
    def agent_card(self) -> Optional[AgentCard]:
        """Get cached agent card."""
        return self._agent_card
        
    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create HTTP session."""
        if self._session is None:
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            self._session = httpx.AsyncClient(headers=headers)
        return self._session
        
    async def fetch_agent_card(self, force_refresh: bool = False) -> AgentCard:
        """Fetch the agent card from a remote A2A server.
        
        Args:
            force_refresh: Force fetching even if cached
            
        Returns:
            The agent's card with capabilities and skills
        """
        if self._agent_card and not force_refresh:
            return self._agent_card
            
        try:
            session = await self._get_session()
            url = urljoin(self.base_url, "/.well-known/agent.json")
            response = await session.get(url)
            response.raise_for_status()
            
            card_data = response.json()
            self._agent_card = AgentCard(**card_data)
            return self._agent_card
            
        except httpx.HTTPError as e:
            raise ConnectionError(f"Failed to fetch agent card from {self.base_url}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error parsing agent card: {e}")
    
    async def send_task(
        self, 
        message: str, 
        skill_id: Optional[str] = None,
        session_id: Optional[str] = None,
        timeout: Optional[float] = 30.0
    ) -> Dict[str, Any]:
        """Send a task to the remote agent.
        
        Args:
            message: The message/prompt to send
            skill_id: Optional specific skill to use
            session_id: Optional session ID for conversation continuity
            timeout: Request timeout in seconds
            
        Returns:
            The agent's response
        """
        task_id = str(uuid.uuid4())
        session_id = session_id or str(uuid.uuid4())
        
        # Build JSON-RPC 2.0 request
        request_data = {
            "jsonrpc": "2.0",
            "method": "task.execute",
            "id": task_id,
            "params": {
                "message": message,
                "sessionId": session_id
            }
        }
        
        # Add skill hint if specified
        if skill_id:
            request_data["params"]["skillId"] = skill_id
            
        try:
            session = await self._get_session()
            response = await session.post(
                self.base_url,
                json=request_data,
                timeout=timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Check for JSON-RPC error
            if "error" in result:
                error = result["error"]
                raise RuntimeError(f"A2A Error {error.get('code', 'unknown')}: {error.get('message', 'Unknown error')}")
                
            return result
            
        except httpx.TimeoutException:
            raise TimeoutError(f"Request to {self.base_url} timed out after {timeout}s")
        except httpx.HTTPError as e:
            raise ConnectionError(f"HTTP error communicating with {self.base_url}: {e}")
            
    async def send_task_and_wait(
        self,
        message: str,
        skill_id: Optional[str] = None,
        session_id: Optional[str] = None,
        timeout: Optional[float] = 30.0,
        poll_interval: float = 0.5
    ) -> str:
        """Send a task and wait for completion, returning just the text result.
        
        Args:
            message: The message/prompt to send
            skill_id: Optional specific skill to use
            session_id: Optional session ID for conversation continuity
            timeout: Total timeout for task completion
            poll_interval: Interval between status checks
            
        Returns:
            The text content of the agent's response
        """
        response = await self.send_task(message, skill_id, session_id, timeout)
        
        # Extract text content from response
        result = response.get("result", {})
        content = result.get("content", "")
        
        # Handle different content formats
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            # Try to extract text from structured content
            return content.get("text", str(content))
        else:
            return str(content)
            
    async def close(self):
        """Close the client session."""
        if self._session:
            await self._session.aclose()
            self._session = None
            
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Synchronous convenience functions
def fetch_agent_card_sync(agent_url: str, auth_token: Optional[str] = None) -> AgentCard:
    """Synchronously fetch an agent card.
    
    Args:
        agent_url: URL of the A2A agent
        auth_token: Optional authentication token
        
    Returns:
        The agent's card
    """
    async def _fetch():
        async with A2AProtocolClient(agent_url, auth_token) as client:
            return await client.fetch_agent_card()
    
    return asyncio.run(_fetch())


def send_a2a_request_sync(
    agent_url: str, 
    message: str,
    auth_token: Optional[str] = None,
    skill_id: Optional[str] = None
) -> str:
    """Synchronously send a request to an A2A agent.
    
    Args:
        agent_url: URL of the A2A agent
        message: Message to send
        auth_token: Optional authentication token
        skill_id: Optional specific skill to use
        
    Returns:
        The agent's text response
    """
    async def _send():
        async with A2AProtocolClient(agent_url, auth_token) as client:
            return await client.send_task_and_wait(message, skill_id)
    
    return asyncio.run(_send()) 