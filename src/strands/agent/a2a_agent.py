"""A2A Agent client for Strands Agents.

This module provides the A2AAgent class, which acts as a client wrapper for remote A2A agents,
allowing them to be used standalone or as part of multi-agent patterns.

A2AAgent can be used to get the Agent Card and interact with the agent.
"""

import logging
from typing import Any, AsyncIterator

import httpx
from a2a.client import A2ACardResolver, Client, ClientConfig, ClientFactory
from a2a.types import AgentCard, Message, TaskArtifactUpdateEvent, TaskState, TaskStatusUpdateEvent

from .._async import run_async
from ..multiagent.a2a.converters import convert_input_to_message, convert_response_to_agent_result
from ..types._events import AgentResultEvent
from ..types.a2a import A2AResponse, A2AStreamEvent
from ..types.agent import AgentInput
from .agent_result import AgentResult

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 300


class A2AAgent:
    """Client wrapper for remote A2A agents."""

    def __init__(
        self,
        endpoint: str,
        *,
        name: str | None = None,
        description: str = "",
        timeout: int = DEFAULT_TIMEOUT,
        a2a_client_factory: ClientFactory | None = None,
    ):
        """Initialize A2A agent.

        Args:
            endpoint: The base URL of the remote A2A agent.
            name: Agent name. If not provided, will be populated from agent card.
            description: Agent description. If empty, will be populated from agent card.
            timeout: Timeout for HTTP operations in seconds (defaults to 300).
            a2a_client_factory: Optional pre-configured A2A ClientFactory. If provided,
                it will be used to create the A2A client after discovering the agent card.
        """
        self.endpoint = endpoint
        self.name = name
        self.description = description
        self.timeout = timeout
        self._httpx_client: httpx.AsyncClient | None = None
        self._owns_client = a2a_client_factory is None
        self._agent_card: AgentCard | None = None
        self._a2a_client: Client | None = None
        self._a2a_client_factory: ClientFactory | None = a2a_client_factory

    def _get_httpx_client(self) -> httpx.AsyncClient:
        """Get or create the httpx client for this agent.

        Returns:
            Configured httpx.AsyncClient instance.
        """
        if self._httpx_client is None:
            self._httpx_client = httpx.AsyncClient(timeout=self.timeout)
        return self._httpx_client

    async def _get_agent_card(self) -> AgentCard:
        """Discover and cache the agent card from the remote endpoint.

        Returns:
            The discovered AgentCard.
        """
        if self._agent_card is not None:
            return self._agent_card

        httpx_client = self._get_httpx_client()
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.endpoint)
        self._agent_card = await resolver.get_agent_card()

        # Populate name from card if not set
        if self.name is None and self._agent_card.name:
            self.name = self._agent_card.name

        # Populate description from card if not set
        if not self.description and self._agent_card.description:
            self.description = self._agent_card.description

        logger.debug("agent=<%s>, endpoint=<%s> | discovered agent card", self.name, self.endpoint)
        return self._agent_card

    def _create_default_factory(self) -> ClientFactory:
        """Create default A2A client factory with non-streaming config.

        Returns:
            Configured ClientFactory instance.
        """
        httpx_client = self._get_httpx_client()
        config = ClientConfig(httpx_client=httpx_client, streaming=False)
        return ClientFactory(config)

    async def _get_a2a_client(self) -> Client:
        """Get or create the A2A client for this agent.

        Returns:
            Configured A2A client instance.
        """
        if self._a2a_client is None:
            agent_card = await self._get_agent_card()
            factory = self._a2a_client_factory or self._create_default_factory()
            self._a2a_client = factory.create(agent_card)
        return self._a2a_client

    async def _send_message(self, prompt: AgentInput) -> AsyncIterator[A2AResponse]:
        """Send message to A2A agent.

        Args:
            prompt: Input to send to the agent.

        Returns:
            Async iterator of A2A events.

        Raises:
            ValueError: If prompt is None.
        """
        if prompt is None:
            raise ValueError("prompt is required for A2AAgent")

        client = await self._get_a2a_client()
        message = convert_input_to_message(prompt)

        logger.debug("agent=<%s>, endpoint=<%s> | sending message", self.name, self.endpoint)
        return client.send_message(message)

    def _is_complete_event(self, event: A2AResponse) -> bool:
        """Check if an A2A event represents a complete response.

        Args:
            event: A2A event.

        Returns:
            True if the event represents a complete response.
        """
        # Direct Message is always complete
        if isinstance(event, Message):
            return True

        # Handle tuple responses (Task, UpdateEvent | None)
        if isinstance(event, tuple) and len(event) == 2:
            task, update_event = event

            # Initial task response (no update event)
            if update_event is None:
                return True

            # Artifact update with last_chunk flag
            if isinstance(update_event, TaskArtifactUpdateEvent):
                if hasattr(update_event, "last_chunk") and update_event.last_chunk is not None:
                    return update_event.last_chunk
                return False

            # Status update with completed state
            if isinstance(update_event, TaskStatusUpdateEvent):
                if update_event.status and hasattr(update_event.status, "state"):
                    return update_event.status.state == TaskState.completed

        return False

    async def invoke_async(
        self,
        prompt: AgentInput = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Asynchronously invoke the remote A2A agent.

        Delegates to stream_async and returns the final result.

        Args:
            prompt: Input to the agent (string, message list, or content blocks).
            **kwargs: Additional arguments (ignored).

        Returns:
            AgentResult containing the agent's response.

        Raises:
            ValueError: If prompt is None.
            RuntimeError: If no response received from agent.
        """
        result: AgentResult | None = None
        async for event in self.stream_async(prompt, **kwargs):
            if "result" in event:
                result = event["result"]

        if result is None:
            raise RuntimeError("No response received from A2A agent")

        return result

    def __call__(
        self,
        prompt: AgentInput = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Synchronously invoke the remote A2A agent.

        Args:
            prompt: Input to the agent (string, message list, or content blocks).
            **kwargs: Additional arguments (ignored).

        Returns:
            AgentResult containing the agent's response.

        Raises:
            ValueError: If prompt is None.
            RuntimeError: If no response received from agent.
        """
        return run_async(lambda: self.invoke_async(prompt, **kwargs))

    async def stream_async(
        self,
        prompt: AgentInput = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream agent execution asynchronously.

        Args:
            prompt: Input to the agent (string, message list, or content blocks).
            **kwargs: Additional arguments (ignored).

        Yields:
            A2A events and a final AgentResult event.

        Raises:
            ValueError: If prompt is None.
        """
        last_event = None
        last_complete_event = None

        async for event in await self._send_message(prompt):
            last_event = event
            if self._is_complete_event(event):
                last_complete_event = event
            yield A2AStreamEvent(event)

        # Use the last complete event if available, otherwise fall back to last event
        final_event = last_complete_event if last_complete_event is not None else last_event

        if final_event is not None:
            result = convert_response_to_agent_result(final_event)
            yield AgentResultEvent(result)

    def __del__(self) -> None:
        """Best-effort cleanup on garbage collection."""
        if self._owns_client and self._httpx_client is not None:
            try:
                client = self._httpx_client
                run_async(lambda: client.aclose())
            except Exception:
                pass  # Best effort cleanup, ignore errors in __del__
