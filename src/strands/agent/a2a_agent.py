"""A2A Agent client for Strands Agents.

This module provides the A2AAgent class, which acts as a client wrapper for remote A2A agents,
allowing them to be used standalone or as part of multi-agent patterns.

A2AAgent can be used to get the Agent Card and interact with the agent.
"""

import logging
import warnings
from collections.abc import AsyncIterator
from typing import Any

import httpx
from a2a.client import A2ACardResolver, Client, ClientConfig, ClientFactory
from a2a.types import AgentCard, Message, TaskArtifactUpdateEvent, TaskState, TaskStatusUpdateEvent

from .._async import run_async
from ..multiagent.a2a._converters import convert_input_to_message, convert_response_to_agent_result
from ..types._events import AgentResultEvent
from ..types.a2a import A2AResponse, A2AStreamEvent
from ..types.agent import AgentInput
from .agent_result import AgentResult
from .base import AgentBase

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 300


class A2AAgent(AgentBase):
    """Client wrapper for remote A2A agents."""

    def __init__(
        self,
        endpoint: str,
        *,
        name: str | None = None,
        description: str | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
        client_config: ClientConfig | None = None,
        a2a_client_factory: ClientFactory | None = None,
    ):
        """Initialize A2A agent.

        Args:
            endpoint: The base URL of the remote A2A agent.
            name: Agent name. If not provided, will be populated from agent card.
            description: Agent description. If not provided, will be populated from agent card.
            timeout: Timeout for HTTP operations in seconds (defaults to 300).
            client_config: Optional A2A ClientConfig for authentication and transport settings.
                When provided, the config (including any authenticated httpx client) is used
                for both agent card discovery and A2A message sending. This is the recommended
                way to configure authentication (e.g. SigV4, OAuth bearer tokens).
            a2a_client_factory: Deprecated. Use ``client_config`` instead. Optional pre-configured
                A2A ClientFactory. When provided, ``factory.create()`` is used for client creation,
                preserving any configured interceptors, consumers, and custom transports. For card
                resolution, ``client_config`` is preferred if provided; otherwise the factory's
                internal config is used as a fallback. The caller is responsible for managing the
                lifecycle of any httpx client configured in the factory.
        """
        if a2a_client_factory is not None:
            warnings.warn(
                "a2a_client_factory is deprecated, use client_config instead. "
                "a2a_client_factory will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.endpoint = endpoint
        self.name = name
        self.description = description
        self.timeout = timeout
        self._client_config = client_config
        self._a2a_client_factory: ClientFactory | None = a2a_client_factory
        self._agent_card: AgentCard | None = None
        self._a2a_client: Client | None = None

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

    async def invoke_async(
        self,
        prompt: AgentInput = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Asynchronously invoke the remote A2A agent.

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

    async def stream_async(
        self,
        prompt: AgentInput = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream remote agent execution asynchronously.

        This method provides an asynchronous interface for streaming A2A protocol events.
        Unlike Agent.stream_async() which yields text deltas and tool events, this method
        yields raw A2A protocol events wrapped in A2AStreamEvent dictionaries.

        Args:
            prompt: Input to the agent (string, message list, or content blocks).
            **kwargs: Additional arguments (ignored).

        Yields:
            An async iterator that yields events. Each event is a dictionary:
                - A2AStreamEvent: {"type": "a2a_stream", "event": <A2A object>}
                  where the A2A object can be a Message, or a tuple of
                  (Task, TaskStatusUpdateEvent) or (Task, TaskArtifactUpdateEvent).
                - AgentResultEvent: {"result": AgentResult} - always emitted last.

        Raises:
            ValueError: If prompt is None.

        Example:
            ```python
            async for event in a2a_agent.stream_async("Hello"):
                if event.get("type") == "a2a_stream":
                    print(f"A2A event: {event['event']}")
                elif "result" in event:
                    print(f"Final result: {event['result'].message}")
            ```
        """
        last_event = None
        last_complete_event = None

        async for event in self._send_message(prompt):
            last_event = event
            if self._is_complete_event(event):
                last_complete_event = event
            yield A2AStreamEvent(event)

        # Use the last complete event if available, otherwise fall back to last event
        final_event = last_complete_event or last_event

        if final_event is not None:
            result = convert_response_to_agent_result(final_event)
            yield AgentResultEvent(result)

    async def get_agent_card(self) -> AgentCard:
        """Fetch and return the remote agent's card.

        This method eagerly fetches the agent card from the remote endpoint,
        populating name and description if not already set. The card is cached
        after the first fetch.

        When ``client_config`` is provided, its httpx client is used for card resolution
        so that any pre-configured authentication (SigV4, OAuth bearer tokens, etc.) is applied.

        Returns:
            The remote agent's AgentCard containing name, description, capabilities, skills, etc.
        """
        if self._agent_card is not None:
            return self._agent_card

        config = self._resolve_client_config()
        if config is not None and config.httpx_client is not None:
            resolver = A2ACardResolver(httpx_client=config.httpx_client, base_url=self.endpoint)
            self._agent_card = await resolver.get_agent_card()
        else:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resolver = A2ACardResolver(httpx_client=client, base_url=self.endpoint)
                self._agent_card = await resolver.get_agent_card()

        # Populate name from card if not set
        if self.name is None and self._agent_card.name:
            self.name = self._agent_card.name

        # Populate description from card if not set
        if self.description is None and self._agent_card.description:
            self.description = self._agent_card.description

        logger.debug("agent=<%s>, endpoint=<%s> | discovered agent card", self.name, self.endpoint)
        return self._agent_card

    def _resolve_client_config(self) -> ClientConfig | None:
        """Resolve the effective client config for card resolution and client creation.

        Precedence:
        1. Explicit ``client_config`` parameter (always preferred)
        2. Factory's internal config (fallback for deprecated factory path)
        3. None (use defaults)

        Returns:
            Resolved ClientConfig, or None if no config is available.
        """
        if self._client_config is not None:
            return self._client_config

        if self._a2a_client_factory is not None:
            config = getattr(self._a2a_client_factory, "_config", None)
            if config is None:
                logger.warning(
                    "endpoint=<%s> | could not access factory client config, "
                    "falling back to default config for card resolution",
                    self.endpoint,
                )
            return config

        return None

    async def _get_or_create_client(self) -> Client:
        """Get or create an A2A client for communicating with the remote agent.

        When a deprecated factory is provided, ``factory.create()`` is used for client creation
        to preserve interceptors, consumers, and custom transports. The resulting client is cached.

        When ``client_config`` is provided without a factory, ``ClientFactory.connect()`` is used
        with the config for both card resolution and client creation. The client is cached.

        When neither is provided, a transient client is created per call to avoid long-lived
        httpx connections which can cause connection breakdown and deadlocks on Windows.

        Returns:
            Configured A2A Client instance.
        """
        # Deprecated factory path: use factory.create() to preserve interceptors/consumers/transports
        if self._a2a_client_factory is not None:
            if self._a2a_client is not None:
                return self._a2a_client

            agent_card = await self.get_agent_card()
            self._a2a_client = self._a2a_client_factory.create(agent_card)
            return self._a2a_client

        # client_config path: cache the client
        if self._client_config is not None:
            if self._a2a_client is not None:
                return self._a2a_client

            self._a2a_client = await ClientFactory.connect(self.endpoint, client_config=self._client_config)
            return self._a2a_client

        # No factory or config: create transient client per call
        return await ClientFactory.connect(self.endpoint)

    async def _send_message(self, prompt: AgentInput) -> AsyncIterator[A2AResponse]:
        """Send message to A2A agent.

        Args:
            prompt: Input to send to the agent.

        Yields:
            A2A response events.

        Raises:
            ValueError: If prompt is None.
        """
        if prompt is None:
            raise ValueError("prompt is required for A2AAgent")

        message = convert_input_to_message(prompt)
        logger.debug("agent=<%s>, endpoint=<%s> | sending message", self.name, self.endpoint)

        client = await self._get_or_create_client()
        async for event in client.send_message(message):
            yield event

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
