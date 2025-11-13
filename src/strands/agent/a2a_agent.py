"""A2A Agent client for Strands Agents.

This module provides the A2AAgent class, which acts as a client wrapper for remote A2A agents,
allowing them to be used in graphs, swarms, and other multi-agent patterns.
"""

import logging
from typing import Any, AsyncIterator

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import AgentCard

from .._async import run_async
from ..multiagent.a2a.converters import convert_input_to_message, convert_response_to_agent_result
from ..types.agent import AgentInput
from .agent_result import AgentResult

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 300


class A2AAgent:
    """Client wrapper for remote A2A agents.

    Implements the AgentBase protocol to enable remote A2A agents to be used
    in graphs, swarms, and other multi-agent patterns.
    """

    def __init__(
        self,
        endpoint: str,
        timeout: int = DEFAULT_TIMEOUT,
        httpx_client_args: dict[str, Any] | None = None,
    ):
        """Initialize A2A agent client.

        Args:
            endpoint: The base URL of the remote A2A agent
            timeout: Timeout for HTTP operations in seconds (defaults to 300)
            httpx_client_args: Optional dictionary of arguments to pass to httpx.AsyncClient
                constructor. Allows custom auth, headers, proxies, etc.
                Example: {"headers": {"Authorization": "Bearer token"}}
        """
        self.endpoint = endpoint
        self.timeout = timeout
        self._httpx_client_args: dict[str, Any] = httpx_client_args or {}

        if "timeout" not in self._httpx_client_args:
            self._httpx_client_args["timeout"] = self.timeout

        self._agent_card: AgentCard | None = None

    def _get_httpx_client(self) -> httpx.AsyncClient:
        """Get a fresh httpx client for the current operation.

        Returns:
            Configured httpx.AsyncClient instance.
        """
        return httpx.AsyncClient(**self._httpx_client_args)

    def _get_client_factory(self, streaming: bool = False) -> ClientFactory:
        """Get a ClientFactory for the current operation.

        Args:
            streaming: Whether to enable streaming mode.

        Returns:
            Configured ClientFactory instance.
        """
        httpx_client = self._get_httpx_client()
        config = ClientConfig(
            httpx_client=httpx_client,
            streaming=streaming,
        )
        return ClientFactory(config)

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
        logger.info("endpoint=<%s> | discovered agent card", self.endpoint)
        return self._agent_card

    async def _send_message(self, prompt: AgentInput, streaming: bool) -> AsyncIterator[Any]:
        """Send message to A2A agent.

        Args:
            prompt: Input to send to the agent.
            streaming: Whether to use streaming mode.

        Returns:
            Async iterator of A2A events.

        Raises:
            ValueError: If prompt is None.
        """
        if prompt is None:
            raise ValueError("prompt is required for A2AAgent")

        agent_card = await self._get_agent_card()
        client = self._get_client_factory(streaming=streaming).create(agent_card)
        message = convert_input_to_message(prompt)

        logger.info("endpoint=<%s> | %s message", self.endpoint, "streaming" if streaming else "sending")
        return client.send_message(message)

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
        async for event in await self._send_message(prompt, streaming=False):
            return convert_response_to_agent_result(event)

        raise RuntimeError("No response received from A2A agent")

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
            A2A events wrapped in dictionaries with an 'a2a_event' key.

        Raises:
            ValueError: If prompt is None.
        """
        async for event in await self._send_message(prompt, streaming=True):
            yield {"a2a_event": event}
