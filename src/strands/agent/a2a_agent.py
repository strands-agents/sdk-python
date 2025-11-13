"""A2A Agent client for Strands Agents.

This module provides the A2AAgent class, which acts as a client wrapper for remote A2A agents,
allowing them to be used in graphs, swarms, and other multi-agent patterns.
"""

import logging
from typing import Any, AsyncIterator, cast
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import AgentCard, Part, Role, Task, TaskArtifactUpdateEvent, TaskStatusUpdateEvent, TextPart
from a2a.types import Message as A2AMessage

from .._async import run_async
from ..telemetry.metrics import EventLoopMetrics
from ..types.agent import AgentInput
from ..types.content import ContentBlock, Message
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

    async def _discover_agent_card(self) -> AgentCard:
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

    def _convert_input_to_message(self, prompt: AgentInput) -> A2AMessage:
        """Convert AgentInput to A2A Message.

        Args:
            prompt: Input in various formats (string, message list, or content blocks).

        Returns:
            A2AMessage ready to send to the remote agent.

        Raises:
            ValueError: If prompt format is unsupported.
        """
        message_id = uuid4().hex

        if isinstance(prompt, str):
            return A2AMessage(
                kind="message",
                role=Role.user,
                parts=[Part(TextPart(kind="text", text=prompt))],
                message_id=message_id,
            )

        if isinstance(prompt, list) and prompt and (isinstance(prompt[0], dict)):
            if "role" in prompt[0]:
                # Message list - extract last user message
                for msg in reversed(prompt):
                    if msg.get("role") == "user":
                        content = cast(list[ContentBlock], msg.get("content", []))
                        parts = self._convert_content_blocks_to_parts(content)
                        return A2AMessage(
                            kind="message",
                            role=Role.user,
                            parts=parts,
                            message_id=message_id,
                        )
            else:
                # ContentBlock list
                parts = self._convert_content_blocks_to_parts(cast(list[ContentBlock], prompt))
                return A2AMessage(
                    kind="message",
                    role=Role.user,
                    parts=parts,
                    message_id=message_id,
                )

        raise ValueError(f"Unsupported input type: {type(prompt)}")

    def _convert_content_blocks_to_parts(self, content_blocks: list[ContentBlock]) -> list[Part]:
        """Convert Strands ContentBlocks to A2A Parts.

        Args:
            content_blocks: List of Strands content blocks.

        Returns:
            List of A2A Part objects.
        """
        parts = []
        for block in content_blocks:
            if "text" in block:
                parts.append(Part(TextPart(kind="text", text=block["text"])))
        return parts

    def _convert_response_to_agent_result(self, response: Any) -> AgentResult:
        """Convert A2A response to AgentResult.

        Args:
            response: A2A response (either A2AMessage or tuple of task and update event).

        Returns:
            AgentResult with extracted content and metadata.
        """
        content: list[ContentBlock] = []

        if isinstance(response, tuple) and len(response) == 2:
            task, update_event = response
            if update_event is None and task and hasattr(task, "artifacts"):
                # Non-streaming response: extract from task artifacts
                for artifact in task.artifacts:
                    if hasattr(artifact, "parts"):
                        for part in artifact.parts:
                            if hasattr(part, "root") and hasattr(part.root, "text"):
                                content.append({"text": part.root.text})
        elif isinstance(response, A2AMessage):
            # Direct message response
            for part in response.parts:
                if hasattr(part, "root") and hasattr(part.root, "text"):
                    content.append({"text": part.root.text})

        message: Message = {
            "role": "assistant",
            "content": content,
        }

        return AgentResult(
            stop_reason="end_turn",
            message=message,
            metrics=EventLoopMetrics(),
            state={},
        )

    async def _send_message(
        self, prompt: AgentInput, streaming: bool
    ) -> AsyncIterator[tuple[Task, TaskStatusUpdateEvent | TaskArtifactUpdateEvent | None] | A2AMessage]:
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

        agent_card = await self._discover_agent_card()
        client = self._get_client_factory(streaming=streaming).create(agent_card)
        message = self._convert_input_to_message(prompt)

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
            return self._convert_response_to_agent_result(event)

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
