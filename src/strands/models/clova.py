"""CLOVA Studio model provider for Strands Agents SDK."""

import json
import logging
import os
from typing import Any, AsyncGenerator, AsyncIterable, Dict, List, Optional, Type, Union

import httpx
from pydantic import BaseModel

from ..types.content import Messages
from ..types.streaming import StreamEvent
from ..types.tools import ToolSpec
from .model import Model

logger = logging.getLogger(__name__)


class ClovaModelException(Exception):
    """Exception for CLOVA model errors."""

    pass


class ClovaModel(Model):
    """CLOVA Studio model provider implementation."""

    def __init__(
        self,
        model: str = "HCX-005",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 0.8,
        top_k: int = 0,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Initialize CLOVA model.

        Args:
            model: Model ID (default: HCX-005)
            api_key: CLOVA API key (can be set via CLOVA_API_KEY env var)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repeat_penalty: Repetition penalty
            stop: List of stop sequences
            **kwargs: Additional parameters
        """
        self.model = model
        self.api_key = api_key or os.getenv("CLOVA_API_KEY")

        if not self.api_key:
            raise ValueError(
                "CLOVA API key is required. Set CLOVA_API_KEY environment variable or pass api_key parameter."
            )

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty
        self.stop = stop or []
        self.base_url = f"https://clovastudio.stream.ntruss.com/v3/chat-completions/{model}"

        # Store additional kwargs for future use
        self.additional_params = kwargs

    def update_config(self, **model_config: Any) -> None:
        """Update the model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        for key, value in model_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.additional_params[key] = value

    def get_config(self) -> Dict[str, Any]:
        """Return the model configuration.

        Returns:
            The model's configuration.
        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "stop": self.stop,
            **self.additional_params,
        }

    async def stream(
        self,
        messages: Union[Messages, str],
        tool_specs: Optional[List[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """Stream responses from CLOVA model.

        Args:
            messages: Messages to be processed by the model.
            tool_specs: List of tool specifications (not yet supported).
            system_prompt: Optional system message.
            **kwargs: Additional parameters.

        Yields:
            Formatted message chunks from the model.
        """
        if tool_specs:
            logger.warning("Tool specs are not yet supported for CLOVA models")

        # Convert messages to CLOVA format
        clova_messages = []

        if system_prompt:
            clova_messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

        # Handle both Messages type and simple string
        if isinstance(messages, str):
            clova_messages.append({"role": "user", "content": [{"type": "text", "text": messages}]})
        elif hasattr(messages, "__iter__"):
            for msg in messages:
                if hasattr(msg, "role") and hasattr(msg, "content"):
                    # Convert content to CLOVA format
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    clova_messages.append({"role": msg.role, "content": [{"type": "text", "text": content}]})
                else:
                    # Fallback for dict-like messages
                    if isinstance(msg, dict) and "content" in msg:
                        if isinstance(msg["content"], str):
                            msg["content"] = [{"type": "text", "text": msg["content"]}]
                    clova_messages.append(msg)

        request_body = {
            "messages": clova_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "maxTokens": kwargs.get("max_tokens", self.max_tokens),
            "topP": kwargs.get("top_p", self.top_p),
            "topK": kwargs.get("top_k", self.top_k),
            "repetitionPenalty": kwargs.get("repeat_penalty", self.repeat_penalty),
            "stop": kwargs.get("stop", self.stop),
            "seed": kwargs.get("seed", 0),
            "includeAiFilters": kwargs.get("includeAiFilters", True),
            "stream": True,
        }

        # Add any additional parameters from initialization or kwargs
        for key, value in self.additional_params.items():
            if key not in request_body:
                request_body[key] = value

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream",
        }

        # Add required request ID header
        request_id = os.getenv("CLOVA_REQUEST_ID", "test-request-001")
        headers["X-NCP-CLOVASTUDIO-REQUEST-ID"] = request_id

        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            response = await client.post(
                self.base_url,
                json=request_body,
                headers=headers,
            )

            if response.status_code != 200:
                error_text = await response.aread()
                error_msg = f"CLOVA API request failed with status {response.status_code}: {error_text.decode('utf-8')}"
                raise ClovaModelException(error_msg)

            # Process SSE stream
            buffer = b""
            async for chunk in response.aiter_bytes():
                buffer += chunk
                # Split by double newline which separates SSE events
                events = buffer.split(b"\n\n")
                # Keep the last incomplete event in buffer
                buffer = events[-1]

                # Process complete events
                for event in events[:-1]:
                    if not event:
                        continue

                    # Parse SSE event
                    lines = event.split(b"\n")
                    data_line = None
                    for line in lines:
                        if line.startswith(b"data:"):
                            data_line = line[5:].strip()
                            break

                    if not data_line:
                        continue

                    try:
                        data_str = data_line.decode("utf-8")
                        data = json.loads(data_str)

                        # Handle different event types and convert to StreamEvent
                        # CLOVA returns content in message.content format
                        if "message" in data and data["message"].get("content"):
                            # Yield as a StreamEvent dict with text chunk
                            yield {
                                "type": "text",
                                "text": data["message"]["content"],
                            }

                        # Check for finish reason (not stopReason)
                        if "finishReason" in data and data["finishReason"] == "stop":
                            # Yield completion event
                            yield {
                                "type": "message_stop",
                                "stop_reason": "stop",
                            }
                            break

                    except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
                        # Skip malformed data
                        continue

    async def structured_output(
        self,
        output_model: Type[BaseModel],
        prompt: Messages,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Dict[str, Union[BaseModel, Any]], None]:
        """Get structured output from the model.

        Note: This is not yet implemented for CLOVA models.

        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: Structured output is not yet supported for CLOVA models.
        """
        raise NotImplementedError("Structured output is not yet supported for CLOVA models")
        # Make this a generator (unreachable code, but satisfies type hint)
        yield  # pragma: no cover

    def __str__(self) -> str:
        """String representation of the model."""
        return f"ClovaModel(model='{self.model}', temperature={self.temperature}, max_tokens={self.max_tokens})"
