"""Google Gemini model provider.

- Docs: https://ai.google.dev/docs/gemini_api_overview
"""

import base64
import json
import logging
import mimetypes
from typing import Any, Iterable, Optional, TypedDict

import google.generativeai.generative_models as genai  # mypy: disable-error-code=import
from typing_extensions import Required, Unpack, override

from ..types.content import ContentBlock, Messages
from ..types.exceptions import ContextWindowOverflowException, ModelThrottledException
from ..types.models import Model
from ..types.streaming import StreamEvent
from ..types.tools import ToolSpec

logger = logging.getLogger(__name__)


class GeminiModel(Model):
    """Google Gemini model provider implementation."""

    EVENT_TYPES = {
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_stop",
    }

    OVERFLOW_MESSAGES = {
        "input is too long",
        "input length exceeds context window",
        "input and output tokens exceed your context limit",
    }

    class GeminiConfig(TypedDict, total=False):
        """Configuration options for Gemini models.

        Attributes:
            max_tokens: Maximum number of tokens to generate.
            model_id: Gemini model ID (e.g., "gemini-pro").
                For a complete list of supported models, see
                https://ai.google.dev/models/gemini.
            params: Additional model parameters (e.g., temperature).
                For a complete list of supported parameters, see
                https://ai.google.dev/docs/gemini_api_overview#generation_config.
        """

        max_tokens: Required[int]
        model_id: Required[str]
        params: Optional[dict[str, Any]]

    def __init__(self, *, client_args: Optional[dict[str, Any]] = None, **model_config: Unpack[GeminiConfig]):
        """Initialize provider instance.

        Args:
            client_args: Arguments for the underlying Gemini client (e.g., api_key).
                For a complete list of supported arguments, see
                https://ai.google.dev/docs/gemini_api_overview#client_libraries.
            **model_config: Configuration options for the Gemini model.
        """
        self.config = GeminiModel.GeminiConfig(**model_config)

        logger.debug("config=<%s> | initializing", self.config)

        client_args = client_args or {}
        genai.client.configure(**client_args)
        self.model = genai.GenerativeModel(self.config["model_id"])

    @override
    def update_config(self, **model_config: Unpack[GeminiConfig]) -> None:  # type: ignore[override]
        """Update the Gemini model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)
        self.model = genai.GenerativeModel(self.config["model_id"])

    @override
    def get_config(self) -> GeminiConfig:
        """Get the Gemini model configuration.

        Returns:
            The Gemini model configuration.
        """
        return self.config

    def _format_request_message_content(self, content: ContentBlock) -> dict[str, Any]:
        """Format a Gemini content block.

        Args:
            content: Message content.

        Returns:
            Gemini formatted content block.
        """
        if "image" in content:
            return {
                "inline_data": {
                    "data": base64.b64encode(content["image"]["source"]["bytes"]).decode("utf-8"),
                    "mime_type": mimetypes.types_map.get(f".{content['image']['format']}", "application/octet-stream"),
                }
            }

        if "text" in content:
            return {"text": content["text"]}

        return {"text": json.dumps(content)}

    def _format_request_messages(self, messages: Messages) -> list[dict[str, Any]]:
        """Format a Gemini messages array.

        Args:
            messages: List of message objects to be processed by the model.

        Returns:
            A Gemini messages array.
        """
        formatted_messages = []

        for message in messages:
            formatted_contents = []

            for content in message["content"]:
                if "cachePoint" in content:
                    continue

                formatted_contents.append(self._format_request_message_content(content))

            if formatted_contents:
                formatted_messages.append({"role": message["role"], "parts": formatted_contents})

        return formatted_messages

    @override
    def format_request(
        self, messages: Messages, tool_specs: Optional[list[ToolSpec]] = None, system_prompt: Optional[str] = None
    ) -> dict[str, Any]:
        """Format a Gemini streaming request.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            A Gemini streaming request.
        """
        generation_config = {"max_output_tokens": self.config["max_tokens"], **(self.config.get("params") or {})}

        return {
            "contents": self._format_request_messages(messages),
            "generation_config": generation_config,
            "tools": [
                {
                    "function_declarations": [
                        {
                            "name": tool_spec["name"],
                            "description": tool_spec["description"],
                            "parameters": tool_spec["inputSchema"]["json"],
                        }
                        for tool_spec in tool_specs or []
                    ]
                }
            ]
            if tool_specs
            else None,
            "system_instruction": system_prompt,
        }

    @override
    def format_chunk(self, event: dict[str, Any]) -> StreamEvent:
        """Format the Gemini response events into standardized message chunks.

        Args:
            event: A response event from the Gemini model.

        Returns:
            The formatted chunk.

        Raises:
            RuntimeError: If chunk_type is not recognized.
                This error should never be encountered as we control chunk_type in the stream method.
        """
        match event["type"]:
            case "message_start":
                return {"messageStart": {"role": "assistant"}}

            case "content_block_start":
                return {"contentBlockStart": {"start": {}}}

            case "content_block_delta":
                return {"contentBlockDelta": {"delta": {"text": event["text"]}}}

            case "content_block_stop":
                return {"contentBlockStop": {}}

            case "message_stop":
                return {"messageStop": {"stopReason": event["stop_reason"]}}

            case "metadata":
                return {
                    "metadata": {
                        "usage": {
                            "inputTokens": event["usage"]["prompt_token_count"],
                            "outputTokens": event["usage"]["candidates_token_count"],
                            "totalTokens": event["usage"]["total_token_count"],
                        },
                        "metrics": {
                            "latencyMs": 0,
                        },
                    }
                }

            case _:
                raise RuntimeError(f"event_type=<{event['type']} | unknown type")

    @override
    def stream(self, request: dict[str, Any]) -> Iterable[dict[str, Any]]:
        """Send the request to the Gemini model and get the streaming response.

        Args:
            request: The formatted request to send to the Gemini model.

        Returns:
            An iterable of response events from the Gemini model.

        Raises:
            ContextWindowOverflowException: If the input exceeds the model's context window.
            ModelThrottledException: If the request is throttled by Gemini.
        """
        try:
            response = self.model.generate_content(**request, stream=True)

            yield {"type": "message_start"}
            yield {"type": "content_block_start"}

            for chunk in response:
                if chunk.text:
                    yield {"type": "content_block_delta", "text": chunk.text}

            yield {"type": "content_block_stop"}
            yield {"type": "message_stop", "stop_reason": "end_turn"}

            # Get usage information
            usage = response.usage_metadata
            yield {
                "type": "metadata",
                "usage": {
                    "prompt_token_count": usage.prompt_token_count,
                    "candidates_token_count": usage.candidates_token_count,
                    "total_token_count": usage.total_token_count,
                },
            }

        except Exception as error:
            if "quota" in str(error).lower():
                raise ModelThrottledException(str(error)) from error

            if any(overflow_message in str(error).lower() for overflow_message in GeminiModel.OVERFLOW_MESSAGES):
                raise ContextWindowOverflowException(str(error)) from error

            raise error
