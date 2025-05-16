"""Ollama model provider.

- Docs: https://ollama.com/
"""

import json
import logging
from typing import Any, Iterable, Optional, Union

from ollama import Client as OllamaClient
from typing_extensions import TypedDict, Unpack, override

from ..types.content import ContentBlock, Message, Messages
from ..types.media import DocumentContent, ImageContent
from ..types.models import Model
from ..types.streaming import StopReason, StreamEvent
from ..types.tools import ToolSpec

logger = logging.getLogger(__name__)


class OllamaModel(Model):
    """Ollama model provider implementation.

    The implementation handles Ollama-specific features such as:

    - Local model invocation
    - Streaming responses
    - Tool/function calling
    """

    class OllamaConfig(TypedDict, total=False):
        """Configuration parameters for Ollama models.

        Attributes:
            additional_args: Any additional arguments to include in the request.
            keep_alive: Controls how long the model will stay loaded into memory following the request (default: "5m").
            max_tokens: Maximum number of tokens to generate in the response.
            model_id: Ollama model ID (e.g., "llama3", "mistral", "phi3").
            options: Additional model parameters (e.g., top_k).
            stop_sequences: List of sequences that will stop generation when encountered.
            temperature: Controls randomness in generation (higher = more random).
            top_p: Controls diversity via nucleus sampling (alternative to temperature).
        """

        additional_args: Optional[dict[str, Any]]
        keep_alive: Optional[str]
        max_tokens: Optional[int]
        model_id: str
        options: Optional[dict[str, Any]]
        stop_sequences: Optional[list[str]]
        temperature: Optional[float]
        top_p: Optional[float]

    def __init__(
        self,
        host: Optional[str],
        *,
        ollama_client_args: Optional[dict[str, Any]] = None,
        **model_config: Unpack[OllamaConfig],
    ) -> None:
        """Initialize provider instance.

        Args:
            host: The address of the Ollama server hosting the model.
            ollama_client_args: Additional arguments for the Ollama client.
            **model_config: Configuration options for the Ollama model.
        """
        self.config = OllamaModel.OllamaConfig(**model_config)

        logger.debug("config=<%s> | initializing", self.config)

        ollama_client_args = ollama_client_args if ollama_client_args is not None else {}

        self.client = OllamaClient(host, **ollama_client_args)

    @override
    def update_config(self, **model_config: Unpack[OllamaConfig]) -> None:  # type: ignore
        """Update the Ollama Model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)

    @override
    def get_config(self) -> OllamaConfig:
        """Get the Ollama model configuration.

        Returns:
            The Ollama model configuration.
        """
        return self.config

    @override
    def format_request(
        self, messages: Messages, tool_specs: Optional[list[ToolSpec]] = None, system_prompt: Optional[str] = None
    ) -> dict[str, Any]:
        """Format an Ollama chat streaming request.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            An Ollama chat streaming request.
        """

        def format_message(message: Message, content: ContentBlock) -> dict[str, Any]:
            if "text" in content:
                return {"role": message["role"], "content": content["text"]}

            if "image" in content:
                return {"role": message["role"], "images": [content["image"]["source"]["bytes"]]}

            if "toolUse" in content:
                return {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": content["toolUse"]["toolUseId"],
                                "arguments": content["toolUse"]["input"],
                            }
                        }
                    ],
                }

            if "toolResult" in content:
                result_content: Union[str, ImageContent, DocumentContent, Any] = None
                result_images = []
                for tool_result_content in content["toolResult"]["content"]:
                    if "text" in tool_result_content:
                        result_content = tool_result_content["text"]
                    elif "json" in tool_result_content:
                        result_content = tool_result_content["json"]
                    elif "image" in tool_result_content:
                        result_content = "see images"
                        result_images.append(tool_result_content["image"]["source"]["bytes"])
                    else:
                        result_content = content["toolResult"]["content"]

                return {
                    "role": "tool",
                    "content": json.dumps(
                        {
                            "name": content["toolResult"]["toolUseId"],
                            "result": result_content,
                            "status": content["toolResult"]["status"],
                        }
                    ),
                    **({"images": result_images} if result_images else {}),
                }

            return {"role": message["role"], "content": json.dumps(content)}

        def format_messages() -> list[dict[str, Any]]:
            return [format_message(message, content) for message in messages for content in message["content"]]

        formatted_messages = format_messages()

        return {
            "messages": [
                *([{"role": "system", "content": system_prompt}] if system_prompt else []),
                *formatted_messages,
            ],
            "model": self.config["model_id"],
            "options": {
                **(self.config.get("options") or {}),
                **{
                    key: value
                    for key, value in [
                        ("num_predict", self.config.get("max_tokens")),
                        ("temperature", self.config.get("temperature")),
                        ("top_p", self.config.get("top_p")),
                        ("stop", self.config.get("stop_sequences")),
                    ]
                    if value is not None
                },
            },
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_spec["name"],
                        "description": tool_spec["description"],
                        "parameters": tool_spec["inputSchema"]["json"],
                    },
                }
                for tool_spec in tool_specs or []
            ],
            **({"keep_alive": self.config["keep_alive"]} if self.config.get("keep_alive") else {}),
            **(
                self.config["additional_args"]
                if "additional_args" in self.config and self.config["additional_args"] is not None
                else {}
            ),
        }

    @override
    def format_chunk(self, event: dict[str, Any]) -> StreamEvent:
        """Format the Ollama response events into standardized message chunks.

        Args:
            event: A response event from the Ollama model.

        Returns:
            The formatted chunk.

        Raises:
            RuntimeError: If chunk_type is not recognized.
                This error should never be encountered as we control chunk_type in the stream method.
        """
        if event["chunk_type"] == "message_start":
            return {"messageStart": {"role": "assistant"}}

        if event["chunk_type"] == "content_start":
            if event["data_type"] == "text":
                return {"contentBlockStart": {"start": {}}}

            tool_name = event["data"].function.name
            return {"contentBlockStart": {"start": {"toolUse": {"name": tool_name, "toolUseId": tool_name}}}}

        if event["chunk_type"] == "content_delta":
            if event["data_type"] == "text":
                return {"contentBlockDelta": {"delta": {"text": event["data"]}}}

            tool_arguments = event["data"].function.arguments
            return {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(tool_arguments)}}}}

        if event["chunk_type"] == "content_stop":
            return {"contentBlockStop": {}}

        if event["chunk_type"] == "message_stop":
            reason: StopReason
            if event["data"] == "tool_use":
                reason = "tool_use"
            elif event["data"] == "length":
                reason = "max_tokens"
            else:
                reason = "end_turn"

            return {"messageStop": {"stopReason": reason}}

        if event["chunk_type"] == "metadata":
            return {
                "metadata": {
                    "usage": {
                        "inputTokens": event["data"].eval_count,
                        "outputTokens": event["data"].prompt_eval_count,
                        "totalTokens": event["data"].eval_count + event["data"].prompt_eval_count,
                    },
                    "metrics": {
                        "latencyMs": event["data"].total_duration / 1e6,
                    },
                },
            }

        raise RuntimeError(f"chunk_type=<{event['chunk_type']} | unknown type")

    @override
    def stream(self, request: dict[str, Any]) -> Iterable[dict[str, Any]]:
        """Send the request to the Ollama model and get the streaming response.

        This method calls the Ollama chat API and returns the stream of response events.

        Args:
            request: The formatted request to send to the Ollama model.

        Returns:
            An iterable of response events from the Ollama model.
        """
        tool_requested = False

        response = self.client.chat(**request)

        yield {"chunk_type": "message_start"}
        yield {"chunk_type": "content_start", "data_type": "text"}

        for event in response:
            for tool_call in event.message.tool_calls or []:
                yield {"chunk_type": "content_start", "data_type": "tool", "data": tool_call}
                yield {"chunk_type": "content_delta", "data_type": "tool", "data": tool_call}
                yield {"chunk_type": "content_stop", "data_type": "tool", "data": tool_call}
                tool_requested = True

            yield {"chunk_type": "content_delta", "data_type": "text", "data": event.message.content}

        yield {"chunk_type": "content_stop", "data_type": "text"}
        yield {"chunk_type": "message_stop", "data": "tool_use" if tool_requested else event.done_reason}
        yield {"chunk_type": "metadata", "data": event}
