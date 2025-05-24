"""SageMaker AI model provider.

- Docs: https://aws.amazon.com/sagemaker-ai/
"""

import json
import logging
import mimetypes
from typing import Any, Iterable, Optional, TypedDict

import boto3
from botocore.config import Config as BotocoreConfig
from typing_extensions import Unpack, override

from ..types.content import ContentBlock, Messages
from ..types.models import Model
from ..types.streaming import StreamEvent
from ..types.tools import ToolResult, ToolSpec, ToolUse

logger = logging.getLogger(__name__)


class SageMakerAIModel(Model):
    """Amazon SageMaker model provider implementation.

    The implementation handles SageMaker-specific features such as:

    - Endpoint invocation
    - Tool configuration for function calling
    - Context window overflow detection
    - Endpoint not found error handling
    - Inference component capacity error handling with automatic retries
    """

    class ModelConfig(TypedDict, total=False):
        """Configuration options for SageMaker models.

        Attributes:
            additional_args: Any additional arguments to include in the request.
            endpoint_name: The name of the SageMaker endpoint to invoke.
            inference_component_name: The name of the inference component to use.
            max_tokens: Maximum number of tokens to generate in the response.
            stop_sequences: List of sequences that will stop generation when encountered.
            temperature: Controls randomness in generation (higher = more random).
            top_p: Controls diversity via nucleus sampling (alternative to temperature).
        """

        additional_args: Optional[dict[str, Any]]
        endpoint_name: str
        inference_component_name: Optional[str]
        max_tokens: Optional[int]
        stop_sequences: Optional[list[str]]
        temperature: Optional[float]
        top_p: Optional[float]

    def __init__(
        self,
        *,
        endpoint_name: str,
        inference_component_name: Optional[str] = None,
        boto_session: Optional[boto3.Session] = None,
        boto_client_config: Optional[BotocoreConfig] = None,
        region_name: Optional[str] = None,
        **model_config: Unpack[ModelConfig],
    ):
        """Initialize provider instance.

        Args:
            endpoint_name: The name of the SageMaker endpoint to invoke.
            inference_component_name: The name of the inference component to use.
            boto_session: Boto Session to use when calling the SageMaker Runtime.
            boto_client_config: Configuration to use when creating the SageMaker-Runtime Boto Client.
            region_name: AWS region name to use for the SageMaker Runtime client.
            **model_config: Model parameters for the SageMaker request payload.
        """
        self.config = SageMakerAIModel.ModelConfig(
            endpoint_name=endpoint_name, inference_component_name=inference_component_name
        )
        self.update_config(**model_config)

        logger.debug("config=<%s> | initializing", self.config)

        if boto_session:
            session = boto_session
        elif region_name:
            session = boto3.Session(region_name=region_name)
        else:
            session = boto3.Session()

        self.client = session.client(
            service_name="sagemaker-runtime",
            config=boto_client_config,
        )

    @override
    def update_config(self, **model_config: Unpack[ModelConfig]) -> None:  # type: ignore
        """Update the SageMaker AI Model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)

    @override
    def get_config(self) -> ModelConfig:
        """Get the SageMaker AI Model configuration.

        Returns:
            The SageMaker model configuration.
        """
        return self.config

    def _format_request_message_content(self, content: ContentBlock) -> dict[str, Any]:
        """Format a SageMaker content block.

        Args:
            content: Message content.

        Returns:
            SageMaker formatted content block.
        """
        if "image" in content:
            mime_type = mimetypes.types_map.get(f".{content['image']['format']}", "application/octet-stream")
            image_data = content["image"]["source"]["bytes"].decode("utf-8")
            return {
                "image_url": {
                    "detail": "auto",
                    "format": mime_type,
                    "url": f"data:{mime_type};base64,{image_data}",
                },
                "type": "image_url",
            }

        if "reasoningContent" in content:
            return {
                "signature": content["reasoningContent"]["reasoningText"]["signature"],
                "thinking": content["reasoningContent"]["reasoningText"]["text"],
                "type": "thinking",
            }

        if "text" in content:
            return {"text": content["text"], "type": "text"}

        if "video" in content:
            return {
                "type": "video_url",
                "video_url": {
                    "detail": "auto",
                    "url": content["video"]["source"]["bytes"],
                },
            }

        return {"text": json.dumps(content), "type": "text"}

    def _format_request_message_tool_call(self, tool_use: ToolUse) -> dict[str, Any]:
        """Format a SageMaker tool call.

        Args:
            tool_use: Tool use requested by the model.

        Returns:
            SageMaker formatted tool call.
        """
        return {
            "function": {
                "arguments": json.dumps(tool_use["input"]),
                "name": tool_use["name"],
            },
            "id": tool_use["toolUseId"],
            "type": "function",
        }

    def _format_request_tool_message(self, tool_result: ToolResult) -> dict[str, Any]:
        """Format a SageMaker tool message.

        Args:
            tool_result: Tool result collected from a tool execution.

        Returns:
            SageMaker formatted tool message.
        """
        return {
            "role": "tool",
            "tool_call_id": tool_result["toolUseId"],
            "content": json.dumps(
                {
                    "content": tool_result["content"],
                    "status": tool_result["status"],
                }
            ),
        }

    def _format_request_messages(self, messages: Messages, system_prompt: Optional[str] = None) -> list[dict[str, Any]]:
        """Format a SageMaker messages array.

        Args:
            messages: List of message objects to be processed by the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            A SageMaker messages array.
        """
        formatted_messages: list[dict[str, Any]]
        formatted_messages = [{"role": "system", "content": system_prompt}] if system_prompt else []

        for message in messages:
            contents = message["content"]

            formatted_contents = [
                self._format_request_message_content(content)
                for content in contents
                if not any(block_type in content for block_type in ["toolResult", "toolUse"])
            ]
            formatted_tool_calls = [
                self._format_request_message_tool_call(content["toolUse"])
                for content in contents
                if "toolUse" in content
            ]
            formatted_tool_messages = [
                self._format_request_tool_message(content["toolResult"])
                for content in contents
                if "toolResult" in content
            ]

            formatted_message = {
                "role": message["role"],
                "content": formatted_contents,
                **({"tool_calls": formatted_tool_calls} if formatted_tool_calls else {}),
            }
            formatted_messages.append(formatted_message)
            formatted_messages.extend(formatted_tool_messages)

        return [message for message in formatted_messages if message["content"] or "tool_calls" in message]

    @override
    def format_request(
        self, messages: Messages, tool_specs: Optional[list[ToolSpec]] = None, system_prompt: Optional[str] = None
    ) -> dict[str, Any]:
        """Format a SageMaker chat streaming request.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            A SageMaker chat streaming request.
        """
        return {
            "messages": self._format_request_messages(messages, system_prompt),
            "model": "lmi",
            "stream": True,
            # "stream_options": {"include_usage": True},
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
            # **(self.config.get("params") or {}),
            **({"max_tokens": self.config["max_tokens"] if "max_tokens" in self.config else 2048}),
            **({"temperature": self.config["temperature"] if "temperature" in self.config else 0.1}),
            **({"top_p": self.config["top_p"] if "top_p" in self.config else 0.1}),
            **({"stop": self.config["stop_sequences"]} if "stop_sequences" in self.config else {}),
            **(
                self.config["additional_args"]
                if "additional_args" in self.config and self.config["additional_args"] is not None
                else {}
            ),
        }

    @override
    def format_chunk(self, event: dict[str, Any]) -> StreamEvent:
        """Format the SageMaker AI response events into standardized message chunks.

        Args:
            event: A response event from the SageMaker AI model.

        Returns:
            The formatted chunk.

        Raises:
            RuntimeError: If chunk_type is not recognized.
                This error should never be encountered as we control chunk_type in the stream method.
        """
        match event["chunk_type"]:
            case "message_start":
                return {"messageStart": {"role": "assistant"}}

            case "content_start":
                if event["data_type"] == "tool":
                    return {
                        "contentBlockStart": {
                            "start": {
                                "toolUse": {
                                    "name": event["data"]["function"]["name"],
                                    "toolUseId": event["data"]["id"],
                                }
                            }
                        }
                    }

                return {"contentBlockStart": {"start": {}}}

            case "content_delta":
                if event["data_type"] == "tool":
                    return {
                        "contentBlockDelta": {"delta": {"toolUse": {"input": event["data"]["function"]["arguments"]}}}
                    }

                return {"contentBlockDelta": {"delta": {"text": event["data"]}}}

            case "content_stop":
                return {"contentBlockStop": {}}

            case "message_stop":
                match event["data"]:
                    case "tool_calls":
                        return {"messageStop": {"stopReason": "tool_use"}}
                    case "length":
                        return {"messageStop": {"stopReason": "max_tokens"}}
                    case _:
                        return {"messageStop": {"stopReason": "end_turn"}}

            case "metadata":
                return {
                    "metadata": {
                        "usage": {
                            "inputTokens": event["data"]["prompt_tokens"],
                            "outputTokens": event["data"]["completion_tokens"],
                            "totalTokens": event["data"]["total_tokens"],
                        },
                        "metrics": {
                            "latencyMs": 0,  # TODO
                        },
                    },
                }

            case _:
                raise RuntimeError(f"chunk_type=<{event['chunk_type']} | unknown type")

    @override
    def stream(self, request: dict[str, Any]) -> Iterable[dict[str, Any]]:
        """Send the request to the SageMaker AI model and get the streaming response.

        This method calls the SageMaker AI chat API and returns the stream of response events.

        Args:
            request: The formatted request to send to the SageMaker AI model.

        Returns:
            An iterable of response events from the SageMaker AI model.
        """
        # Format the request according to the SageMaker Runtime API requirements
        # Delete content key-value from request["messages"] if content == []
        for message in request["messages"]:
            if "content" in message and message["content"] == [] and message["role"] == "assistant":
                del message["content"]
        payload = {
            "EndpointName": self.config["endpoint_name"],
            "Body": json.dumps(request),
            "ContentType": "application/json",
            "Accept": "application/json",
        }

        # Add InferenceComponentName if provided
        if self.config.get("inference_component_name"):
            payload["InferenceComponentName"] = self.config["inference_component_name"]

        # Invoke with streaming
        response = self.client.invoke_endpoint_with_response_stream(**payload)

        # Message start
        yield {"chunk_type": "message_start"}
        yield {"chunk_type": "content_start", "data_type": "text"}

        tool_calls: dict[int, list[Any]] = {}
        data = ""

        for event in response["Body"]:
            try:
                chunk_data = event["PayloadPart"]["Bytes"].decode("utf-8")
                chunk_json = json.loads(chunk_data)
            except json.JSONDecodeError:
                data += chunk_data
                try:
                    chunk_json = json.loads(data)
                    data = ""
                except json.JSONDecodeError:
                    continue

            choice = chunk_json["choices"][0]

            if choice["finish_reason"]:
                break

            content = choice["delta"].get("content", None)
            if content:
                yield {"chunk_type": "content_delta", "data_type": "text", "data": content}

            delta_tool_calls = choice["delta"].get("tool_calls", [])
            if delta_tool_calls:
                for tool_call in delta_tool_calls:
                    tool_calls.setdefault(tool_call["index"], []).append(tool_call)

        yield {"chunk_type": "content_stop", "data_type": "text"}

        for tool_deltas in tool_calls.values():
            logger.warning(tool_deltas)
            tool_start, tool_deltas = tool_deltas[0], tool_deltas[1:]
            yield {"chunk_type": "content_start", "data_type": "tool", "data": tool_start}

            for tool_delta in tool_deltas:
                yield {"chunk_type": "content_delta", "data_type": "tool", "data": tool_delta}

            yield {"chunk_type": "content_stop", "data_type": "tool"}

        yield {"chunk_type": "message_stop", "data": choice["finish_reason"]}

        # # Skip remaining events as we don't have use for anything except the final usage payload
        # for event in response['Body']:
        #     logger.warning(f"Final events: {event["PayloadPart"]["Bytes"].decode("utf-8")}")
        #     _ = event

        # yield {"chunk_type": "metadata", "data": event["usage"]}
