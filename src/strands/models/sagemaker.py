"""Amazon SageMaker model provider."""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Literal, Optional, Type, TypedDict, TypeVar, Union, cast

import boto3
from botocore.config import Config as BotocoreConfig
from pydantic import BaseModel
from typing_extensions import Unpack, override

from ..types.content import ContentBlock, Messages
from ..types.models import OpenAIModel
from ..types.tools import ToolSpec

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


@dataclass
class UsageMetadata:
    """Usage metadata for the model.

    Attributes:
        total_tokens: Total number of tokens used in the request
        completion_tokens: Number of tokens used in the completion
        prompt_tokens: Number of tokens used in the prompt
        prompt_tokens_details: Additional information about the prompt tokens (optional)
    """

    total_tokens: int
    completion_tokens: int
    prompt_tokens: int
    prompt_tokens_details: Optional[int] = 0


@dataclass
class FunctionCall:
    """Function call for the model.

    Attributes:
        name: Name of the function to call
        arguments: Arguments to pass to the function
    """

    name: str
    arguments: str

    def __init__(self, **kwargs: dict):
        """Initialize function call.

        Args:
            **kwargs: Keyword arguments for the function call.
        """
        self.name = kwargs.get("name", "")
        self.arguments = kwargs.get("arguments", "")


@dataclass
class ToolCall:
    """Tool call for the model object.

    Attributes:
        id: Tool call ID
        type: Tool call type
        function: Tool call function
    """

    id: str
    type: Literal["function"]
    function: FunctionCall

    def __init__(self, **kwargs: dict):
        """Initialize tool call object.

        Args:
            **kwargs: Keyword arguments for the tool call.
        """
        self.id = kwargs.get("id", "")
        self.type = "function"
        self.function = FunctionCall(**kwargs.get("function", {}))


class SageMakerAIModel(OpenAIModel):
    """Amazon SageMaker model provider implementation."""

    class SageMakerAIPayloadSchema(TypedDict, total=False):
        """Payload schema for the Amazon SageMaker AI model.

        Attributes:
            max_tokens: Maximum number of tokens to generate in the completion
            stream: Whether to stream the response
            temperature: Sampling temperature to use for the model (optional)
            top_p: Nucleus sampling parameter (optional)
            top_k: Top-k sampling parameter (optional)
            stop: List of stop sequences to use for the model (optional)
            additional_args: Additional request parameters, as supported by https://bit.ly/djl-lmi-request-schema
        """

        max_tokens: int
        stream: bool
        temperature: Optional[float]
        top_p: Optional[float]
        top_k: Optional[int]
        stop: Optional[list[str]]
        additional_args: Optional[dict[str, Any]]

    class SageMakerAIEndpointConfig(TypedDict, total=False):
        """Configuration options for SageMaker models.

        Attributes:
            endpoint_name: The name of the SageMaker endpoint to invoke
            inference_component_name: The name of the inference component to use

            additional_args: Other request parameters, as supported by https://bit.ly/sagemaker-invoke-endpoint-params
        """

        endpoint_name: str
        region_name: str
        inference_component_name: Union[str, None]
        target_model: Union[Optional[str], None]
        target_variant: Union[Optional[str], None]
        additional_args: Optional[dict[str, Any]]

    def __init__(
        self,
        endpoint_config: SageMakerAIEndpointConfig,
        payload_config: SageMakerAIPayloadSchema,
        boto_session: Optional[boto3.Session] = None,
        boto_client_config: Optional[BotocoreConfig] = None,
    ):
        """Initialize provider instance.

        Args:
            endpoint_config: Endpoint configuration for SageMaker.
            payload_config: Payload configuration for the model.
            boto_session: Boto Session to use when calling the SageMaker Runtime.
            boto_client_config: Configuration to use when creating the SageMaker-Runtime Boto Client.
        """
        payload_config.setdefault("stream", True)
        self.endpoint_config = dict(endpoint_config)
        self.payload_config = dict(payload_config)
        logger.debug(
            "endpoint_config=<%s> payload_config=<%s> | initializing", self.endpoint_config, self.payload_config
        )

        session = boto_session or boto3.Session(
            region_name=self.endpoint_config.get("region_name") or os.getenv("AWS_REGION") or "us-west-2",
        )

        # Add strands-agents to the request user agent
        if boto_client_config:
            existing_user_agent = getattr(boto_client_config, "user_agent_extra", None)

            # Append 'strands-agents' to existing user_agent_extra or set it if not present
            new_user_agent = f"{existing_user_agent} strands-agents" if existing_user_agent else "strands-agents"

            client_config = boto_client_config.merge(BotocoreConfig(user_agent_extra=new_user_agent))
        else:
            client_config = BotocoreConfig(user_agent_extra="strands-agents")

        self.client = session.client(
            service_name="sagemaker-runtime",
            config=client_config,
        )

    @override
    def update_config(self, **endpoint_config: Unpack[SageMakerAIEndpointConfig]) -> None:  # type: ignore[override]
        """Update the Amazon SageMaker model configuration with the provided arguments.

        Args:
            **endpoint_config: Configuration overrides.
        """
        self.endpoint_config.update(endpoint_config)

    @override
    def get_config(self) -> SageMakerAIEndpointConfig:
        """Get the Amazon SageMaker model configuration.

        Returns:
            The Amazon SageMaker model configuration.
        """
        return cast(SageMakerAIModel.SageMakerAIEndpointConfig, self.endpoint_config)

    @override
    def format_request(
        self, messages: Messages, tool_specs: Optional[list[ToolSpec]] = None, system_prompt: Optional[str] = None
    ) -> dict[str, Any]:
        """Format an Amazon SageMaker chat streaming request.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            An Amazon SageMaker chat streaming request.
        """
        payload = {
            "messages": self.format_request_messages(messages, system_prompt),
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
            # Add payload configuration parameters
            **{k: v for k, v in self.payload_config.items() if k != "additional_args"},
        }

        # Remove tools and tool_choice if tools = []
        if not payload["tools"]:
            payload.pop("tools")
            payload.pop("tool_choice", None)
        else:
            # Ensure the model can use tools when available
            payload["tool_choice"] = "auto"

        # TODO: this should be a @override of format_request_message
        for message in payload["messages"]:
            # Assistant message must have either content or tool_calls, but not both
            if message.get("role", "") == "assistant" and message.get("tool_calls", []) != []:
                _ = message.pop("content")
            # Tool messages should have content as pure text
            elif message.get("role", "") == "tool":
                logger.debug("message content:<%s> | streaming message content", message["content"])
                logger.debug("message content type:<%s> | streaming message content type", type(message["content"]))
                if isinstance(message["content"], str):
                    message["content"] = json.loads(message["content"])["content"]
                message["content"] = message["content"][0]["text"]

        logger.debug("payload=<%s>", payload)
        # Format the request according to the SageMaker Runtime API requirements
        request = {
            "EndpointName": self.endpoint_config["endpoint_name"],
            "Body": json.dumps(payload),
            "ContentType": "application/json",
            "Accept": "application/json",
        }

        # Add optional SageMaker parameters if provided
        if self.endpoint_config.get("inference_component_name"):
            request["InferenceComponentName"] = self.endpoint_config["inference_component_name"]
        if self.endpoint_config.get("target_model"):
            request["TargetModel"] = self.endpoint_config["target_model"]
        if self.endpoint_config.get("target_variant"):
            request["TargetVariant"] = self.endpoint_config["target_variant"]

        # Add additional args if provided
        if self.endpoint_config.get("additional_args"):
            request.update(self.endpoint_config["additional_args"])

        return request

    @override
    async def stream(self, request: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        """Send the request to the Amazon SageMaker AI model and get the streaming response.

        This method calls the Amazon SageMaker AI chat API and returns the stream of response events.

        Args:
            request: The formatted request to send to the Amazon SageMaker AI model.

        Returns:
            An iterable of response events from the Amazon SageMaker AI model.
        """
        try:
            if self.payload_config.get("stream", True):
                response = self.client.invoke_endpoint_with_response_stream(**request)

                # Message start
                yield {"chunk_type": "message_start"}

                # Parse the content
                finish_reason = ""
                partial_content = ""
                tool_calls: dict[int, list[Any]] = {}
                has_text_content = False
                text_content_started = False
                reasoning_content_started = False

                for event in response["Body"]:
                    chunk = event["PayloadPart"]["Bytes"].decode("utf-8")
                    partial_content += chunk  # Some messages are randomly split and not JSON decodable- not sure why
                    try:
                        content = json.loads(partial_content)
                        partial_content = ""
                        choice = content["choices"][0]

                        # Handle text content
                        if choice["delta"].get("content", None):
                            if not text_content_started:
                                yield {"chunk_type": "content_start", "data_type": "text"}
                                text_content_started = True
                            has_text_content = True
                            yield {
                                "chunk_type": "content_delta",
                                "data_type": "text",
                                "data": choice["delta"]["content"],
                            }

                        # Handle reasoning content
                        if choice["delta"].get("reasoning_content", None):
                            if not reasoning_content_started:
                                yield {"chunk_type": "content_start", "data_type": "reasoning_content"}
                                reasoning_content_started = True
                            yield {
                                "chunk_type": "content_delta",
                                "data_type": "reasoning_content",
                                "data": choice["delta"]["reasoning_content"],
                            }

                        # Handle tool calls
                        for tool_call in choice["delta"].get("tool_calls", []):
                            tool_calls.setdefault(tool_call["index"], []).append(tool_call)

                        if choice["finish_reason"] is not None:
                            finish_reason = choice["finish_reason"]
                            break

                    except json.JSONDecodeError:
                        # Continue accumulating content until we have valid JSON
                        continue

                # Close reasoning content if it was started
                if reasoning_content_started:
                    yield {"chunk_type": "content_stop", "data_type": "reasoning_content"}

                # Close text content if it was started
                if text_content_started:
                    yield {"chunk_type": "content_stop", "data_type": "text"}

                # Handle tool calling
                for tool_deltas in tool_calls.values():
                    yield {"chunk_type": "content_start", "data_type": "tool", "data": ToolCall(**tool_deltas[0])}
                    for tool_delta in tool_deltas:
                        yield {"chunk_type": "content_delta", "data_type": "tool", "data": ToolCall(**tool_delta)}
                    yield {"chunk_type": "content_stop", "data_type": "tool"}

                # If no content was generated at all, ensure we have empty text content
                if not has_text_content and not tool_calls:
                    yield {"chunk_type": "content_start", "data_type": "text"}
                    yield {"chunk_type": "content_stop", "data_type": "text"}

                # Message close
                yield {"chunk_type": "message_stop", "data": finish_reason}

                # Return metadata
                if choice.get("usage", None):
                    yield {"chunk_type": "metadata", "data": UsageMetadata(**choice["usage"])}

            else:
                # Not all SageMaker AI models support streaming!
                response = self.client.invoke_endpoint(**request)
                final_response_json = json.loads(response["Body"].read().decode("utf-8"))

                # Obtain the key elements from the response
                message = final_response_json["choices"][0]["message"]
                message_stop_reason = final_response_json["choices"][0]["finish_reason"]

                # Message start
                yield {"chunk_type": "message_start"}

                # Handle text
                yield {"chunk_type": "content_start", "data_type": "text"}
                yield {"chunk_type": "content_delta", "data_type": "text", "data": message["content"] or ""}
                yield {"chunk_type": "content_stop", "data_type": "text"}

                # Handle reasoning content
                if message.get("reasoning_content", None):
                    yield {"chunk_type": "content_start", "data_type": "reasoning_content"}
                    yield {
                        "chunk_type": "content_delta",
                        "data_type": "reasoning_content",
                        "data": message["reasoning_content"],
                    }
                    yield {"chunk_type": "content_stop", "data_type": "reasoning_content"}

                # Handle the tool calling, if any
                if message_stop_reason == "tool_calls":
                    for tool_call in message["tool_calls"] or []:
                        yield {"chunk_type": "content_start", "data_type": "tool", "data": ToolCall(**tool_call)}
                        yield {"chunk_type": "content_delta", "data_type": "tool", "data": ToolCall(**tool_call)}
                        yield {"chunk_type": "content_stop", "data_type": "tool", "data": ToolCall(**tool_call)}

                # Message close
                yield {"chunk_type": "message_stop", "data": message_stop_reason}
                # Handle usage metadata
                yield {"chunk_type": "metadata", "data": UsageMetadata(**final_response_json["usage"])}
        except (
            self.client.exceptions.InternalFailure,
            self.client.exceptions.ServiceUnavailable,
            self.client.exceptions.ValidationError,
            self.client.exceptions.ModelError,
            self.client.exceptions.InternalDependencyException,
            self.client.exceptions.ModelNotReadyException,
        ) as e:
            logger.error("SageMaker error: %s", str(e))
            raise e

    @override
    @classmethod
    def format_request_message_content(cls, content: ContentBlock) -> dict[str, Any]:
        """Format a content block.

        Args:
            content: Message content.

        Returns:
            Formatted content block.

        Raises:
            TypeError: If the content block type cannot be converted to a SageMaker-compatible format.
        """
        if "reasoningContent" in content:
            return {
                "signature": content["reasoningContent"]["reasoningText"]["signature"],
                "thinking": content["reasoningContent"]["reasoningText"]["text"],
                "type": "thinking",
            }

        if "video" in content:
            return {
                "type": "video_url",
                "video_url": {
                    "detail": "auto",
                    "url": content["video"]["source"]["bytes"],
                },
            }

        return super().format_request_message_content(content)

    @override
    async def structured_output(
        self, output_model: Type[T], prompt: Messages
    ) -> AsyncGenerator[dict[str, Union[T, Any]], None]:
        """Get structured output from the model.

        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.

        Yields:
            Model events with the last being the structured output.
        """
        # Format the request for structured output
        request = self.format_request(prompt)

        # Parse the payload to add response format
        payload = json.loads(request["Body"])
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": output_model.__name__, "schema": output_model.model_json_schema(), "strict": True},
        }
        request["Body"] = json.dumps(payload)

        try:
            # Use non-streaming mode for structured output
            response = self.client.invoke_endpoint(**request)
            final_response_json = json.loads(response["Body"].read().decode("utf-8"))

            # Extract the structured content
            message = final_response_json["choices"][0]["message"]

            if message.get("content"):
                try:
                    # Parse the JSON content and create the output model instance
                    content_data = json.loads(message["content"])
                    parsed_output = output_model(**content_data)
                    yield {"output": parsed_output}
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    raise ValueError(f"Failed to parse structured output: {e}") from e
            else:
                raise ValueError("No content found in SageMaker response")

        except (
            self.client.exceptions.InternalFailure,
            self.client.exceptions.ServiceUnavailable,
            self.client.exceptions.ValidationError,
            self.client.exceptions.ModelError,
            self.client.exceptions.InternalDependencyException,
            self.client.exceptions.ModelNotReadyException,
        ) as e:
            logger.error("SageMaker structured output error: %s", str(e))
            raise ValueError(f"SageMaker structured output error: {str(e)}") from e
