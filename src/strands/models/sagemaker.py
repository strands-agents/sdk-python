"""Amazon SageMaker model provider."""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional, TypedDict, cast, Union

import boto3
from botocore.config import Config as BotocoreConfig
from typing_extensions import Unpack, override

from ..types.content import Messages
from ..types.models import OpenAIModel
from ..types.tools import ToolSpec

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
        self.name = kwargs.get("name")
        self.arguments = kwargs.get("arguments")


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
        self.id = kwargs.get("id")
        self.type = "function"
        self.function = FunctionCall(**kwargs.get("function"))


class SageMakerAIModel(OpenAIModel):
    """Amazon SageMaker model provider implementation."""

    class SageMakerAIModelConfig(TypedDict, total=False):
        """Configuration options for SageMaker models.

        Attributes:
            endpoint_name: The name of the SageMaker endpoint to invoke
            inference_component_name: The name of the inference component to use
            stream: Whether streaming is enabled or not (default: True)
            additional_args: Other request parameters, as supported by https://bit.ly/djl-lmi-request-schema
        """

        endpoint_name: str
        inference_component_name: Union[str, None]
        stream: bool
        additional_args: Optional[dict[str, Any]]

    def __init__(
        self,
        model_config: SageMakerAIModelConfig,
        boto_session: Optional[boto3.Session] = None,
        boto_client_config: Optional[BotocoreConfig] = None,
        region_name: Optional[str] = None,
    ):
        """Initialize provider instance.

        Args:
            model_config: Model parameters for the SageMaker request payload.
            region_name: Name of the AWS region (e.g.: us-west-2)
            boto_session: Boto Session to use when calling the SageMaker Runtime.
            boto_client_config: Configuration to use when creating the SageMaker-Runtime Boto Client.
        """
        self.config = dict(model_config)

        logger.debug("config=<%s> | initializing", self.config)

        session = boto_session or boto3.Session(
            region_name=region_name or os.getenv("AWS_REGION") or "us-west-2",
        )

        # Add strands-agents to the request user agent
        if boto_client_config:
            existing_user_agent = getattr(boto_client_config, "user_agent_extra", None)

            # Append 'strands-agents' to existing user_agent_extra or set it if not present
            if existing_user_agent:
                new_user_agent = f"{existing_user_agent} strands-agents"
            else:
                new_user_agent = "strands-agents"

            client_config = boto_client_config.merge(BotocoreConfig(user_agent_extra=new_user_agent))
        else:
            client_config = BotocoreConfig(user_agent_extra="strands-agents")

        self.client = session.client(
            service_name="sagemaker-runtime",
            config=client_config,
        )

    @override
    def update_config(self, **model_config: Unpack[SageMakerAIModelConfig]) -> None:  # type: ignore[override]
        """Update the Amazon SageMaker model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)

    @override
    def get_config(self) -> SageMakerAIModelConfig:
        """Get the Amazon SageMaker model configuration.

        Returns:
            The Amazon SageMaker model configuration.
        """
        return cast(SageMakerAIModel.SageMakerAIModelConfig, self.config)

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
            # Add all key-values from the model config to the payload except endpoint_name and inference_component_name
            **{k: v for k, v in self.config.items() if k not in ["endpoint_name", "inference_component_name"]},
        }

        # TODO: this should be a @override of format_request_message
        for message in payload["messages"]:
            # Assistant message must have either content or tool_calls, but not both
            if message.get("role", "") == "assistant" and message.get("tool_calls", []) != []:
                _ = message.pop("content")
            # Tool messages should have content as pure text
            elif message.get("role", "") == "tool":
                logger.debug("message content:<%s> | streaming message content", message["content"])
                logger.debug("message content type:<%s> | streaming message content type", type(message["content"]))
                if type(message["content"]) == str:
                    message["content"] = json.loads(message["content"])["content"]
                message["content"] = message["content"][0]["text"]

        logger.debug("payload=<%s>", payload)
        # Format the request according to the SageMaker Runtime API requirements
        request = {
            "EndpointName": self.config["endpoint_name"],
            "Body": json.dumps(payload),
            "ContentType": "application/json",
            "Accept": "application/json",
        }

        # Add InferenceComponentName if provided
        if self.config.get("inference_component_name"):
            request["InferenceComponentName"] = self.config["inference_component_name"]
        return request

    @override
    def stream(self, request: dict[str, Any]) -> Iterable[dict[str, Any]]:
        """Send the request to the Amazon SageMaker AI model and get the streaming response.

        This method calls the Amazon SageMaker AI chat API and returns the stream of response events.

        Args:
            request: The formatted request to send to the Amazon SageMaker AI model.

        Returns:
            An iterable of response events from the Amazon SageMaker AI model.
        """
        if self.config.get("stream", True):
            response = self.client.invoke_endpoint_with_response_stream(**request)

            # Message start
            yield {"chunk_type": "message_start"}

            yield {"chunk_type": "content_start", "data_type": "text"}

            # Parse the content
            finish_reason = ""
            partial_content = ""
            tool_calls: dict[int, list[Any]] = {}
            for event in response["Body"]:
                chunk = event["PayloadPart"]["Bytes"].decode("utf-8")
                partial_content += chunk  # Some messages are randomly split and not JSON decodable- not sure why
                try:
                    content = json.loads(partial_content)
                    partial_content = ""
                    choice = content["choices"][0]

                    # Start yielding message chunks
                    if choice["delta"].get("content", None):
                        yield {"chunk_type": "content_delta", "data_type": "text", "data": choice["delta"]["content"]}
                    for tool_call in choice["delta"].get("tool_calls", []):
                        tool_calls.setdefault(tool_call["index"], []).append(tool_call)
                    if choice["finish_reason"] is not None:
                        finish_reason = choice["finish_reason"]
                        break

                except json.JSONDecodeError:
                    # Continue accumulating content until we have valid JSON
                    continue

            yield {"chunk_type": "content_stop", "data_type": "text"}

            # Handle tool calling
            for tool_deltas in tool_calls.values():
                yield {"chunk_type": "content_start", "data_type": "tool", "data": ToolCall(**tool_deltas[0])}
                for tool_delta in tool_deltas:
                    yield {"chunk_type": "content_delta", "data_type": "tool", "data": ToolCall(**tool_delta)}
                yield {"chunk_type": "content_stop", "data_type": "tool"}

            # Message close
            yield {"chunk_type": "message_stop", "data": finish_reason}
            # Handle usage metadata - TODO: not supported in current Response Schema!
            # Ref: https://docs.djl.ai/master/docs/serving/serving/docs/lmi/user_guides/chat_input_output_schema.html#response-schema
            # yield {"chunk_type": "metadata", "data": UsageMetadata(**choice["usage"])}

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
