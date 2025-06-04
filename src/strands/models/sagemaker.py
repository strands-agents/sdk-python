"""Amazon SageMaker model provider."""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional, TypedDict, cast

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

    def __init__(self, **kwargs):
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

    def __init__(self, **kwargs):
        """Initialize tool call object.

        Args:
            **kwargs: Keyword arguments for the tool call.
        """
        self.id = kwargs.get("id")
        self.type = kwargs.get("type")
        self.function = FunctionCall(**kwargs.get("function"))


class SageMakerAIModel(OpenAIModel):
    """Amazon SageMaker model provider implementation.

    The implementation handles SageMaker-specific features such as:

    - Endpoint invocation
    - Tool configuration for function calling
    - Context window overflow detection
    - Endpoint not found error handling
    - Inference component capacity error handling with automatic retries
    """

    class SageMakerAIModelConfig(TypedDict, total=False):
        """Configuration options for SageMaker models.

        Attributes:
            endpoint_name: The name of the SageMaker endpoint to invoke
            inference_component_name: The name of the inference component to use
            max_tokens: Maximum number of tokens to generate in the response
            stop_sequences: List of sequences that will stop generation when encountered
            temperature: Controls randomness in generation (higher = more random)
            top_p: Controls diversity via nucleus sampling (alternative to temperature)
            additional_args: Any additional arguments to include in the request
        """

        endpoint_name: str
        inference_component_name: Optional[str]
        max_tokens: Optional[int]
        stop_sequences: Optional[list[str]]
        temperature: Optional[float]
        top_p: Optional[float]
        additional_args: Optional[dict[str, Any]]

    def __init__(
        self,
        *,
        boto_session: Optional[boto3.Session] = None,
        boto_client_config: Optional[BotocoreConfig] = None,
        region_name: Optional[str] = None,
        **model_config: Unpack["SageMakerAIModelConfig"],
    ):
        """Initialize provider instance.

        Args:
            boto_session: Boto Session to use when calling the SageMaker Runtime.
            boto_client_config: Configuration to use when creating the SageMaker-Runtime Boto Client.
            region_name: Name of the AWS region (e.g.: us-west-2)
            **model_config: Model parameters for the SageMaker request payload.
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
            **({"max_tokens": self.config["max_tokens"]} if "max_tokens" in self.config else {}),
            **({"temperature": self.config["temperature"]} if "temperature" in self.config else {}),
            **({"top_p": self.config["top_p"]} if "top_p" in self.config else {}),
            **({"stop": self.config["stop_sequences"]} if "stop_sequences" in self.config else {}),
            **(
                self.config["additional_args"]
                if "additional_args" in self.config and self.config["additional_args"] is not None
                else {}
            ),
        }

        # Assistant message must have either content or tool_calls, but not both
        for message in payload["messages"]:
            if message.get("tool_calls", []) != []:
                _ = message.pop("content")

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
        response = self.client.invoke_endpoint_with_response_stream(**request)

        # Wait until all the answer has been streamed
        final_response = ""
        for event in response["Body"]:
            chunk_data = event["PayloadPart"]["Bytes"].decode("utf-8")
            final_response += chunk_data
        final_response_json = json.loads(final_response)

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
