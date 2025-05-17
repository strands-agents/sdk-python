"""Ollama model provider.

- Docs: https://ollama.com/
"""

import json
import logging
import uuid
from typing import Any, Iterable, Optional, TypedDict, Union

from typing_extensions import Unpack, override

import boto3
from botocore.config import Config as BotocoreConfig

from strands.types.content import Messages, Message, ContentBlock
from strands.types.media import DocumentContent, ImageContent
from strands.types.models import Model
from strands.types.streaming import StreamEvent, StopReason
from strands.types.tools import ToolSpec

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
        endpoint_name: str,
        inference_component_name: Optional[str] = None,
        boto_session: Optional[boto3.Session] = None,
        boto_client_config: Optional[BotocoreConfig] = None,
        retry_attempts: int = 3,
        retry_delay: int = 30,
        **model_config: Unpack["SageMakerAIModel.ModelConfig"],
    ):
        """Initialize provider instance.

        Args:
            endpoint_name: The name of the SageMaker endpoint to invoke.
            inference_component_name: The name of the inference component to use.
            boto_session: Boto Session to use when calling the SageMaker Runtime.
            boto_client_config: Configuration to use when creating the SageMaker-Runtime Boto Client.
            retry_attempts: Number of retry attempts for capacity errors (default: 3).
            retry_delay: Delay in seconds between retry attempts (default: 30).
            **model_config: Model parameters for the SageMaker request payload.
        """
        self.config = SageMakerAIModel.ModelConfig(
            endpoint_name=endpoint_name,
            inference_component_name=inference_component_name
        )
        self.update_config(**model_config)
        
        # Set retry configuration
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        logger.debug("endpoint=%s, config=%s | initializing", self.config["endpoint_name"], self.config)

        default_region = "us-west-2"
        session = boto_session or boto3.Session(
            region_name=default_region,
        )
        self.client = session.client(
            service_name="sagemaker-runtime",
            config=boto_client_config,
        )

    @override
    def update_config(self, **model_config: Unpack[ModelConfig]) -> None:  # type: ignore
        """Update the Ollama Model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)

    @override
    def get_config(self) -> ModelConfig:
        """Get the Ollama Model configuration.

        Returns:
            The Bedrok model configuration.
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
                            'id': content["toolUse"]["toolUseId"],
                            'type':'function',
                            "function": {
                                "name": content["toolUse"]["name"],
                                "arguments": json.dumps(content["toolUse"]["input"]),
                            }
                        }
                    ],
                }

            if "toolResult" in content:
                result_content: Union[str, ImageContent, DocumentContent, Any] = None
                result_images = []
                for toolResultContent in content["toolResult"]["content"]:
                    if "text" in toolResultContent:
                        result_content = toolResultContent["text"]
                    elif "json" in toolResultContent:
                        result_content = toolResultContent["json"]
                    elif "image" in toolResultContent:
                        result_content = "see images"
                        result_images.append(toolResultContent["image"]["source"]["bytes"])
                    else:
                        result_content = content["toolResult"]["content"]
                
                return {
                    "role": "tool",
                    "name": content["toolResult"]["toolUseId"],
                    "tool_call_id": content["toolResult"]["toolUseId"],
                    "content": json.dumps(
                        {
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

        payload = {
            "messages": [
                *([{"role": "system", "content": system_prompt}] if system_prompt else []),
                *formatted_messages,
            ],
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

        # In the payload messages, make sure no message has content empty or None. If so, replace with "Thinking ..."
        messages_new = []
        for message in payload["messages"]:
            try:
                if message["content"] is None or message["content"] == "":
                    message["content"] = "Thinking ..."
                if message['role'] == 'assistant' and message['content']=='Thinking...\n':
                    continue    
            except:
                pass
            messages_new.append(message)
        payload["messages"] = messages_new

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

        elif event["chunk_type"] == "content_start":
            if event["data_type"] == "text":
                return {"contentBlockStart": {"start": {}}}
            # Random string of 9 alphanumerical characters
            tool_id = ''.join(uuid.uuid4().hex[:9])
            tool_name = event["data"]["function"]["name"]
            return {"contentBlockStart": {"start": {"toolUse": {"name": tool_name, "toolUseId": tool_id}}}}

        elif event["chunk_type"] == "content_delta":
            if event["data_type"] == "text":
                return {"contentBlockDelta": {"delta": {"text": event["data"]}}}

            tool_arguments = event["data"]["function"]["arguments"]
            return {"contentBlockDelta": {"delta": {"toolUse": {"input": tool_arguments}}}}

        elif event["chunk_type"] == "content_stop":
            return {"contentBlockStop": {}}

        elif event["chunk_type"] == "message_stop":
            reason: StopReason
            if event["data"] == "tool_use":
                reason = "tool_use"
            elif event["data"] == "length":
                reason = "max_tokens"
            else:
                reason = "end_turn"

            return {"messageStop": {"stopReason": reason}}

        elif event["chunk_type"] == "metadata":
            return {
                "metadata": {
                    "usage": {
                        "inputTokens": event["data"]["prompt_tokens"],
                        "outputTokens": event["data"]["completion_tokens"],
                        "totalTokens": event["data"]["total_tokens"],
                    },
                    "metrics": {"latencyMs": 0},
                },
            }
        else:
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
        response = self.client.invoke_endpoint_with_response_stream(**request)

        # Message start
        yield {"chunk_type": "message_start"}
        yield {"chunk_type": "content_start", "data_type": "text"}

        # Wait until all the answer has been streamed
        final_response = ""
        for event in response["Body"]:
            chunk_data = event['PayloadPart']['Bytes'].decode("utf-8")
            final_response += chunk_data
        final_response_json = json.loads(final_response)
        
        # send messages for tool execution
        tool_requested = False
        message = final_response_json["choices"][0]["message"]
        for tool_call in message["tool_calls"] or []:
            yield {"chunk_type": "content_start", "data_type": "tool", "data": tool_call}
            yield {"chunk_type": "content_delta", "data_type": "tool", "data": tool_call}
            yield {"chunk_type": "content_stop", "data_type": "tool", "data": tool_call}
            tool_requested = True

        if message["content"] == "" or message["content"] is None:
            message["content"] = "Thinking...\n"
        yield {"chunk_type": "content_delta", "data_type": "text", "data": message["content"]}

        # Close the message
        yield {"chunk_type": "content_stop", "data_type": "text"}
        yield {"chunk_type": "message_stop", "data": "tool_use" if tool_requested else final_response_json["choices"][0]["finish_reason"]}
        yield {"chunk_type": "metadata", "data": final_response_json["usage"]}
