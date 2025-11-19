"""AWS Bedrock model provider using InvokeModel APIs.

This implementation uses InvokeModel and InvokeModelWithResponseStream APIs
instead of Converse/ConverseStream for models that don't support the latter.
"""

import asyncio
import json
import logging
import os
from typing import Any, AsyncGenerator, Callable, Optional, Type, TypeVar, Union, cast

import boto3
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError
from pydantic import BaseModel
from typing_extensions import TypedDict, Unpack, override

from .._exception_notes import add_exception_note
from ..event_loop import streaming
from ..tools import convert_pydantic_to_tool_spec
from ..types.content import Messages, SystemContentBlock
from ..types.exceptions import (
    ContextWindowOverflowException,
    ModelThrottledException,
)
from ..types.streaming import StreamEvent
from ..types.tools import ToolChoice, ToolSpec
from ._validation import validate_config_keys
from .model import Model

logger = logging.getLogger(__name__)

DEFAULT_BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
DEFAULT_BEDROCK_REGION = "us-west-2"
DEFAULT_READ_TIMEOUT = 120

BEDROCK_CONTEXT_WINDOW_OVERFLOW_MESSAGES = [
    "Input is too long for requested model",
    "input length and `max_tokens` exceed context limit",
    "too many total text bytes",
]

T = TypeVar("T", bound=BaseModel)


class BedrockModelInvoke(Model):
    """AWS Bedrock model provider using InvokeModel APIs.

    This implementation uses the native InvokeModel and InvokeModelWithResponseStream
    APIs instead of Converse/ConverseStream for models that don't support the latter.
    """

    class BedrockInvokeConfig(TypedDict, total=False):
        """Configuration options for Bedrock InvokeModel."""

        guardrail_id: Optional[str]
        guardrail_version: Optional[str]
        max_tokens: Optional[int]
        model_id: str
        streaming: Optional[bool]
        temperature: Optional[float]
        top_p: Optional[float]
        top_k: Optional[int]
        stop_sequences: Optional[list[str]]

    def __init__(
        self,
        *,
        boto_session: Optional[boto3.Session] = None,
        boto_client_config: Optional[BotocoreConfig] = None,
        region_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        **model_config: Unpack[BedrockInvokeConfig],
    ):
        """Initialize provider instance."""
        if region_name and boto_session:
            raise ValueError("Cannot specify both `region_name` and `boto_session`.")

        session = boto_session or boto3.Session()
        resolved_region = region_name or session.region_name or os.environ.get("AWS_REGION") or DEFAULT_BEDROCK_REGION

        self.config = BedrockModelInvoke.BedrockInvokeConfig(
            model_id=model_config.get("model_id", DEFAULT_BEDROCK_MODEL_ID),
            streaming=True,
        )
        self.update_config(**model_config)

        logger.debug("config=<%s> | initializing", self.config)

        if boto_client_config:
            existing_user_agent = getattr(boto_client_config, "user_agent_extra", None)
            if existing_user_agent:
                new_user_agent = f"{existing_user_agent} strands-agents"
            else:
                new_user_agent = "strands-agents"
            client_config = boto_client_config.merge(BotocoreConfig(user_agent_extra=new_user_agent))
        else:
            client_config = BotocoreConfig(user_agent_extra="strands-agents", read_timeout=DEFAULT_READ_TIMEOUT)

        self.client = session.client(
            service_name="bedrock-runtime",
            config=client_config,
            endpoint_url=endpoint_url,
            region_name=resolved_region,
        )

        logger.debug("region=<%s> | bedrock client created", self.client.meta.region_name)

    @override
    def update_config(self, **model_config: Unpack[BedrockInvokeConfig]) -> None:
        """Update the Bedrock Model configuration."""
        validate_config_keys(model_config, self.BedrockInvokeConfig)
        self.config.update(model_config)

    @override
    def get_config(self) -> BedrockInvokeConfig:
        """Get the current Bedrock Model configuration."""
        return self.config

    def _get_model_family(self) -> str:
        """Determine the model family from model ID."""
        model_id = self.config["model_id"].lower()
        if "anthropic" in model_id or "claude" in model_id:
            return "anthropic"
        elif "amazon" in model_id or "titan" in model_id:
            return "amazon"
        elif "meta" in model_id or "llama" in model_id:
            return "meta"
        elif "cohere" in model_id:
            return "cohere"
        elif "mistral" in model_id:
            return "mistral"
        else:
            # Default to anthropic for imported models or unknown ARNs
            return "anthropic"

    def _format_anthropic_request(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
        tool_choice: ToolChoice | None = None,
    ) -> dict[str, Any]:
        """Format request for Anthropic Claude models."""
        request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.config.get("max_tokens", 4096),
            "messages": [],
        }

        # Add system prompt
        if system_prompt_content:
            system_text = " ".join(block.get("text", "") for block in system_prompt_content if "text" in block)
            if system_text:
                request["system"] = system_text

        # Convert messages
        for msg in messages:
            role = msg["role"]
            content = []

            for block in msg["content"]:
                if "text" in block:
                    content.append({"type": "text", "text": block["text"]})
                elif "image" in block:
                    image_data = block["image"]
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image_data["format"],
                                "data": image_data["source"]["bytes"],
                            },
                        }
                    )
                elif "toolUse" in block:
                    tool_use = block["toolUse"]
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tool_use["toolUseId"],
                            "name": tool_use["name"],
                            "input": tool_use["input"],
                        }
                    )
                elif "toolResult" in block:
                    tool_result = block["toolResult"]
                    result_content = []
                    for result_block in tool_result["content"]:
                        if "text" in result_block:
                            result_content.append({"type": "text", "text": result_block["text"]})

                    content.append(
                        {"type": "tool_result", "tool_use_id": tool_result["toolUseId"], "content": result_content}
                    )

            if content:
                request["messages"].append({"role": role, "content": content})

        # Add tools
        if tool_specs:
            request["tools"] = []
            for tool_spec in tool_specs:
                request["tools"].append(
                    {
                        "name": tool_spec["name"],
                        "description": tool_spec["description"],
                        "input_schema": tool_spec["inputSchema"],
                    }
                )

        # Add inference parameters
        if self.config.get("temperature") is not None:
            request["temperature"] = self.config["temperature"]
        if self.config.get("top_p") is not None:
            request["top_p"] = self.config["top_p"]
        if self.config.get("top_k") is not None:
            request["top_k"] = self.config["top_k"]
        if self.config.get("stop_sequences"):
            request["stop_sequences"] = self.config["stop_sequences"]

        return request

    def _format_openai_request(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
        tool_choice: ToolChoice | None = None,
    ) -> dict[str, Any]:
        """Format request for OpenAI-compatible models."""
        request = {
            "model": self.config["model_id"],
            "messages": [],
            "max_tokens": self.config.get("max_tokens", 4096),
            "stream": self.config.get("streaming", True),
        }

        # Add system message
        if system_prompt_content:
            system_text = " ".join(block.get("text", "") for block in system_prompt_content if "text" in block)
            if system_text:
                request["messages"].append({"role": "system", "content": system_text})

        # Convert messages
        for msg in messages:
            role = msg["role"]
            content = ""

            for block in msg["content"]:
                if "text" in block:
                    content += block["text"]

            if content:
                request["messages"].append({"role": role, "content": content})

        # Add tools
        if tool_specs:
            request["tools"] = []
            for tool_spec in tool_specs:
                request["tools"].append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool_spec["name"],
                            "description": tool_spec["description"],
                            "parameters": tool_spec["inputSchema"],
                        },
                    }
                )

        # Add inference parameters
        if self.config.get("temperature") is not None:
            request["temperature"] = self.config["temperature"]
        if self.config.get("top_p") is not None:
            request["top_p"] = self.config["top_p"]
        if self.config.get("stop_sequences"):
            request["stop"] = self.config["stop_sequences"]

        return request

    def _format_request(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
        tool_choice: ToolChoice | None = None,
    ) -> dict[str, Any]:
        """Format request based on model family."""
        # Try OpenAI format first for imported models
        return self._format_openai_request(messages, tool_specs, system_prompt_content, tool_choice)

    def _parse_anthropic_response(self, response: dict[str, Any]) -> list[StreamEvent]:
        """Parse response into StreamEvent format."""
        events = []

        # Start message
        events.append({"messageStart": {"role": "assistant"}})

        # Extract text from any possible field
        text_content = self._extract_text_from_response(response)

        # Add text content if found
        if text_content:
            events.append({"contentBlockDelta": {"delta": {"text": text_content}}})
            events.append({"contentBlockStop": {}})

        # End message
        events.append({"messageStop": {"stopReason": "end_turn"}})

        return events

    def _extract_text_from_response(self, response: dict[str, Any]) -> str:
        """Extract text content from any response format."""
        # Try common text fields
        for field in ["completion", "outputText", "text", "response", "output"]:
            if field in response and isinstance(response[field], str):
                return response[field]

        # Try Anthropic content format
        if "content" in response and isinstance(response["content"], list):
            text_parts = []
            for block in response["content"]:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block["text"])
                elif isinstance(block, str):
                    text_parts.append(block)
            if text_parts:
                return "".join(text_parts)

        # Fallback: convert entire response to string if no text found
        return json.dumps(response)

    def _parse_anthropic_streaming_chunk(self, chunk: dict[str, Any]) -> Optional[StreamEvent]:
        """Parse a single streaming chunk."""
        # Handle OpenAI-style chat completion chunks
        if "choices" in chunk and chunk["choices"]:
            choice = chunk["choices"][0]
            delta = choice.get("delta", {})

            if "content" in delta and delta["content"]:
                return {"contentBlockDelta": {"delta": {"text": delta["content"]}}}

        # Standard Anthropic format
        chunk_type = chunk.get("type")
        if chunk_type == "content_block_delta":
            delta = chunk.get("delta", {})
            if "text" in delta:
                return {"contentBlockDelta": {"delta": {"text": delta["text"]}}}

        return None

    def _extract_usage_from_response(
        self, response: dict[str, Any], response_body: dict[str, Any] = None
    ) -> Optional[dict[str, Any]]:
        """Extract usage information from response body."""
        usage = {}

        # Check response body for usage info (model-specific formats)
        if response_body:
            # Anthropic format
            if "usage" in response_body:
                body_usage = response_body["usage"]
                if "input_tokens" in body_usage:
                    usage["inputTokens"] = body_usage["input_tokens"]
                if "output_tokens" in body_usage:
                    usage["outputTokens"] = body_usage["output_tokens"]

            # OpenAI format (for imported models)
            elif "usage" in response_body:
                body_usage = response_body["usage"]
                if "prompt_tokens" in body_usage:
                    usage["inputTokens"] = body_usage["prompt_tokens"]
                if "completion_tokens" in body_usage:
                    usage["outputTokens"] = body_usage["completion_tokens"]
                if "total_tokens" in body_usage:
                    usage["totalTokens"] = body_usage["total_tokens"]

        # Calculate total tokens if not provided
        if "inputTokens" in usage and "outputTokens" in usage and "totalTokens" not in usage:
            usage["totalTokens"] = usage["inputTokens"] + usage["outputTokens"]

        return usage if usage else None

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with the Bedrock model using InvokeModel APIs."""

        def callback(event: Optional[StreamEvent] = None) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[Optional[StreamEvent]] = asyncio.Queue()

        # Handle backward compatibility
        if system_prompt and system_prompt_content is None:
            system_prompt_content = [{"text": system_prompt}]

        thread = asyncio.to_thread(self._stream, callback, messages, tool_specs, system_prompt_content, tool_choice)
        task = asyncio.create_task(thread)

        while True:
            event = await queue.get()
            if event is None:
                break
            yield event

        await task

    def _stream(
        self,
        callback: Callable[..., None],
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
        tool_choice: ToolChoice | None = None,
    ) -> None:
        """Stream conversation in separate thread."""
        try:
            logger.debug("formatting request")
            request_body = self._format_request(messages, tool_specs, system_prompt_content, tool_choice)
            logger.debug("request_body=<%s>", request_body)

            streaming = self.config.get("streaming", True)

            if streaming:
                logger.debug("invoking model with streaming")
                response = self.client.invoke_model_with_response_stream(
                    modelId=self.config["model_id"],
                    body=json.dumps(request_body),
                    contentType="application/json",
                    accept="application/json",
                )

                # Send messageStart event first
                callback({"messageStart": {"role": "assistant"}})
                callback({"contentBlockStart": {"start": {"text": {}}}})

                for event in response["body"]:
                    chunk = json.loads(event["chunk"]["bytes"])
                    logger.debug("streaming_chunk=<%s>", chunk)
                    parsed_event = self._parse_anthropic_streaming_chunk(chunk)
                    if parsed_event:
                        callback(parsed_event)

                # Send end events
                callback({"contentBlockStop": {}})
                callback({"messageStop": {"stopReason": "end_turn"}})

                # Usage info not available in streaming responses for InvokeModel API
            else:
                logger.debug("invoking model without streaming")
                response = self.client.invoke_model(
                    modelId=self.config["model_id"],
                    body=json.dumps(request_body),
                    contentType="application/json",
                    accept="application/json",
                )

                response_body = json.loads(response["body"].read())
                logger.debug("response_body=<%s>", response_body)
                events = self._parse_anthropic_response(response_body)
                for event in events:
                    callback(event)

                # Extract usage from response
                usage = self._extract_usage_from_response(response, response_body)
                if usage:
                    callback({"metadata": {"usage": usage}})

        except ClientError as e:
            error_message = str(e)

            if (
                e.response["Error"]["Code"] == "ThrottlingException"
                or e.response["Error"]["Code"] == "throttlingException"
            ):
                raise ModelThrottledException(error_message) from e

            if any(overflow_message in error_message for overflow_message in BEDROCK_CONTEXT_WINDOW_OVERFLOW_MESSAGES):
                logger.warning("bedrock threw context window overflow error")
                raise ContextWindowOverflowException(e) from e

            region = self.client.meta.region_name
            add_exception_note(e, f"└ Bedrock region: {region}")
            add_exception_note(e, f"└ Model id: {self.config.get('model_id')}")

            if (
                e.response["Error"]["Code"] == "AccessDeniedException"
                and "You don't have access to the model" in error_message
            ):
                add_exception_note(
                    e,
                    "└ For more information see "
                    "https://strandsagents.com/latest/user-guide/concepts/model-providers/amazon-bedrock/#model-access-issue",
                )

            raise e

        finally:
            callback()
            logger.debug("finished streaming response from model")

    @override
    async def structured_output(
        self,
        output_model: Type[T],
        prompt: Messages,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Union[T, Any]], None]:
        """Get structured output from the model."""
        tool_spec = convert_pydantic_to_tool_spec(output_model)

        response = self.stream(
            messages=prompt,
            tool_specs=[tool_spec],
            system_prompt=system_prompt,
            tool_choice=cast(ToolChoice, {"any": {}}),
            **kwargs,
        )
        async for event in streaming.process_stream(response):
            yield event

        stop_reason, messages, _, _ = event["stop"]

        if stop_reason != "tool_use":
            raise ValueError(f'Model returned stop_reason: {stop_reason} instead of "tool_use".')

        content = messages["content"]
        for block in content:
            if block.get("toolUse") and block["toolUse"]["name"] == tool_spec["name"]:
                yield {"structured_output": output_model(**block["toolUse"]["input"])}
                return

        raise ValueError(f"No tool use found for {tool_spec['name']}")
