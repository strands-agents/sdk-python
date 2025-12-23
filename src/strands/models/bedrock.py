"""AWS Bedrock model provider.

- Docs: https://aws.amazon.com/bedrock/
"""

import asyncio
import json
import logging
import os
import warnings
from typing import Any, AsyncGenerator, Callable, Iterable, Literal, Optional, Type, TypeVar, Union, ValuesView, cast

import boto3
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError
from pydantic import BaseModel
from typing_extensions import TypedDict, Unpack, override

from .._exception_notes import add_exception_note
from ..event_loop import streaming
from ..tools import convert_pydantic_to_tool_spec
from ..tools._tool_helpers import noop_tool
from ..types.content import ContentBlock, Messages, SystemContentBlock
from ..types.exceptions import (
    ContextWindowOverflowException,
    ModelThrottledException,
)
from ..types.streaming import CitationsDelta, StreamEvent
from ..types.tools import ToolChoice, ToolSpec
from ._validation import validate_config_keys
from .model import Model

logger = logging.getLogger(__name__)

# See: `BedrockModel._get_default_model_with_warning` for why we need both
DEFAULT_BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"
_DEFAULT_BEDROCK_MODEL_ID = "{}.anthropic.claude-sonnet-4-20250514-v1:0"
DEFAULT_BEDROCK_REGION = "us-west-2"

BEDROCK_CONTEXT_WINDOW_OVERFLOW_MESSAGES = [
    "Input is too long for requested model",
    "input length and `max_tokens` exceed context limit",
    "too many total text bytes",
]

# Models that should include tool result status (include_tool_result_status = True)
_MODELS_INCLUDE_STATUS = [
    "anthropic.claude",
]

T = TypeVar("T", bound=BaseModel)

DEFAULT_READ_TIMEOUT = 120


class BedrockModel(Model):
    """AWS Bedrock model provider implementation.

    The implementation handles Bedrock-specific features such as:

    - Tool configuration for function calling
    - Guardrails integration
    - Caching points for system prompts and tools
    - Streaming responses
    - Context window overflow detection
    """

    class BedrockConfig(TypedDict, total=False):
        """Configuration options for Bedrock models.

        Attributes:
            additional_args: Any additional arguments to include in the request
            additional_request_fields: Additional fields to include in the Bedrock request
            additional_response_field_paths: Additional response field paths to extract
            cache_prompt: Cache point type for the system prompt
            cache_tools: Cache point type for tools
            guardrail_id: ID of the guardrail to apply
            guardrail_trace: Guardrail trace mode. Defaults to enabled.
            guardrail_version: Version of the guardrail to apply
            guardrail_stream_processing_mode: The guardrail processing mode
            guardrail_redact_input: Flag to redact input if a guardrail is triggered. Defaults to True.
            guardrail_redact_input_message: If a Bedrock Input guardrail triggers, replace the input with this message.
            guardrail_redact_output: Flag to redact output if guardrail is triggered. Defaults to False.
            guardrail_redact_output_message: If a Bedrock Output guardrail triggers, replace output with this message.
            max_tokens: Maximum number of tokens to generate in the response
            model_id: The Bedrock model ID (e.g., "us.anthropic.claude-sonnet-4-20250514-v1:0")
            include_tool_result_status: Flag to include status field in tool results.
                True includes status, False removes status, "auto" determines based on model_id. Defaults to "auto".
            stop_sequences: List of sequences that will stop generation when encountered
            streaming: Flag to enable/disable streaming. Defaults to True.
            temperature: Controls randomness in generation (higher = more random)
            top_p: Controls diversity via nucleus sampling (alternative to temperature)
        """

        additional_args: Optional[dict[str, Any]]
        additional_request_fields: Optional[dict[str, Any]]
        additional_response_field_paths: Optional[list[str]]
        cache_prompt: Optional[str]
        cache_tools: Optional[str]
        guardrail_id: Optional[str]
        guardrail_trace: Optional[Literal["enabled", "disabled", "enabled_full"]]
        guardrail_stream_processing_mode: Optional[Literal["sync", "async"]]
        guardrail_version: Optional[str]
        guardrail_redact_input: Optional[bool]
        guardrail_redact_input_message: Optional[str]
        guardrail_redact_output: Optional[bool]
        guardrail_redact_output_message: Optional[str]
        max_tokens: Optional[int]
        model_id: str
        include_tool_result_status: Optional[Literal["auto"] | bool]
        stop_sequences: Optional[list[str]]
        streaming: Optional[bool]
        temperature: Optional[float]
        top_p: Optional[float]

    def __init__(
        self,
        *,
        boto_session: Optional[boto3.Session] = None,
        boto_client_config: Optional[BotocoreConfig] = None,
        region_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        **model_config: Unpack[BedrockConfig],
    ):
        """Initialize provider instance.

        Args:
            boto_session: Boto Session to use when calling the Bedrock Model.
            boto_client_config: Configuration to use when creating the Bedrock-Runtime Boto Client.
            region_name: AWS region to use for the Bedrock service.
                Defaults to the AWS_REGION environment variable if set, or "us-west-2" if not set.
            endpoint_url: Custom endpoint URL for VPC endpoints (PrivateLink)
            **model_config: Configuration options for the Bedrock model.
        """
        if region_name and boto_session:
            raise ValueError("Cannot specify both `region_name` and `boto_session`.")

        session = boto_session or boto3.Session()
        resolved_region = region_name or session.region_name or os.environ.get("AWS_REGION") or DEFAULT_BEDROCK_REGION
        self.config = BedrockModel.BedrockConfig(
            model_id=BedrockModel._get_default_model_with_warning(resolved_region, model_config),
            include_tool_result_status="auto",
        )
        self.update_config(**model_config)

        logger.debug("config=<%s> | initializing", self.config)

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
            client_config = BotocoreConfig(user_agent_extra="strands-agents", read_timeout=DEFAULT_READ_TIMEOUT)

        self.client = session.client(
            service_name="bedrock-runtime",
            config=client_config,
            endpoint_url=endpoint_url,
            region_name=resolved_region,
        )

        logger.debug("region=<%s> | bedrock client created", self.client.meta.region_name)

    @override
    def update_config(self, **model_config: Unpack[BedrockConfig]) -> None:  # type: ignore
        """Update the Bedrock Model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        validate_config_keys(model_config, self.BedrockConfig)
        self.config.update(model_config)

    @override
    def get_config(self) -> BedrockConfig:
        """Get the current Bedrock Model configuration.

        Returns:
            The Bedrock model configuration.
        """
        return self.config

    def _format_request(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
        tool_choice: ToolChoice | None = None,
    ) -> dict[str, Any]:
        """Format a Bedrock converse stream request.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            tool_choice: Selection strategy for tool invocation.
            system_prompt_content: System prompt content blocks to provide context to the model.

        Returns:
            A Bedrock converse stream request.
        """
        if not tool_specs:
            has_tool_content = any(
                any("toolUse" in block or "toolResult" in block for block in msg.get("content", [])) for msg in messages
            )
            if has_tool_content:
                tool_specs = [noop_tool.tool_spec]

        # Use system_prompt_content directly (copy for mutability)
        system_blocks: list[SystemContentBlock] = system_prompt_content.copy() if system_prompt_content else []
        # Add cache point if configured (backwards compatibility)
        if cache_prompt := self.config.get("cache_prompt"):
            warnings.warn(
                "cache_prompt is deprecated. Use SystemContentBlock with cachePoint instead.", UserWarning, stacklevel=3
            )
            system_blocks.append({"cachePoint": {"type": cache_prompt}})

        return {
            "modelId": self.config["model_id"],
            "messages": self._format_bedrock_messages(messages),
            "system": system_blocks,
            **(
                {
                    "toolConfig": {
                        "tools": [
                            *[
                                {
                                    "toolSpec": {
                                        "name": tool_spec["name"],
                                        "description": tool_spec["description"],
                                        "inputSchema": tool_spec["inputSchema"],
                                    }
                                }
                                for tool_spec in tool_specs
                            ],
                            *(
                                [{"cachePoint": {"type": self.config["cache_tools"]}}]
                                if self.config.get("cache_tools")
                                else []
                            ),
                        ],
                        **({"toolChoice": tool_choice if tool_choice else {"auto": {}}}),
                    }
                }
                if tool_specs
                else {}
            ),
            **(
                {"additionalModelRequestFields": self.config["additional_request_fields"]}
                if self.config.get("additional_request_fields")
                else {}
            ),
            **(
                {"additionalModelResponseFieldPaths": self.config["additional_response_field_paths"]}
                if self.config.get("additional_response_field_paths")
                else {}
            ),
            **(
                {
                    "guardrailConfig": {
                        "guardrailIdentifier": self.config["guardrail_id"],
                        "guardrailVersion": self.config["guardrail_version"],
                        "trace": self.config.get("guardrail_trace", "enabled"),
                        **(
                            {"streamProcessingMode": self.config.get("guardrail_stream_processing_mode")}
                            if self.config.get("guardrail_stream_processing_mode")
                            else {}
                        ),
                    }
                }
                if self.config.get("guardrail_id") and self.config.get("guardrail_version")
                else {}
            ),
            "inferenceConfig": {
                key: value
                for key, value in [
                    ("maxTokens", self.config.get("max_tokens")),
                    ("temperature", self.config.get("temperature")),
                    ("topP", self.config.get("top_p")),
                    ("stopSequences", self.config.get("stop_sequences")),
                ]
                if value is not None
            },
            **(
                self.config["additional_args"]
                if "additional_args" in self.config and self.config["additional_args"] is not None
                else {}
            ),
        }

    def _format_bedrock_messages(self, messages: Messages) -> list[dict[str, Any]]:
        """Format messages for Bedrock API compatibility.

        This function ensures messages conform to Bedrock's expected format by:
        - Filtering out SDK_UNKNOWN_MEMBER content blocks
        - Eagerly filtering content blocks to only include Bedrock-supported fields
        - Ensuring all message content blocks are properly formatted for the Bedrock API

        Args:
            messages: List of messages to format

        Returns:
            Messages formatted for Bedrock API compatibility

        Note:
            Unlike other APIs that ignore unknown fields, Bedrock only accepts a strict
            subset of fields for each content block type and throws validation exceptions
            when presented with unexpected fields. Therefore, we must eagerly filter all
            content blocks to remove any additional fields before sending to Bedrock.
            https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlock.html
        """
        cleaned_messages: list[dict[str, Any]] = []

        filtered_unknown_members = False
        dropped_deepseek_reasoning_content = False

        for message in messages:
            cleaned_content: list[dict[str, Any]] = []

            for content_block in message["content"]:
                # Filter out SDK_UNKNOWN_MEMBER content blocks
                if "SDK_UNKNOWN_MEMBER" in content_block:
                    filtered_unknown_members = True
                    continue

                # DeepSeek models have issues with reasoningContent
                # TODO: Replace with systematic model configuration registry (https://github.com/strands-agents/sdk-python/issues/780)
                if "deepseek" in self.config["model_id"].lower() and "reasoningContent" in content_block:
                    dropped_deepseek_reasoning_content = True
                    continue

                # Format content blocks for Bedrock API compatibility
                formatted_content = self._format_request_message_content(content_block)
                cleaned_content.append(formatted_content)

            # Create new message with cleaned content (skip if empty)
            if cleaned_content:
                cleaned_messages.append({"content": cleaned_content, "role": message["role"]})

        if filtered_unknown_members:
            logger.warning(
                "Filtered out SDK_UNKNOWN_MEMBER content blocks from messages, consider upgrading boto3 version"
            )
        if dropped_deepseek_reasoning_content:
            logger.debug(
                "Filtered DeepSeek reasoningContent content blocks from messages - https://api-docs.deepseek.com/guides/reasoning_model#multi-round-conversation"
            )

        return cleaned_messages

    def _should_include_tool_result_status(self) -> bool:
        """Determine whether to include tool result status based on current config."""
        include_status = self.config.get("include_tool_result_status", "auto")

        if include_status is True:
            return True
        elif include_status is False:
            return False
        else:  # "auto"
            return any(model in self.config["model_id"] for model in _MODELS_INCLUDE_STATUS)

    def _format_request_message_content(self, content: ContentBlock) -> dict[str, Any]:
        """Format a Bedrock content block.

        Bedrock strictly validates content blocks and throws exceptions for unknown fields.
        This function extracts only the fields that Bedrock supports for each content type.

        Args:
            content: Content block to format.

        Returns:
            Bedrock formatted content block.

        Raises:
            TypeError: If the content block type is not supported by Bedrock.
        """
        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CachePointBlock.html
        if "cachePoint" in content:
            return {"cachePoint": {"type": content["cachePoint"]["type"]}}

        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_DocumentBlock.html
        if "document" in content:
            document = content["document"]
            result: dict[str, Any] = {}

            # Handle required fields (all optional due to total=False)
            if "name" in document:
                result["name"] = document["name"]
            if "format" in document:
                result["format"] = document["format"]

            # Handle source
            if "source" in document:
                result["source"] = {"bytes": document["source"]["bytes"]}

            # Handle optional fields
            if "citations" in document and document["citations"] is not None:
                result["citations"] = {"enabled": document["citations"]["enabled"]}
            if "context" in document:
                result["context"] = document["context"]

            return {"document": result}

        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_GuardrailConverseContentBlock.html
        if "guardContent" in content:
            guard = content["guardContent"]
            guard_text = guard["text"]
            result = {"text": {"text": guard_text["text"], "qualifiers": guard_text["qualifiers"]}}
            return {"guardContent": result}

        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ImageBlock.html
        if "image" in content:
            image = content["image"]
            source = image["source"]
            formatted_source = {}
            if "bytes" in source:
                formatted_source = {"bytes": source["bytes"]}
            result = {"format": image["format"], "source": formatted_source}
            return {"image": result}

        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ReasoningContentBlock.html
        if "reasoningContent" in content:
            reasoning = content["reasoningContent"]
            result = {}

            if "reasoningText" in reasoning:
                reasoning_text = reasoning["reasoningText"]
                result["reasoningText"] = {}
                if "text" in reasoning_text:
                    result["reasoningText"]["text"] = reasoning_text["text"]
                # Only include signature if truthy (avoid empty strings)
                if reasoning_text.get("signature"):
                    result["reasoningText"]["signature"] = reasoning_text["signature"]

            if "redactedContent" in reasoning:
                result["redactedContent"] = reasoning["redactedContent"]

            return {"reasoningContent": result}

        # Pass through text and other simple content types
        if "text" in content:
            return {"text": content["text"]}

        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolResultBlock.html
        if "toolResult" in content:
            tool_result = content["toolResult"]
            formatted_content: list[dict[str, Any]] = []
            for tool_result_content in tool_result["content"]:
                if "json" in tool_result_content:
                    # Handle json field since not in ContentBlock but valid in ToolResultContent
                    formatted_content.append({"json": tool_result_content["json"]})
                else:
                    formatted_content.append(
                        self._format_request_message_content(cast(ContentBlock, tool_result_content))
                    )

            result = {
                "content": formatted_content,
                "toolUseId": tool_result["toolUseId"],
            }
            if "status" in tool_result and self._should_include_tool_result_status():
                result["status"] = tool_result["status"]
            return {"toolResult": result}

        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolUseBlock.html
        if "toolUse" in content:
            tool_use = content["toolUse"]
            return {
                "toolUse": {
                    "input": tool_use["input"],
                    "name": tool_use["name"],
                    "toolUseId": tool_use["toolUseId"],
                }
            }

        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_VideoBlock.html
        if "video" in content:
            video = content["video"]
            source = video["source"]
            formatted_source = {}
            if "bytes" in source:
                formatted_source = {"bytes": source["bytes"]}
            result = {"format": video["format"], "source": formatted_source}
            return {"video": result}

        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CitationsContentBlock.html
        if "citationsContent" in content:
            citations = content["citationsContent"]
            result = {}

            if "citations" in citations:
                result["citations"] = []
                for citation in citations["citations"]:
                    filtered_citation: dict[str, Any] = {}
                    if "location" in citation:
                        filtered_citation["location"] = citation["location"]
                    if "sourceContent" in citation:
                        filtered_source_content: list[dict[str, Any]] = []
                        for source_content in citation["sourceContent"]:
                            if "text" in source_content:
                                filtered_source_content.append({"text": source_content["text"]})
                        if filtered_source_content:
                            filtered_citation["sourceContent"] = filtered_source_content
                    if "title" in citation:
                        filtered_citation["title"] = citation["title"]
                    result["citations"].append(filtered_citation)

            if "content" in citations:
                filtered_content: list[dict[str, Any]] = []
                for generated_content in citations["content"]:
                    if "text" in generated_content:
                        filtered_content.append({"text": generated_content["text"]})
                if filtered_content:
                    result["content"] = filtered_content

            return {"citationsContent": result}

        raise TypeError(f"content_type=<{next(iter(content))}> | unsupported type")

    def _has_blocked_guardrail(self, guardrail_data: dict[str, Any]) -> bool:
        """Check if guardrail data contains any blocked policies.

        Args:
            guardrail_data: Guardrail data from trace information.

        Returns:
            True if any blocked guardrail is detected, False otherwise.
        """
        input_assessment = guardrail_data.get("inputAssessment", {})
        output_assessments = guardrail_data.get("outputAssessments", {})

        # Check input assessments
        if any(self._find_detected_and_blocked_policy(assessment) for assessment in input_assessment.values()):
            return True

        # Check output assessments
        if any(self._find_detected_and_blocked_policy(assessment) for assessment in output_assessments.values()):
            return True

        return False

    def _generate_redaction_events(self) -> list[StreamEvent]:
        """Generate redaction events based on configuration.

        Returns:
            List of redaction events to yield.
        """
        events: list[StreamEvent] = []

        if self.config.get("guardrail_redact_input", True):
            logger.debug("Redacting user input due to guardrail.")
            events.append(
                {
                    "redactContent": {
                        "redactUserContentMessage": self.config.get(
                            "guardrail_redact_input_message", "[User input redacted.]"
                        )
                    }
                }
            )

        if self.config.get("guardrail_redact_output", False):
            logger.debug("Redacting assistant output due to guardrail.")
            events.append(
                {
                    "redactContent": {
                        "redactAssistantContentMessage": self.config.get(
                            "guardrail_redact_output_message",
                            "[Assistant output redacted.]",
                        )
                    }
                }
            )

        return events

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
        """Stream conversation with the Bedrock model.

        This method calls either the Bedrock converse_stream API or the converse API
        based on the streaming parameter in the configuration.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            tool_choice: Selection strategy for tool invocation.
            system_prompt_content: System prompt content blocks to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Model events.

        Raises:
            ContextWindowOverflowException: If the input exceeds the model's context window.
            ModelThrottledException: If the model service is throttling requests.
        """

        def callback(event: Optional[StreamEvent] = None) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)
            if event is None:
                return

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[Optional[StreamEvent]] = asyncio.Queue()

        # Handle backward compatibility: if system_prompt is provided but system_prompt_content is None
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
        """Stream conversation with the Bedrock model.

        This method operates in a separate thread to avoid blocking the async event loop with the call to
        Bedrock's converse_stream.

        Args:
            callback: Function to send events to the main thread.
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt_content: System prompt content blocks to provide context to the model.
            tool_choice: Selection strategy for tool invocation.

        Raises:
            ContextWindowOverflowException: If the input exceeds the model's context window.
            ModelThrottledException: If the model service is throttling requests.
        """
        try:
            logger.debug("formatting request")
            request = self._format_request(messages, tool_specs, system_prompt_content, tool_choice)
            logger.debug("request=<%s>", request)

            logger.debug("invoking model")
            streaming = self.config.get("streaming", True)

            logger.debug("got response from model")
            if streaming:
                response = self.client.converse_stream(**request)
                # Track tool use events to fix stopReason for streaming responses
                has_tool_use = False
                for chunk in response["stream"]:
                    if (
                        "metadata" in chunk
                        and "trace" in chunk["metadata"]
                        and "guardrail" in chunk["metadata"]["trace"]
                    ):
                        guardrail_data = chunk["metadata"]["trace"]["guardrail"]
                        if self._has_blocked_guardrail(guardrail_data):
                            for event in self._generate_redaction_events():
                                callback(event)

                    # Track if we see tool use events
                    if "contentBlockStart" in chunk and chunk["contentBlockStart"].get("start", {}).get("toolUse"):
                        has_tool_use = True

                    # Fix stopReason for streaming responses that contain tool use
                    if (
                        has_tool_use
                        and "messageStop" in chunk
                        and (message_stop := chunk["messageStop"]).get("stopReason") == "end_turn"
                    ):
                        # Create corrected chunk with tool_use stopReason
                        modified_chunk = chunk.copy()
                        modified_chunk["messageStop"] = message_stop.copy()
                        modified_chunk["messageStop"]["stopReason"] = "tool_use"
                        logger.warning("Override stop reason from end_turn to tool_use")
                        callback(modified_chunk)
                    else:
                        callback(chunk)

            else:
                response = self.client.converse(**request)
                for event in self._convert_non_streaming_to_streaming(response):
                    callback(event)

                if (
                    "trace" in response
                    and "guardrail" in response["trace"]
                    and self._has_blocked_guardrail(response["trace"]["guardrail"])
                ):
                    for event in self._generate_redaction_events():
                        callback(event)

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

            # Aid in debugging by adding more information
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

            if (
                e.response["Error"]["Code"] == "ValidationException"
                and "with on-demand throughput isn’t supported" in error_message
            ):
                add_exception_note(
                    e,
                    "└ For more information see "
                    "https://strandsagents.com/latest/user-guide/concepts/model-providers/amazon-bedrock/#on-demand-throughput-isnt-supported",
                )

            raise e

        finally:
            callback()
            logger.debug("finished streaming response from model")

    def _convert_non_streaming_to_streaming(self, response: dict[str, Any]) -> Iterable[StreamEvent]:
        """Convert a non-streaming response to the streaming format.

        Args:
            response: The non-streaming response from the Bedrock model.

        Returns:
            An iterable of response events in the streaming format.
        """
        # Yield messageStart event
        yield {"messageStart": {"role": response["output"]["message"]["role"]}}

        # Process content blocks
        for content in cast(list[ContentBlock], response["output"]["message"]["content"]):
            # Yield contentBlockStart event if needed
            if "toolUse" in content:
                yield {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "toolUseId": content["toolUse"]["toolUseId"],
                                "name": content["toolUse"]["name"],
                            }
                        },
                    }
                }

                # For tool use, we need to yield the input as a delta
                input_value = json.dumps(content["toolUse"]["input"])

                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": input_value}}}}
            elif "text" in content:
                # Then yield the text as a delta
                yield {
                    "contentBlockDelta": {
                        "delta": {"text": content["text"]},
                    }
                }
            elif "reasoningContent" in content:
                # Then yield the reasoning content as a delta
                yield {
                    "contentBlockDelta": {
                        "delta": {"reasoningContent": {"text": content["reasoningContent"]["reasoningText"]["text"]}}
                    }
                }

                if "signature" in content["reasoningContent"]["reasoningText"]:
                    yield {
                        "contentBlockDelta": {
                            "delta": {
                                "reasoningContent": {
                                    "signature": content["reasoningContent"]["reasoningText"]["signature"]
                                }
                            }
                        }
                    }
            elif "citationsContent" in content:
                # For non-streaming citations, emit text and metadata deltas in sequence
                # to match streaming behavior where they flow naturally
                if "content" in content["citationsContent"]:
                    text_content = "".join([content["text"] for content in content["citationsContent"]["content"]])
                    yield {
                        "contentBlockDelta": {"delta": {"text": text_content}},
                    }

                for citation in content["citationsContent"]["citations"]:
                    # Then emit citation metadata (for structure)

                    citation_metadata: CitationsDelta = {
                        "title": citation["title"],
                        "location": citation["location"],
                        "sourceContent": citation["sourceContent"],
                    }
                    yield {"contentBlockDelta": {"delta": {"citation": citation_metadata}}}

            # Yield contentBlockStop event
            yield {"contentBlockStop": {}}

        # Yield messageStop event
        # Fix stopReason for models that return end_turn when they should return tool_use on non-streaming side
        current_stop_reason = response["stopReason"]
        if current_stop_reason == "end_turn":
            message_content = response["output"]["message"]["content"]
            if any("toolUse" in content for content in message_content):
                current_stop_reason = "tool_use"
                logger.warning("Override stop reason from end_turn to tool_use")

        yield {
            "messageStop": {
                "stopReason": current_stop_reason,
                "additionalModelResponseFields": response.get("additionalModelResponseFields"),
            }
        }

        # Yield metadata event
        if "usage" in response or "metrics" in response or "trace" in response:
            metadata: StreamEvent = {"metadata": {}}
            if "usage" in response:
                metadata["metadata"]["usage"] = response["usage"]
            if "metrics" in response:
                metadata["metadata"]["metrics"] = response["metrics"]
            if "trace" in response:
                metadata["metadata"]["trace"] = response["trace"]
            yield metadata

    def _find_detected_and_blocked_policy(self, input: Any) -> bool:
        """Recursively checks if the assessment contains a detected and blocked guardrail.

        Args:
            input: The assessment to check.

        Returns:
            True if the input contains a detected and blocked guardrail, False otherwise.

        """
        # Check if input is a dictionary
        if isinstance(input, dict):
            # Check if current dictionary has action: BLOCKED and detected: true
            if input.get("action") == "BLOCKED" and input.get("detected") and isinstance(input.get("detected"), bool):
                return True

            # Otherwise, recursively check all values in the dictionary
            return self._find_detected_and_blocked_policy(input.values())

        elif isinstance(input, (list, ValuesView)):
            # Handle case where input is a list or dict_values
            return any(self._find_detected_and_blocked_policy(item) for item in input)
        # Otherwise return False
        return False

    @override
    async def structured_output(
        self,
        output_model: Type[T],
        prompt: Messages,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Union[T, Any]], None]:
        """Get structured output from the model.

        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Model events with the last being the structured output.
        """
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
        output_response: dict[str, Any] | None = None
        for block in content:
            # if the tool use name doesn't match the tool spec name, skip, and if the block is not a tool use, skip.
            # if the tool use name never matches, raise an error.
            if block.get("toolUse") and block["toolUse"]["name"] == tool_spec["name"]:
                output_response = block["toolUse"]["input"]
            else:
                continue

        if output_response is None:
            raise ValueError("No valid tool use or tool use input was found in the Bedrock response.")

        yield {"output": output_model(**output_response)}

    @staticmethod
    def _get_default_model_with_warning(region_name: str, model_config: Optional[BedrockConfig] = None) -> str:
        """Get the default Bedrock modelId based on region.

        If the region is not **known** to support inference then we show a helpful warning
        that compliments the exception that Bedrock will throw.
        If the customer provided a model_id in their config or they overrode the `DEFAULT_BEDROCK_MODEL_ID`
        then we should not process further.

        Args:
            region_name (str): region for bedrock model
            model_config (Optional[dict[str, Any]]): Model Config that caller passes in on init
        """
        if DEFAULT_BEDROCK_MODEL_ID != _DEFAULT_BEDROCK_MODEL_ID.format("us"):
            return DEFAULT_BEDROCK_MODEL_ID

        model_config = model_config or {}
        if model_config.get("model_id"):
            return model_config["model_id"]

        prefix_inference_map = {"ap": "apac"}  # some inference endpoints can be a bit different than the region prefix

        prefix = "-".join(region_name.split("-")[:-2]).lower()  # handles `us-east-1` or `us-gov-east-1`
        if prefix not in {"us", "eu", "ap", "us-gov"}:
            warnings.warn(
                f"""
            ================== WARNING ==================

                This region {region_name} does not support
                our default inference endpoint: {_DEFAULT_BEDROCK_MODEL_ID.format(prefix)}.
                Update the agent to pass in a 'model_id' like so:
                ```
                Agent(..., model='valid_model_id', ...)
                ````
                Documentation: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html

            ==================================================
            """,
                stacklevel=2,
            )

        return _DEFAULT_BEDROCK_MODEL_ID.format(prefix_inference_map.get(prefix, prefix))
