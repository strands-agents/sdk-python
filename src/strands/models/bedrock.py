"""AWS Bedrock model provider.

- Docs: https://aws.amazon.com/bedrock/
"""

import json
import logging
import os
from typing import Any, Iterable, List, Literal, Optional, cast

import boto3
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError
from typing_extensions import TypedDict, Unpack, override

from ..types.content import Messages
from ..types.exceptions import ContextWindowOverflowException, ModelThrottledException
from ..types.models import Model
from ..types.streaming import StreamEvent
from ..types.tools import ToolSpec

logger = logging.getLogger(__name__)

DEFAULT_BEDROCK_MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

BEDROCK_CONTEXT_WINDOW_OVERFLOW_MESSAGES = [
    "Input is too long for requested model",
    "input length and `max_tokens` exceed context limit",
    "too many total text bytes",
]


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
            model_id: The Bedrock model ID (e.g., "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
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
        **model_config: Unpack[BedrockConfig],
    ):
        """Initialize provider instance.

        Args:
            boto_session: Boto Session to use when calling the Bedrock Model.
            boto_client_config: Configuration to use when creating the Bedrock-Runtime Boto Client.
            region_name: AWS region to use for the Bedrock service.
                Defaults to the AWS_REGION environment variable if set, or "us-west-2" if not set.
            **model_config: Configuration options for the Bedrock model.
        """
        if region_name and boto_session:
            raise ValueError("Cannot specify both `region_name` and `boto_session`.")

        self.config = BedrockModel.BedrockConfig(model_id=DEFAULT_BEDROCK_MODEL_ID)
        self.update_config(**model_config)

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
            service_name="bedrock-runtime",
            config=client_config,
        )

    @override
    def update_config(self, **model_config: Unpack[BedrockConfig]) -> None:  # type: ignore
        """Update the Bedrock Model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)

    @override
    def get_config(self) -> BedrockConfig:
        """Get the current Bedrock Model configuration.

        Returns:
            The Bedrock model configuration.
        """
        return self.config

    @override
    def format_request(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Format a Bedrock converse stream request.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            A Bedrock converse stream request.
        """
        return {
            "modelId": self.config["model_id"],
            "messages": messages,
            "system": [
                *([{"text": system_prompt}] if system_prompt else []),
                *([{"cachePoint": {"type": self.config["cache_prompt"]}}] if self.config.get("cache_prompt") else []),
            ],
            **(
                {
                    "toolConfig": {
                        "tools": [
                            *[{"toolSpec": tool_spec} for tool_spec in tool_specs],
                            *(
                                [{"cachePoint": {"type": self.config["cache_tools"]}}]
                                if self.config.get("cache_tools")
                                else []
                            ),
                        ],
                        "toolChoice": {"auto": {}},
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

    @override
    def format_chunk(self, event: dict[str, Any]) -> StreamEvent:
        """Format the Bedrock response events into standardized message chunks.

        Args:
            event: A response event from the Bedrock model.

        Returns:
            The formatted chunk.
        """
        return cast(StreamEvent, event)

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
        events: List[StreamEvent] = []

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
                            "guardrail_redact_output_message", "[Assistant output redacted.]"
                        )
                    }
                }
            )

        return events

    @override
    def stream(self, request: dict[str, Any]) -> Iterable[StreamEvent]:
        """Send the request to the Bedrock model and get the response.

        This method calls either the Bedrock converse_stream API or the converse API
        based on the streaming parameter in the configuration.

        Args:
            request: The formatted request to send to the Bedrock model

        Returns:
            An iterable of response events from the Bedrock model

        Raises:
            ContextWindowOverflowException: If the input exceeds the model's context window.
            ModelThrottledException: If the model service is throttling requests.
        """
        streaming = self.config.get("streaming", True)

        try:
            if streaming:
                # Streaming implementation
                response = self.client.converse_stream(**request)
                for chunk in response["stream"]:
                    if (
                        "metadata" in chunk
                        and "trace" in chunk["metadata"]
                        and "guardrail" in chunk["metadata"]["trace"]
                    ):
                        guardrail_data = chunk["metadata"]["trace"]["guardrail"]
                        if self._has_blocked_guardrail(guardrail_data):
                            yield from self._generate_redaction_events()
                    yield chunk
            else:
                # Non-streaming implementation
                response = self.client.converse(**request)

                # Convert and yield from the response
                yield from self._convert_non_streaming_to_streaming(response)

                # Check for guardrail triggers after yielding any events (same as streaming path)
                if (
                    "trace" in response
                    and "guardrail" in response["trace"]
                    and self._has_blocked_guardrail(response["trace"]["guardrail"])
                ):
                    yield from self._generate_redaction_events()

        except ClientError as e:
            error_message = str(e)

            # Handle throttling error
            if e.response["Error"]["Code"] == "ThrottlingException":
                raise ModelThrottledException(error_message) from e

            # Handle context window overflow
            if any(overflow_message in error_message for overflow_message in BEDROCK_CONTEXT_WINDOW_OVERFLOW_MESSAGES):
                logger.warning("bedrock threw context window overflow error")
                raise ContextWindowOverflowException(e) from e

            # Otherwise raise the error
            raise e

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
        for content in response["output"]["message"]["content"]:
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

            # Yield contentBlockStop event
            yield {"contentBlockStop": {}}

        # Yield messageStop event
        yield {
            "messageStop": {
                "stopReason": response["stopReason"],
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

            # Recursively check all values in the dictionary
            for value in input.values():
                if isinstance(value, dict):
                    return self._find_detected_and_blocked_policy(value)
                # Handle case where value is a list of dictionaries
                elif isinstance(value, list):
                    for item in value:
                        return self._find_detected_and_blocked_policy(item)
        elif isinstance(input, list):
            # Handle case where input is a list of dictionaries
            for item in input:
                return self._find_detected_and_blocked_policy(item)
        # Otherwise return False
        return False
