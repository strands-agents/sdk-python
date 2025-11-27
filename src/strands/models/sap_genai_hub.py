"""SAP GenAI Hub model provider.

- Docs: https://help.sap.com/docs/sap-ai-core/sap-ai-core-service-guide/consume-generative-ai-models-using-sap-ai-core#aws-bedrock
- SDK Reference: https://help.sap.com/doc/sap-ai-sdk-gen/CLOUD/en-US/_reference/gen_ai_hub.html
"""

import asyncio
import json
import logging
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Iterable,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

from pydantic import BaseModel
from typing_extensions import TypedDict, Unpack, override

from ..types.content import Messages, SystemContentBlock
from ..types.exceptions import (
    ContextWindowOverflowException,
    ModelThrottledException,
)
from ..types.streaming import StreamEvent
from ..types.tools import ToolChoice, ToolSpec
from ._validation import validate_config_keys
from .model import Model

# Import SAP GenAI Hub SDK
try:
    from gen_ai_hub.proxy.native.amazon.clients import Session
except ImportError as e:
    raise ImportError(
        "SAP GenAI Hub SDK is not installed. Please install it with: pip install 'sap-ai-sdk-gen[all]'"
    ) from e

logger = logging.getLogger(__name__)

DEFAULT_SAP_GENAI_HUB_MODEL_ID = "amazon--nova-lite"

# Common error messages for context window overflow
CONTEXT_WINDOW_OVERFLOW_MESSAGES = [
    "Input is too long for requested model",
    "input length and `max_tokens` exceed context limit",
    "too many total text bytes",
]

T = TypeVar("T", bound=BaseModel)


class SAPGenAIHubModel(Model):
    """SAP GenAI Hub model provider implementation.

    This implementation handles SAP GenAI Hub-specific features such as:
    - Tool configuration for function calling
    - Streaming responses
    - Context window overflow detection
    - Support for different model types (Nova, Claude, Titan)
    """

    class SAPGenAIHubConfig(TypedDict, total=False):
        """Configuration options for SAP GenAI Hub models.

        Attributes:
            additional_args: Any additional arguments to include in the request
            max_tokens: Maximum number of tokens to generate in the response
            model_id: The SAP GenAI Hub model ID (e.g., "amazon--nova-lite", "anthropic--claude-3-sonnet")
            stop_sequences: List of sequences that will stop generation when encountered
            streaming: Flag to enable/disable streaming. Defaults to True.
            temperature: Controls randomness in generation (higher = more random)
            top_p: Controls diversity via nucleus sampling (alternative to temperature)
        """

        additional_args: Optional[dict[str, Any]]
        max_tokens: Optional[int]
        model_id: str
        stop_sequences: Optional[list[str]]
        streaming: Optional[bool]
        temperature: Optional[float]
        top_p: Optional[float]

    def __init__(
        self,
        **model_config: Unpack[SAPGenAIHubConfig],
    ):
        """Initialize provider instance.

        Args:
            **model_config: Configuration options for the SAP GenAI Hub model.
        """
        self.config = SAPGenAIHubModel.SAPGenAIHubConfig(
            model_id=DEFAULT_SAP_GENAI_HUB_MODEL_ID
        )
        self.update_config(**model_config)

        logger.debug("config=<%s> | initializing", self.config)

        # Initialize the SAP GenAI Hub client
        self.client = Session().client(model_name=self.config["model_id"])

    @override
    def update_config(self, **model_config: Unpack[SAPGenAIHubConfig]) -> None:  # type: ignore
        """Update the SAP GenAI Hub Model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        validate_config_keys(model_config, self.SAPGenAIHubConfig)
        self.config.update(model_config)

    @override
    def get_config(self) -> SAPGenAIHubConfig:
        """Get the current SAP GenAI Hub Model configuration.

        Returns:
            The SAP GenAI Hub model configuration.
        """
        return self.config

    def _is_nova_model(self) -> bool:
        """Check if the current model is an Amazon Nova model.

        Returns:
            True if the model is an Amazon Nova model, False otherwise.
        """
        nova_models = ["amazon--nova-pro", "amazon--nova-micro", "amazon--nova-lite"]
        return self.config["model_id"] in nova_models

    def _is_claude_model(self) -> bool:
        """Check if the current model is an Anthropic Claude model.

        Returns:
            True if the model is an Anthropic Claude model, False otherwise.
        """
        return self.config["model_id"].startswith("anthropic--claude")

    def _is_titan_embed_model(self) -> bool:
        """Check if the current model is an Amazon Titan Embedding model.

        Returns:
            True if the model is an Amazon Titan Embedding model, False otherwise.
        """
        return self.config["model_id"] == "amazon--titan-embed-text"

    def _format_request(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
        tool_choice: ToolChoice | None = None,
    ) -> dict[str, Any]:
        """Format a request for the SAP GenAI Hub model.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt_content: System prompt content blocks to provide context to the model.
            tool_choice: Selection strategy for tool invocation.

        Returns:
            A formatted request for the SAP GenAI Hub model.
        """
        # Format request based on model type
        if self._is_nova_model():
            return self._format_nova_request(
                messages, tool_specs, system_prompt_content, tool_choice
            )
        elif self._is_claude_model():
            return self._format_claude_request(
                messages, tool_specs, system_prompt_content, tool_choice
            )
        elif self._is_titan_embed_model():
            return self._format_titan_embed_request(messages)
        else:
            raise ValueError(
                f"model_id=<{self.config['model_id']}> | unsupported model"
            )

    def _format_nova_request(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
        tool_choice: ToolChoice | None = None,
    ) -> dict[str, Any]:
        """Format a request for Amazon Nova models.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt_content: System prompt content blocks to provide context to the model.
            tool_choice: Selection strategy for tool invocation.

        Returns:
            A formatted request for Amazon Nova models.
        """
        request: dict[str, Any] = {
            "messages": messages,
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
        }

        # Add system prompt if provided
        if system_prompt_content:
            request["system"] = system_prompt_content

        # Add tool specs if provided
        if tool_specs:
            request["toolConfig"] = {
                "tools": [{"toolSpec": tool_spec} for tool_spec in tool_specs],
                "toolChoice": tool_choice if tool_choice else {"auto": {}},
            }

        # Add additional arguments if provided
        if (
            "additional_args" in self.config
            and self.config["additional_args"] is not None
        ):
            request.update(self.config["additional_args"])

        return request

    def _format_claude_request(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
        tool_choice: ToolChoice | None = None,
    ) -> dict[str, Any]:
        """Format a request for Anthropic Claude models.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt_content: System prompt content blocks to provide context to the model.
            tool_choice: Selection strategy for tool invocation.

        Returns:
            A formatted request for Anthropic Claude models.
        """
        # For Claude models, we use the same format as Nova models
        # since we're using the converse API for both
        return self._format_nova_request(
            messages, tool_specs, system_prompt_content, tool_choice
        )

    def _format_titan_embed_request(self, messages: Messages) -> dict[str, Any]:
        """Format a request for Amazon Titan Embedding models.

        Args:
            messages: List of message objects to be processed by the model.

        Returns:
            A formatted request for Amazon Titan Embedding models.
        """
        # Extract the text from the last user message
        input_text = ""
        for message in reversed(messages):
            if message["role"] == "user" and "content" in message:
                content_blocks = message["content"]
                if isinstance(content_blocks, list):
                    for block in content_blocks:
                        if "text" in block:
                            input_text = block["text"]
                            break
                if input_text:
                    break

        request: dict[str, Any] = {
            "inputText": input_text,
        }

        # Add additional arguments if provided
        if (
            "additional_args" in self.config
            and self.config["additional_args"] is not None
        ):
            request.update(self.config["additional_args"])

        return request

    def _format_chunk(self, event: dict[str, Any]) -> StreamEvent:
        """Format the SAP GenAI Hub response events into standardized message chunks.

        Args:
            event: A response event from the SAP GenAI Hub model.

        Returns:
            The formatted chunk.
        """
        # Handle string events by wrapping them in a proper structure
        if isinstance(event, str):
            return {"contentBlockDelta": {"delta": {"text": event}}}

        # If it's already a proper dictionary, return it as is
        if isinstance(event, dict):
            return cast(StreamEvent, event)

        # For any other type, convert to string and wrap
        return {"contentBlockDelta": {"delta": {"text": str(event)}}}

    def _convert_streaming_response(
        self, stream_response: Any
    ) -> Iterable[StreamEvent]:
        """Convert a streaming response to the standardized streaming format.

        Args:
            stream_response: The streaming response from the SAP GenAI Hub model.

        Returns:
            An iterable of response events in the streaming format.
        """
        try:
            logger.debug(
                "stream_type=<%s> | converting streaming response",
                type(stream_response).__name__,
            )

            message_started = False
            event_count = 0

            # Check if it's a Bedrock-style response with 'stream' key
            if hasattr(stream_response, "get") and callable(stream_response.get):
                event_stream = stream_response.get("stream")
                if event_stream:
                    logger.debug("processing bedrock-style event stream")

                    for event in event_stream:
                        event_count += 1

                        # Start message if not started
                        if not message_started:
                            yield {"messageStart": {"role": "assistant"}}
                            message_started = True

                        # Process the event based on its structure
                        if isinstance(event, dict):
                            # Pass through properly formatted events
                            if any(
                                key in event
                                for key in [
                                    "contentBlockDelta",
                                    "contentBlockStart",
                                    "contentBlockStop",
                                    "messageStart",
                                    "messageStop",
                                    "metadata",
                                ]
                            ):
                                yield event
                            else:
                                # Format unknown events
                                yield self._format_chunk(event)
                        else:
                            # Handle non-dict events (strings, etc.)
                            yield self._format_chunk(event)

                    logger.debug(
                        "event_count=<%d> | processed bedrock stream events",
                        event_count,
                    )
                    return

            # Try to iterate directly over the stream_response
            try:
                if hasattr(stream_response, "__iter__"):
                    logger.debug("processing iterable stream response")

                    for event in stream_response:
                        event_count += 1

                        # Start message if not started
                        if not message_started:
                            yield {"messageStart": {"role": "assistant"}}
                            message_started = True

                        # Process the event
                        if isinstance(event, dict):
                            # Check if it's already a properly formatted streaming event
                            if any(
                                key in event
                                for key in [
                                    "contentBlockDelta",
                                    "contentBlockStart",
                                    "contentBlockStop",
                                    "messageStart",
                                    "messageStop",
                                    "metadata",
                                ]
                            ):
                                yield event

                                # If this is a messageStop event, we're done
                                if "messageStop" in event:
                                    logger.debug(
                                        "received messageStop event from stream"
                                    )
                                    return
                            else:
                                # Format unknown events
                                yield self._format_chunk(event)
                        else:
                            # Handle strings and other types
                            yield self._format_chunk(event)

                    logger.debug(
                        "event_count=<%d> | processed direct iteration events",
                        event_count,
                    )
                else:
                    logger.debug(
                        "stream response not iterable, treating as single response"
                    )
                    yield {"messageStart": {"role": "assistant"}}
                    yield self._format_chunk(stream_response)

            except TypeError as te:
                # stream_response is not iterable
                logger.debug(
                    "error=<%s> | stream response not iterable, treating as single response",
                    te,
                )
                yield {"messageStart": {"role": "assistant"}}
                yield self._format_chunk(stream_response)

        except Exception as e:
            logger.error(
                "error=<%s>, stream_type=<%s> | error processing streaming response",
                e,
                type(stream_response),
            )
            raise e

    def _convert_non_streaming_to_streaming(
        self, response: dict[str, Any]
    ) -> Iterable[StreamEvent]:
        """Convert a non-streaming response to the streaming format.

        Args:
            response: The non-streaming response from the SAP GenAI Hub model.

        Yields:
            Response events in the streaming format.
        """
        if self._is_nova_model() or self._is_claude_model():
            # Nova and Claude models have a similar response format when using converse API
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

                    yield {
                        "contentBlockDelta": {
                            "delta": {"toolUse": {"input": input_value}}
                        }
                    }
                elif "text" in content:
                    # Then yield the text as a delta
                    yield {
                        "contentBlockDelta": {
                            "delta": {"text": content["text"]},
                        }
                    }

                # Yield contentBlockStop event
                yield {"contentBlockStop": {}}

            # Yield messageStop event
            yield {
                "messageStop": {
                    "stopReason": response.get("stopReason", "stop"),
                }
            }

            # Yield metadata event
            if "usage" in response or "metrics" in response:
                metadata: StreamEvent = {"metadata": {}}
                if "usage" in response:
                    metadata["metadata"]["usage"] = response["usage"]
                if "metrics" in response:
                    metadata["metadata"]["metrics"] = response["metrics"]
                yield metadata

        elif self._is_titan_embed_model():
            # Titan Embedding models have a different response format
            # Yield messageStart event
            yield {"messageStart": {"role": "assistant"}}

            # Yield content block for embedding
            if "embedding" in response:
                yield {
                    "contentBlockDelta": {
                        "delta": {
                            "text": f"Embedding generated with {len(response['embedding'])} dimensions"
                        },
                    }
                }

            # Yield contentBlockStop event
            yield {"contentBlockStop": {}}

            # Yield messageStop event
            yield {
                "messageStop": {
                    "stopReason": "stop",
                }
            }

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
        """Send the request to the SAP GenAI Hub model and get the response.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            tool_choice: Selection strategy for tool invocation.
            system_prompt_content: System prompt content blocks to provide context to the model.
            **kwargs: Additional keyword arguments.

        Yields:
            Response events from the SAP GenAI Hub model

        Raises:
            ContextWindowOverflowException: If the input exceeds the model's context window.
            ModelThrottledException: If the model service is throttling requests.
        """

        def callback(event: Optional[StreamEvent] = None) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[Optional[StreamEvent]] = asyncio.Queue()

        # Handle backward compatibility: if system_prompt is provided but system_prompt_content is None
        if system_prompt and system_prompt_content is None:
            system_prompt_content = [{"text": system_prompt}]

        thread = asyncio.to_thread(
            self._stream,
            callback,
            messages,
            tool_specs,
            system_prompt_content,
            tool_choice,
        )
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
        """Stream conversation with the SAP GenAI Hub model.

        This method operates in a separate thread to avoid blocking the async event loop.

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
            request = self._format_request(
                messages, tool_specs, system_prompt_content, tool_choice
            )

            logger.debug("invoking model")
            streaming = self.config.get("streaming", True)

            if self._is_nova_model() or self._is_claude_model():
                # Try converse_stream first, fall back to converse if not supported
                try:
                    logger.debug("attempting converse_stream api")
                    stream_response = self.client.converse_stream(**request)

                    # Process all streaming events
                    event_count = 0
                    has_content = False

                    for event in self._convert_streaming_response(stream_response):
                        event_count += 1

                        # Check if we have actual content
                        if "contentBlockDelta" in event:
                            has_content = True

                        callback(event)

                    logger.debug(
                        "event_count=<%d>, has_content=<%s> | processed streaming events",
                        event_count,
                        has_content,
                    )

                    # If we didn't get any content, fallback to non-streaming
                    if event_count == 0 or not has_content:
                        logger.debug(
                            "no content received from streaming, falling back to converse"
                        )
                        response = self.client.converse(**request)
                        for event in self._convert_non_streaming_to_streaming(response):
                            callback(event)

                except NotImplementedError as nie:
                    # converse_stream not supported by this model/deployment, use converse
                    logger.debug("converse_stream not supported, using converse api")
                    response = self.client.converse(**request)
                    for event in self._convert_non_streaming_to_streaming(response):
                        callback(event)

                except Exception as e:
                    # Other errors - log and fallback to converse
                    logger.debug(
                        "error=<%s> | converse_stream failed, falling back to converse",
                        e,
                    )
                    response = self.client.converse(**request)
                    for event in self._convert_non_streaming_to_streaming(response):
                        callback(event)

            elif self._is_titan_embed_model():
                if streaming:
                    # Try streaming for Titan models
                    try:
                        logger.debug(
                            "using invoke_model_with_response_stream for titan"
                        )
                        stream_response = self.client.invoke_model_with_response_stream(
                            **request
                        )

                        event_count = 0
                        for event in self._convert_streaming_response(stream_response):
                            event_count += 1
                            callback(event)

                        if event_count == 0:
                            logger.warning(
                                "no events from titan streaming, falling back to non-streaming"
                            )
                            response = self.client.invoke_model(**request)
                            for event in self._convert_non_streaming_to_streaming(
                                response
                            ):
                                callback(event)

                    except (AttributeError, Exception) as e:
                        logger.warning(
                            "error=<%s> | titan streaming failed, falling back to non-streaming",
                            e,
                        )
                        response = self.client.invoke_model(**request)
                        for event in self._convert_non_streaming_to_streaming(response):
                            callback(event)
                else:
                    # Non-streaming path for Titan
                    logger.debug("using non-streaming invoke_model api for titan")
                    response = self.client.invoke_model(**request)
                    for event in self._convert_non_streaming_to_streaming(response):
                        callback(event)

        except Exception as e:
            error_message = str(e)

            # Handle throttling error
            if "ThrottlingException" in error_message:
                raise ModelThrottledException(error_message) from e

            # Handle context window overflow
            if any(
                overflow_message in error_message
                for overflow_message in CONTEXT_WINDOW_OVERFLOW_MESSAGES
            ):
                logger.warning("sap genai hub threw context window overflow error")
                raise ContextWindowOverflowException(e) from e

            # Otherwise raise the error
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
        """Get structured output from the model.

        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Model events with the last being the structured output.
        """
        from ..event_loop import streaming
        from ..tools import convert_pydantic_to_tool_spec

        # Create a tool spec from the schema
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
            raise ValueError(
                f'Model returned stop_reason: {stop_reason} instead of "tool_use".'
            )

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
            raise ValueError(
                "No valid tool use or tool use input was found in the response."
            )

        yield {"output": output_model(**output_response)}
