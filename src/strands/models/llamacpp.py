"""llama.cpp model provider.

Provides integration with llama.cpp servers running in OpenAI-compatible mode,
with support for advanced llama.cpp-specific features.

- Docs: https://github.com/ggml-org/llama.cpp
- Server docs: https://github.com/ggml-org/llama.cpp/tree/master/tools/server
- OpenAI API compatibility:
  https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#api-endpoints
"""

import base64
import json
import logging
import mimetypes
from typing import Any, AsyncGenerator, Optional, Type, TypedDict, TypeVar, Union

import httpx
from pydantic import BaseModel
from typing_extensions import Unpack, override

from ..types.content import ContentBlock, Messages
from ..types.exceptions import ContextWindowOverflowException, ModelThrottledException
from ..types.streaming import StreamEvent
from ..types.tools import ToolSpec
from .model import Model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LlamaCppError(Exception):
    """Base exception for llama.cpp specific errors.

    This exception serves as the base class for all llama.cpp-specific errors,
    allowing for targeted error handling in client code.
    """


class LlamaCppContextOverflowError(LlamaCppError, ContextWindowOverflowException):
    """Raised when context window is exceeded in llama.cpp.

    This error occurs when the combined input and output tokens exceed
    the model's context window size. Common causes include:
    - Long input prompts
    - Extended conversations
    - Large system prompts or tool definitions
    """


class LlamaCppModel(Model):
    """llama.cpp model provider implementation.

    Connects to a llama.cpp server running in OpenAI-compatible mode with
    support for advanced llama.cpp-specific features like grammar constraints,
    Mirostat sampling, native JSON schema validation, and native multimodal
    support for audio and image content.

    The llama.cpp server must be started with the OpenAI-compatible API enabled:
        llama-server -m model.gguf --host 0.0.0.0 --port 8080

    Example:
        Basic usage:
        >>> model = LlamaCppModel(base_url="http://localhost:8080")
        >>> model.update_config(params={"temperature": 0.7, "top_k": 40})

        Grammar constraints:
        >>> model.use_grammar_constraint('''
        ...     root ::= answer
        ...     answer ::= "yes" | "no"
        ... ''')

        Advanced sampling:
        >>> model.update_config(params={
        ...     "mirostat": 2,
        ...     "mirostat_lr": 0.1,
        ...     "tfs_z": 0.95,
        ...     "repeat_penalty": 1.1
        ... })

        Multimodal usage (requires multimodal model like Qwen2.5-Omni):
        >>> # Audio analysis
        >>> audio_content = [{
        ...     "audio": {"source": {"bytes": audio_bytes}, "format": "wav"},
        ...     "text": "What do you hear in this audio?"
        ... }]
        >>> response = agent(audio_content)

        >>> # Image analysis
        >>> image_content = [{
        ...     "image": {"source": {"bytes": image_bytes}, "format": "png"},
        ...     "text": "Describe this image"
        ... }]
        >>> response = agent(image_content)
    """

    class LlamaCppConfig(TypedDict, total=False):
        """Configuration options for llama.cpp models.

        Attributes:
            model_id: Model identifier for the loaded model in llama.cpp server.
                Default is "default" as llama.cpp typically loads a single model.
            params: Model parameters supporting both OpenAI and llama.cpp-specific options.

                OpenAI-compatible parameters:
                - max_tokens: Maximum number of tokens to generate
                - temperature: Sampling temperature (0.0 to 2.0)
                - top_p: Nucleus sampling parameter (0.0 to 1.0)
                - frequency_penalty: Frequency penalty (-2.0 to 2.0)
                - presence_penalty: Presence penalty (-2.0 to 2.0)
                - stop: List of stop sequences
                - seed: Random seed for reproducibility
                - n: Number of completions to generate
                - logprobs: Include log probabilities in output
                - top_logprobs: Number of top log probabilities to include

                llama.cpp-specific parameters:
                - repeat_penalty: Penalize repeat tokens (1.0 = no penalty)
                - top_k: Top-k sampling (0 = disabled)
                - min_p: Min-p sampling threshold (0.0 to 1.0)
                - typical_p: Typical-p sampling (0.0 to 1.0)
                - tfs_z: Tail-free sampling parameter (0.0 to 1.0)
                - top_a: Top-a sampling parameter
                - mirostat: Mirostat sampling mode (0, 1, or 2)
                - mirostat_lr: Mirostat learning rate
                - mirostat_ent: Mirostat target entropy
                - grammar: GBNF grammar string for constrained generation
                - json_schema: JSON schema for structured output
                - penalty_last_n: Number of tokens to consider for penalties
                - n_probs: Number of probabilities to return per token
                - min_keep: Minimum tokens to keep in sampling
                - ignore_eos: Ignore end-of-sequence token
                - logit_bias: Token ID to bias mapping
                - cache_prompt: Cache the prompt for faster generation
                - slot_id: Slot ID for parallel inference
                - samplers: Custom sampler order
        """

        model_id: str
        params: Optional[dict[str, Any]]

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: Optional[Union[float, tuple[float, float]]] = None,
        **model_config: Unpack[LlamaCppConfig],
    ) -> None:
        """Initialize llama.cpp provider instance.

        Args:
            base_url: Base URL for the llama.cpp server.
                Default is "http://localhost:8080" for local server.
            timeout: Request timeout in seconds. Can be float or tuple of
                (connect, read) timeouts.
            **model_config: Configuration options for the llama.cpp model.
        """
        # Set default model_id if not provided
        if "model_id" not in model_config:
            model_config["model_id"] = "default"

        self.base_url = base_url.rstrip("/")
        self.config = dict(model_config)

        # Configure HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout or 30.0,
        )

        logger.debug(
            "base_url=<%s>, model_id=<%s> | initializing llama.cpp provider",
            base_url,
            model_config.get("model_id"),
        )

    @override
    def update_config(
        self, **model_config: Unpack[LlamaCppConfig]
    ) -> None:  # type: ignore[override]
        """Update the llama.cpp model configuration with provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)

    @override
    def get_config(self) -> LlamaCppConfig:
        """Get the llama.cpp model configuration.

        Returns:
            The llama.cpp model configuration.
        """
        return self.config  # type: ignore[return-value]

    def use_grammar_constraint(self, grammar: str) -> None:
        r"""Apply a GBNF grammar constraint to the generation.

        Args:
            grammar: GBNF (Backus-Naur Form) grammar string defining allowed outputs.
                     See https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md

        Example:
            >>> # Constrain output to yes/no answers
            >>> model.use_grammar_constraint('''
            ...     root ::= answer
            ...     answer ::= "yes" | "no"
            ... ''')

            >>> # JSON object grammar
            >>> model.use_grammar_constraint('''
            ...     root ::= object
            ...     object ::= "{" pair ("," pair)* "}"
            ...     pair ::= string ":" value
            ...     string ::= "\"" [^"]* "\""
            ...     value ::= string | number | "true" | "false" | "null"
            ...     number ::= "-"? [0-9]+ ("." [0-9]+)?
            ... ''')
        """
        if not self.config.get("params"):
            self.config["params"] = {}
        self.config["params"]["grammar"] = grammar
        logger.debug(
            "grammar=<%s> | applied grammar constraint",
            grammar[:50] + "..." if len(grammar) > 50 else grammar,
        )

    def use_json_schema(self, schema: dict[str, Any]) -> None:
        """Apply a JSON schema constraint for structured output.

        Args:
            schema: JSON schema dictionary defining the expected output structure.

        Example:
            >>> model.use_json_schema({
            ...     "type": "object",
            ...     "properties": {
            ...         "name": {"type": "string"},
            ...         "age": {"type": "integer", "minimum": 0}
            ...     },
            ...     "required": ["name", "age"]
            ... })
        """
        if not self.config.get("params"):
            self.config["params"] = {}
        self.config["params"]["json_schema"] = schema
        logger.debug("schema=<%s> | applied JSON schema constraint", schema)

    def _format_message_content(self, content: ContentBlock) -> dict[str, Any]:
        """Format a content block for llama.cpp.

        Args:
            content: Message content.

        Returns:
            llama.cpp compatible content block.

        Raises:
            TypeError: If the content block type cannot be converted to a compatible format.
        """
        if "document" in content:
            mime_type = mimetypes.types_map.get(
                f".{content['document']['format']}", "application/octet-stream"
            )
            file_data = base64.b64encode(content["document"]["source"]["bytes"]).decode(
                "utf-8"
            )
            return {
                "file": {
                    "file_data": f"data:{mime_type};base64,{file_data}",
                    "filename": content["document"]["name"],
                },
                "type": "file",
            }

        if "image" in content:
            mime_type = mimetypes.types_map.get(
                f".{content['image']['format']}", "application/octet-stream"
            )
            image_data = base64.b64encode(content["image"]["source"]["bytes"]).decode(
                "utf-8"
            )
            return {
                "image_url": {
                    "detail": "auto",
                    "format": mime_type,
                    "url": f"data:{mime_type};base64,{image_data}",
                },
                "type": "image_url",
            }

        if "audio" in content:
            audio_data = base64.b64encode(content["audio"]["source"]["bytes"]).decode(
                "utf-8"
            )
            audio_format = content["audio"].get("format", "wav")
            return {
                "type": "input_audio",
                "input_audio": {"data": audio_data, "format": audio_format},
            }

        if "text" in content:
            return {"text": content["text"], "type": "text"}

        raise TypeError(f"content_type=<{next(iter(content))}> | unsupported type")

    def _format_tool_call(self, tool_use: dict[str, Any]) -> dict[str, Any]:
        """Format a tool call for llama.cpp.

        Args:
            tool_use: Tool use requested by the model.

        Returns:
            llama.cpp compatible tool call.
        """
        return {
            "function": {
                "arguments": json.dumps(tool_use["input"]),
                "name": tool_use["name"],
            },
            "id": tool_use["toolUseId"],
            "type": "function",
        }

    def _format_tool_message(self, tool_result: dict[str, Any]) -> dict[str, Any]:
        """Format a tool message for llama.cpp.

        Args:
            tool_result: Tool result collected from a tool execution.

        Returns:
            llama.cpp compatible tool message.
        """
        contents = [
            {"text": json.dumps(content["json"])} if "json" in content else content
            for content in tool_result["content"]
        ]

        return {
            "role": "tool",
            "tool_call_id": tool_result["toolUseId"],
            "content": [self._format_message_content(content) for content in contents],
        }

    def _format_messages(
        self, messages: Messages, system_prompt: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Format messages for llama.cpp.

        Args:
            messages: List of message objects to be processed.
            system_prompt: System prompt to provide context to the model.

        Returns:
            Formatted messages array compatible with llama.cpp.
        """
        formatted_messages = []

        # Add system prompt if provided
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})

        for message in messages:
            contents = message["content"]

            formatted_contents = [
                self._format_message_content(content)
                for content in contents
                if not any(
                    block_type in content for block_type in ["toolResult", "toolUse"]
                )
            ]
            formatted_tool_calls = [
                self._format_tool_call(content["toolUse"])
                for content in contents
                if "toolUse" in content
            ]
            formatted_tool_messages = [
                self._format_tool_message(content["toolResult"])
                for content in contents
                if "toolResult" in content
            ]

            formatted_message = {
                "role": message["role"],
                "content": formatted_contents,
                **(
                    {}
                    if not formatted_tool_calls
                    else {"tool_calls": formatted_tool_calls}
                ),
            }
            formatted_messages.append(formatted_message)
            formatted_messages.extend(formatted_tool_messages)

        return [
            message
            for message in formatted_messages
            if message["content"] or "tool_calls" in message
        ]

    def _format_request(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Format a request for the llama.cpp server.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            A request formatted for llama.cpp server's OpenAI-compatible API.
        """
        # Separate OpenAI-compatible and llama.cpp-specific parameters
        request = {
            "messages": self._format_messages(messages, system_prompt),
            "model": self.config["model_id"],
            "stream": True,
            "stream_options": {"include_usage": True},
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
        }

        # Handle parameters if provided
        if self.config.get("params"):
            params = self.config["params"]

            # llama.cpp-specific parameters that must be passed via extra_body
            llamacpp_specific_params = {
                "repeat_penalty",
                "top_k",
                "min_p",
                "typical_p",
                "tfs_z",
                "top_a",
                "mirostat",
                "mirostat_lr",
                "mirostat_ent",
                "grammar",
                "json_schema",
                "penalty_last_n",
                "n_probs",
                "min_keep",
                "ignore_eos",
                "logit_bias",
                "cache_prompt",
                "slot_id",
                "samplers",
            }

            # Standard OpenAI parameters that go directly in the request
            openai_params = {
                "temperature",
                "max_tokens",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "stop",
                "seed",
                "n",
                "logprobs",
                "top_logprobs",
                "response_format",
            }

            # Add OpenAI parameters directly to request
            for param, value in params.items():
                if param in openai_params:
                    request[param] = value

            # Collect llama.cpp-specific parameters for extra_body
            extra_body = {}
            for param, value in params.items():
                if param in llamacpp_specific_params:
                    extra_body[param] = value

            # Add extra_body if we have llama.cpp-specific parameters
            if extra_body:
                request["extra_body"] = extra_body

        return request

    def _format_chunk(self, event: dict[str, Any]) -> StreamEvent:
        """Format a llama.cpp response event into a standardized message chunk.

        Args:
            event: A response event from the llama.cpp server.

        Returns:
            The formatted chunk.

        Raises:
            RuntimeError: If chunk_type is not recognized.
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
                                    "name": event["data"].function.name,
                                    "toolUseId": event["data"].id,
                                }
                            }
                        }
                    }
                return {"contentBlockStart": {"start": {}}}

            case "content_delta":
                if event["data_type"] == "tool":
                    return {
                        "contentBlockDelta": {
                            "delta": {
                                "toolUse": {
                                    "input": event["data"].function.arguments or ""
                                }
                            }
                        }
                    }
                if event["data_type"] == "reasoning_content":
                    return {
                        "contentBlockDelta": {
                            "delta": {"reasoningContent": {"text": event["data"]}}
                        }
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
                            "inputTokens": event["data"].prompt_tokens,
                            "outputTokens": event["data"].completion_tokens,
                            "totalTokens": event["data"].total_tokens,
                        },
                        "metrics": {
                            "latencyMs": 0,  # TODO: Add actual latency calculation
                        },
                    },
                }

            case _:
                raise RuntimeError(f"chunk_type=<{event['chunk_type']}> | unknown type")

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with the llama.cpp model.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Formatted message chunks from the model.

        Raises:
            LlamaCppContextOverflowError: When the context window is exceeded.
            ModelThrottledException: When the llama.cpp server is overloaded.
        """
        try:
            logger.debug("formatting request for llama.cpp server")
            request = self._format_request(messages, tool_specs, system_prompt)
            logger.debug("request=<%s>", request)

            logger.debug("sending request to llama.cpp server")
            response = await self.client.post("/v1/chat/completions", json=request)
            response.raise_for_status()

            logger.debug("processing streaming response")
            yield self._format_chunk({"chunk_type": "message_start"})
            yield self._format_chunk(
                {"chunk_type": "content_start", "data_type": "text"}
            )

            tool_calls = {}
            usage_data = None

            async for line in response.aiter_lines():
                if not line.strip() or not line.startswith("data: "):
                    continue

                data_content = line[6:]  # Remove "data: " prefix
                if data_content.strip() == "[DONE]":
                    break

                try:
                    event = json.loads(data_content)
                except json.JSONDecodeError:
                    continue

                # Handle usage information
                if "usage" in event:
                    usage_data = event["usage"]
                    continue

                if not event.get("choices"):
                    continue

                choice = event["choices"][0]
                delta = choice.get("delta", {})

                # Handle content deltas
                if "content" in delta and delta["content"]:
                    yield self._format_chunk(
                        {
                            "chunk_type": "content_delta",
                            "data_type": "text",
                            "data": delta["content"],
                        }
                    )

                # Handle tool calls
                if "tool_calls" in delta:
                    for tool_call in delta["tool_calls"]:
                        index = tool_call["index"]
                        if index not in tool_calls:
                            tool_calls[index] = []
                        tool_calls[index].append(tool_call)

                # Check for finish reason
                if choice.get("finish_reason"):
                    break

            yield self._format_chunk({"chunk_type": "content_stop"})

            # Process tool calls
            for tool_deltas in tool_calls.values():
                first_delta = tool_deltas[0]
                yield self._format_chunk(
                    {
                        "chunk_type": "content_start",
                        "data_type": "tool",
                        "data": type(
                            "ToolCall",
                            (),
                            {
                                "function": type(
                                    "Function",
                                    (),
                                    {
                                        "name": first_delta.get("function", {}).get(
                                            "name", ""
                                        ),
                                    },
                                )(),
                                "id": first_delta.get("id", ""),
                            },
                        )(),
                    }
                )

                for tool_delta in tool_deltas:
                    yield self._format_chunk(
                        {
                            "chunk_type": "content_delta",
                            "data_type": "tool",
                            "data": type(
                                "ToolCall",
                                (),
                                {
                                    "function": type(
                                        "Function",
                                        (),
                                        {
                                            "arguments": tool_delta.get(
                                                "function", {}
                                            ).get("arguments", ""),
                                        },
                                    )(),
                                },
                            )(),
                        }
                    )

                yield self._format_chunk({"chunk_type": "content_stop"})

            # Send stop reason
            stop_reason = (
                "tool_use"
                if tool_calls
                else getattr(choice, "finish_reason", "end_turn")
            )
            yield self._format_chunk(
                {"chunk_type": "message_stop", "data": stop_reason}
            )

            # Send usage metadata if available
            if usage_data:
                yield self._format_chunk(
                    {
                        "chunk_type": "metadata",
                        "data": type(
                            "Usage",
                            (),
                            {
                                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                                "completion_tokens": usage_data.get(
                                    "completion_tokens", 0
                                ),
                                "total_tokens": usage_data.get("total_tokens", 0),
                            },
                        )(),
                    }
                )

            logger.debug("finished streaming response")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                # Parse error response from llama.cpp server
                try:
                    error_data = e.response.json()
                    error_msg = str(
                        error_data.get("error", {}).get("message", str(error_data))
                    )
                except (json.JSONDecodeError, KeyError, AttributeError):
                    error_msg = e.response.text

                # Check for context overflow by looking for specific error indicators
                if any(
                    term in error_msg.lower()
                    for term in ["context", "kv cache", "slot"]
                ):
                    raise LlamaCppContextOverflowError(
                        f"Context window exceeded: {error_msg}"
                    ) from e
            elif e.response.status_code == 503:
                raise ModelThrottledException(
                    "llama.cpp server is busy or overloaded"
                ) from e
            raise
        except Exception as e:
            # Handle other potential errors like rate limiting
            error_msg = str(e).lower()
            if "rate" in error_msg or "429" in str(e):
                raise ModelThrottledException(str(e)) from e
            raise

    @override
    async def structured_output(
        self,
        output_model: Type[T],
        prompt: Messages,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Union[T, Any]], None]:
        """Get structured output using llama.cpp's native JSON schema support.

        This implementation uses llama.cpp's json_schema parameter to constrain
        the model output to valid JSON matching the provided schema.

        Args:
            output_model: The Pydantic model defining the expected output structure.
            prompt: The prompt messages to use for generation.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Model events with the last being the structured output.

        Raises:
            json.JSONDecodeError: If the model output is not valid JSON.
            pydantic.ValidationError: If the output doesn't match the model schema.
        """
        # Get the JSON schema from the Pydantic model
        schema = output_model.model_json_schema()

        # Store current params to restore later
        original_params = self.config.get("params", {}).copy()

        try:
            # Configure for JSON output with schema constraint
            if not self.config.get("params"):
                self.config["params"] = {}

            self.config["params"]["json_schema"] = schema
            self.config["params"]["cache_prompt"] = True

            # Collect the response
            response_text = ""
            async for event in self.stream(
                prompt, system_prompt=system_prompt, **kwargs
            ):
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"]["delta"]
                    if "text" in delta:
                        response_text += delta["text"]
                # Forward events to caller
                yield event

            # Parse and validate the JSON response
            data = json.loads(response_text.strip())
            output_instance = output_model(**data)
            yield {"output": output_instance}

        finally:
            # Restore original configuration
            self.config["params"] = original_params
