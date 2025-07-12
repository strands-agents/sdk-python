"""DeepSeek model provider.

- Docs: https://platform.deepseek.com/api-docs/
"""

import json
import logging
from typing import Any, AsyncGenerator, Optional, Type, TypeVar

import openai
from pydantic import BaseModel
from typing_extensions import TypedDict, Unpack, override

from ..types.content import Messages
from ..types.streaming import StreamEvent
from ..types.tools import ToolSpec
from .model import Model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class DeepSeekModel(Model):
    """DeepSeek model provider implementation using OpenAI-compatible API."""

    class DeepSeekConfig(TypedDict, total=False):
        """Configuration parameters for DeepSeek models.

        Attributes:
            model_id: DeepSeek model ID (e.g., "deepseek-chat", "deepseek-reasoner").
            api_key: DeepSeek API key.
            base_url: API base URL.
            use_beta: Whether to use beta endpoint for advanced features.
            params: Additional model parameters.
        """

        model_id: str
        api_key: str
        base_url: Optional[str]
        use_beta: Optional[bool]
        params: Optional[dict[str, Any]]

    def __init__(
        self,
        api_key: str,
        *,
        model_id: str = "deepseek-chat",
        base_url: Optional[str] = None,
        use_beta: bool = False,
        params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize DeepSeek provider instance.

        Args:
            api_key: DeepSeek API key.
            model_id: Model ID to use.
            base_url: Custom base URL. Defaults to standard or beta endpoint.
            use_beta: Whether to use beta endpoint.
            params: Additional model parameters.
            **kwargs: Additional arguments for future extensibility.
        """
        if base_url is None:
            base_url = "https://api.deepseek.com/beta" if use_beta else "https://api.deepseek.com"

        self.config = DeepSeekModel.DeepSeekConfig(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            use_beta=use_beta,
            params=params or {},
        )

        logger.debug("config=<%s> | initializing", self.config)

        self.client = openai.AsyncOpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["base_url"],
        )

    @override
    def update_config(self, **model_config: Unpack[DeepSeekConfig]) -> None:  # type: ignore
        """Update the DeepSeek model configuration.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)

        # Recreate client if API settings changed
        if any(key in model_config for key in ["api_key", "base_url"]):
            self.client = openai.AsyncOpenAI(
                api_key=self.config["api_key"],
                base_url=self.config["base_url"],
            )

    @override
    def get_config(self) -> DeepSeekConfig:
        """Get the DeepSeek model configuration.

        Returns:
            The DeepSeek model configuration.
        """
        return self.config

    def format_request(
        self, messages: Messages, tool_specs: Optional[list[ToolSpec]] = None, system_prompt: Optional[str] = None
    ) -> dict[str, Any]:
        """Format a DeepSeek chat request.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            A DeepSeek chat request.
        """
        formatted_messages = []

        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})

        # Convert Strands messages to OpenAI format
        for message in messages:
            if message["role"] in ["user", "assistant"]:
                content = ""
                for block in message["content"]:
                    if "text" in block:
                        content += block["text"]
                formatted_messages.append({"role": message["role"], "content": content})

        request = {
            "model": self.config["model_id"],
            "messages": formatted_messages,
            "stream": True,
            **self.config.get("params", {}),
        }

        if tool_specs:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool_spec["name"],
                        "description": tool_spec["description"],
                        "parameters": tool_spec["inputSchema"]["json"],
                    },
                }
                for tool_spec in tool_specs
            ]

        return request

    def format_chunk(self, event: dict[str, Any]) -> StreamEvent:
        """Format DeepSeek response event into standardized message chunk.

        Args:
            event: A response event from the DeepSeek model.

        Returns:
            The formatted chunk.
        """
        match event["chunk_type"]:
            case "message_start":
                return {"messageStart": {"role": "assistant"}}

            case "content_start":
                return {"contentBlockStart": {"start": {}}}

            case "content_delta":
                if event["data_type"] == "reasoning_content":
                    return {"contentBlockDelta": {"delta": {"reasoningContent": {"text": event["data"]}}}}
                return {"contentBlockDelta": {"delta": {"text": event["data"]}}}

            case "content_stop":
                return {"contentBlockStop": {}}

            case "message_stop":
                return {"messageStop": {"stopReason": "end_turn"}}

            case _:
                return {"contentBlockDelta": {"delta": {"text": ""}}}

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with the DeepSeek model.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Formatted message chunks from the model.
        """
        logger.debug("formatting request")
        request = self.format_request(messages, tool_specs, system_prompt)
        logger.debug("request=<%s>", request)

        logger.debug("invoking model")
        response = await self.client.chat.completions.create(**request)

        logger.debug("got response from model")
        yield self.format_chunk({"chunk_type": "message_start"})
        yield self.format_chunk({"chunk_type": "content_start"})

        async for chunk in response:
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]

                if hasattr(choice, "delta") and choice.delta:
                    delta = choice.delta

                    # Handle reasoning content
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        yield self.format_chunk(
                            {
                                "chunk_type": "content_delta",
                                "data_type": "reasoning_content",
                                "data": delta.reasoning_content,
                            }
                        )

                    # Handle regular content
                    if hasattr(delta, "content") and delta.content:
                        yield self.format_chunk(
                            {
                                "chunk_type": "content_delta",
                                "data_type": "text",
                                "data": delta.content,
                            }
                        )

                if hasattr(choice, "finish_reason") and choice.finish_reason:
                    break

        yield self.format_chunk({"chunk_type": "content_stop"})
        yield self.format_chunk({"chunk_type": "message_stop"})

        logger.debug("finished streaming response from model")

    @override
    async def structured_output(
        self, output_model: Type[T], prompt: Messages, **kwargs: Any
    ) -> AsyncGenerator[dict[str, T], None]:
        """Get structured output from the model.

        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Model events with the structured output.
        """
        # Extract text from prompt
        text_prompt = ""
        for message in prompt:
            if message.get("role") == "user":
                for block in message.get("content", []):
                    if "text" in block:
                        text_prompt += block["text"]

        # Create JSON schema prompt
        schema = output_model.model_json_schema()
        properties = schema.get("properties", {})
        field_descriptions = [
            f"- {field_name}: {field_info.get('description', field_name)} ({field_info.get('type', 'string')})"
            for field_name, field_info in properties.items()
        ]

        json_prompt = f"""{text_prompt}

Extract the information and return it as JSON with these fields:
{chr(10).join(field_descriptions)}

Return only the JSON object with the extracted data, no additional text."""

        request_params = {
            "model": self.config["model_id"],
            "messages": [{"role": "user", "content": json_prompt}],
            "response_format": {"type": "json_object"},
        }

        # Add max_tokens for reasoning model
        if self.config["model_id"] == "deepseek-reasoner":
            request_params["max_tokens"] = 32000

        response = await self.client.chat.completions.create(**request_params)
        json_data = json.loads(response.choices[0].message.content)
        result = output_model(**json_data)

        yield {"output": result}
