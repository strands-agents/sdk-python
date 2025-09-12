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

    @classmethod
    def format_request_messages(cls, messages: Messages, system_prompt: Optional[str] = None) -> list[dict[str, Any]]:
        """Format DeepSeek compatible messages array (exactly like OpenAI)."""
        formatted_messages: list[dict[str, Any]]
        formatted_messages = [{"role": "system", "content": system_prompt}] if system_prompt else []

        for message in messages:
            contents = message["content"]

            formatted_contents = [
                {"text": content["text"], "type": "text"} for content in contents if "text" in content
            ]
            formatted_tool_calls = [
                {
                    "function": {
                        "arguments": json.dumps(content["toolUse"]["input"]),
                        "name": content["toolUse"]["name"],
                    },
                    "id": content["toolUse"]["toolUseId"],
                    "type": "function",
                }
                for content in contents
                if "toolUse" in content
            ]
            formatted_tool_messages = [
                {
                    "role": "tool",
                    "tool_call_id": content["toolResult"]["toolUseId"],
                    "content": "".join(
                        [
                            json.dumps(tool_content["json"]) if "json" in tool_content else tool_content["text"]
                            for tool_content in content["toolResult"]["content"]
                        ]
                    ),
                }
                for content in contents
                if "toolResult" in content
            ]

            # Flatten content for DeepSeek
            text_content = "".join([c["text"] for c in formatted_contents])

            formatted_message = {
                "role": message["role"],
                "content": text_content,
                **({"tool_calls": formatted_tool_calls} if formatted_tool_calls else {}),
            }
            formatted_messages.append(formatted_message)
            formatted_messages.extend(formatted_tool_messages)

        return [message for message in formatted_messages if message["content"] or "tool_calls" in message]

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
        request = {
            "model": self.config["model_id"],
            "messages": self.format_request_messages(messages, system_prompt),
            "stream": True,
            **(self.config.get("params") or {}),
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
                        "contentBlockDelta": {"delta": {"toolUse": {"input": event["data"].function.arguments or ""}}}
                    }
                if event["data_type"] == "reasoning_content":
                    return {"contentBlockDelta": {"delta": {"reasoningContent": {"text": event["data"]}}}}
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
                            "latencyMs": 0,
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
        logger.debug("request messages=<%s>", request.get("messages", []))
        logger.debug("request tools=<%s>", len(request.get("tools", [])))

        # Debug logging removed for production

        # Debug logging disabled for production
        # import logging
        # logging.basicConfig(level=logging.DEBUG)
        # logging.getLogger(__name__).setLevel(logging.DEBUG)

        logger.debug("invoking model")
        response = await self.client.chat.completions.create(**request)

        logger.debug("got response from model")
        yield self.format_chunk({"chunk_type": "message_start"})
        yield self.format_chunk({"chunk_type": "content_start", "data_type": "text"})

        tool_calls: dict[int, list[Any]] = {}

        async for event in response:
            # Defensive: skip events with empty or missing choices
            if not getattr(event, "choices", None):
                continue
            choice = event.choices[0]

            if choice.delta.content:
                yield self.format_chunk(
                    {"chunk_type": "content_delta", "data_type": "text", "data": choice.delta.content}
                )

            if hasattr(choice.delta, "reasoning_content") and choice.delta.reasoning_content:
                yield self.format_chunk(
                    {
                        "chunk_type": "content_delta",
                        "data_type": "reasoning_content",
                        "data": choice.delta.reasoning_content,
                    }
                )

            for tool_call in choice.delta.tool_calls or []:
                tool_calls.setdefault(tool_call.index, []).append(tool_call)

            if choice.finish_reason:
                break

        yield self.format_chunk({"chunk_type": "content_stop", "data_type": "text"})

        for tool_deltas in tool_calls.values():
            yield self.format_chunk({"chunk_type": "content_start", "data_type": "tool", "data": tool_deltas[0]})

            for tool_delta in tool_deltas:
                yield self.format_chunk({"chunk_type": "content_delta", "data_type": "tool", "data": tool_delta})

            yield self.format_chunk({"chunk_type": "content_stop", "data_type": "tool"})

        yield self.format_chunk({"chunk_type": "message_stop", "data": choice.finish_reason})

        # Skip remaining events as we don't have use for anything except the final usage payload
        async for event in response:
            _ = event

        yield self.format_chunk({"chunk_type": "metadata", "data": event.usage})

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

        request_params: dict[str, Any] = {
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
