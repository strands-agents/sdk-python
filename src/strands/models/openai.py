"""OpenAI model provider.

- Docs: https://platform.openai.com/docs/overview
"""

import logging
from typing import Any, AsyncGenerator, Optional, Protocol, Type, TypedDict, TypeVar, Union, cast

import openai
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from pydantic import BaseModel
from typing_extensions import Unpack, override

from ..types.content import Messages
from ..types.models import OpenAIModel as SAOpenAIModel
from ..types.streaming import StreamEvent
from ..types.tools import ToolSpec

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class Client(Protocol):
    """Protocol defining the OpenAI-compatible interface for the underlying provider client."""

    @property
    # pragma: no cover
    def chat(self) -> Any:
        """Chat completions interface."""
        ...


class OpenAIModel(SAOpenAIModel):
    """OpenAI model provider implementation."""

    client: Client

    class OpenAIConfig(TypedDict, total=False):
        """Configuration options for OpenAI models.

        Attributes:
            model_id: Model ID (e.g., "gpt-4o").
                For a complete list of supported models, see https://platform.openai.com/docs/models.
            params: Model parameters (e.g., max_tokens).
                For a complete list of supported parameters, see
                https://platform.openai.com/docs/api-reference/chat/create.
        """

        model_id: str
        params: Optional[dict[str, Any]]

    def __init__(self, client_args: Optional[dict[str, Any]] = None, **model_config: Unpack[OpenAIConfig]) -> None:
        """Initialize provider instance.

        Args:
            client_args: Arguments for the OpenAI client.
                For a complete list of supported arguments, see https://pypi.org/project/openai/.
            **model_config: Configuration options for the OpenAI model.
        """
        self.config = dict(model_config)

        logger.debug("config=<%s> | initializing", self.config)

        client_args = client_args or {}
        self.client = openai.AsyncOpenAI(**client_args)

    @override
    def update_config(self, **model_config: Unpack[OpenAIConfig]) -> None:  # type: ignore[override]
        """Update the OpenAI model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)

    @override
    def get_config(self) -> OpenAIConfig:
        """Get the OpenAI model configuration.

        Returns:
            The OpenAI model configuration.
        """
        return cast(OpenAIModel.OpenAIConfig, self.config)

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with the OpenAI model.

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
        logger.debug("formatted request=<%s>", request)

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
    ) -> AsyncGenerator[dict[str, Union[T, Any]], None]:
        """Get structured output from the model.

        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Model events with the last being the structured output.
        """
        response: ParsedChatCompletion = await self.client.beta.chat.completions.parse(  # type: ignore
            model=self.get_config()["model_id"],
            messages=self.format_request(prompt)["messages"],
            response_format=output_model,
        )

        parsed: T | None = None
        # Find the first choice with tool_calls
        if len(response.choices) > 1:
            raise ValueError("Multiple choices found in the OpenAI response.")

        for choice in response.choices:
            if isinstance(choice.message.parsed, output_model):
                parsed = choice.message.parsed
                break

        if parsed:
            yield {"output": parsed}
        else:
            raise ValueError("No valid tool use or tool use input was found in the OpenAI response.")
