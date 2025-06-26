"""OpenAI model provider.

- Docs: https://platform.openai.com/docs/overview
"""

import logging
from typing import Any, Generator, Iterable, Optional, Protocol, Type, TypedDict, TypeVar, Union, cast

import openai
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from pydantic import BaseModel
from typing_extensions import Unpack, override

from ..types.content import Messages
from ..types.models import OpenAIModel as SAOpenAIModel

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
        self.client = openai.OpenAI(**client_args)

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
    def stream(self, request: dict[str, Any]) -> Iterable[dict[str, Any]]:
        """Send the request to the OpenAI model and get the streaming response.

        Args:
            request: The formatted request to send to the OpenAI model.

        Returns:
            An iterable of response events from the OpenAI model.
        """
        response = self.client.chat.completions.create(**request)

        yield {"chunk_type": "message_start"}
        yield {"chunk_type": "content_start", "data_type": "text"}

        tool_calls: dict[int, list[Any]] = {}

        for event in response:
            # Defensive: skip events with empty or missing choices
            if not getattr(event, "choices", None):
                continue
            choice = event.choices[0]

            if choice.delta.content:
                yield {"chunk_type": "content_delta", "data_type": "text", "data": choice.delta.content}

            if hasattr(choice.delta, "reasoning_content") and choice.delta.reasoning_content:
                yield {
                    "chunk_type": "content_delta",
                    "data_type": "reasoning_content",
                    "data": choice.delta.reasoning_content,
                }

            for tool_call in choice.delta.tool_calls or []:
                tool_calls.setdefault(tool_call.index, []).append(tool_call)

            if choice.finish_reason:
                break

        yield {"chunk_type": "content_stop", "data_type": "text"}

        for tool_deltas in tool_calls.values():
            yield {"chunk_type": "content_start", "data_type": "tool", "data": tool_deltas[0]}

            for tool_delta in tool_deltas:
                yield {"chunk_type": "content_delta", "data_type": "tool", "data": tool_delta}

            yield {"chunk_type": "content_stop", "data_type": "tool"}

        yield {"chunk_type": "message_stop", "data": choice.finish_reason}

        # Skip remaining events as we don't have use for anything except the final usage payload
        for event in response:
            _ = event

        yield {"chunk_type": "metadata", "data": event.usage}

    @override
    def structured_output(
        self, output_model: Type[T], prompt: Messages
    ) -> Generator[dict[str, Union[T, Any]], None, None]:
        """Get structured output from the model.

        Args:
            output_model(Type[BaseModel]): The output model to use for the agent.
            prompt(Messages): The prompt messages to use for the agent.

        Yields:
            Model events with the last being the structured output.
        """
        response: ParsedChatCompletion = self.client.beta.chat.completions.parse(  # type: ignore
            model=self.get_config()["model_id"],
            messages=super().format_request(prompt)["messages"],
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
