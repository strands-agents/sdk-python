"""Baseten model provider.

- Docs: https://docs.baseten.co/
"""

import logging
from typing import Any, Generator, Iterable, Optional, Protocol, Type, TypedDict, TypeVar, Union, cast

import openai
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from pydantic import BaseModel
from typing_extensions import Unpack, override

from ..types.content import Messages
from ..types.models import OpenAIModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class Client(Protocol):
    """Protocol defining the OpenAI-compatible interface for the underlying provider client."""

    @property
    # pragma: no cover
    def chat(self) -> Any:
        """Chat completions interface."""
        ...


class BasetenModel(OpenAIModel):
    """Baseten model provider implementation."""

    client: Client

    class BasetenConfig(TypedDict, total=False):
        """Configuration options for Baseten models.

        Attributes:
            model_id: Model ID for the Baseten model.
                For Model APIs, use model slugs like "deepseek-ai/DeepSeek-R1-0528" or "meta-llama/Llama-4-Maverick-17B-128E-Instruct".
                For dedicated deployments, use the deployment ID.
            base_url: Base URL for the Baseten API.
                For Model APIs: https://inference.baseten.co/v1
                For dedicated deployments: https://model-xxxxxxx.api.baseten.co/environments/production/sync/v1
            params: Model parameters (e.g., max_tokens).
                For a complete list of supported parameters, see
                https://platform.openai.com/docs/api-reference/chat/create.
        """

        model_id: str
        base_url: Optional[str]
        params: Optional[dict[str, Any]]

    def __init__(self, client_args: Optional[dict[str, Any]] = None, **model_config: Unpack[BasetenConfig]) -> None:
        """Initialize provider instance.

        Args:
            client_args: Arguments for the Baseten client.
                For a complete list of supported arguments, see https://pypi.org/project/openai/.
            **model_config: Configuration options for the Baseten model.
        """
        self.config = dict(model_config)

        logger.debug("config=<%s> | initializing", self.config)

        client_args = client_args or {}
        
        # Set default base URL for Model APIs if not provided
        if "base_url" not in client_args and "base_url" not in self.config:
            client_args["base_url"] = "https://inference.baseten.co/v1"
        elif "base_url" in self.config:
            client_args["base_url"] = self.config["base_url"]
        
        self.client = openai.OpenAI(**client_args)

    @override
    def update_config(self, **model_config: Unpack[BasetenConfig]) -> None:  # type: ignore[override]
        """Update the Baseten model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)

    @override
    def get_config(self) -> BasetenConfig:
        """Get the Baseten model configuration.

        Returns:
            The Baseten model configuration.
        """
        return cast(BasetenModel.BasetenConfig, self.config)

    @override
    def stream(self, request: dict[str, Any]) -> Iterable[dict[str, Any]]:
        """Send the request to the Baseten model and get the streaming response.

        Args:
            request: The formatted request to send to the Baseten model.

        Returns:
            An iterable of response events from the Baseten model.
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
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.

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
            raise ValueError("Multiple choices found in the Baseten response.")

        for choice in response.choices:
            if isinstance(choice.message.parsed, output_model):
                parsed = choice.message.parsed
                break

        if parsed:
            yield {"output": parsed}
        else:
            raise ValueError("No valid tool use or tool use input was found in the Baseten response.") 