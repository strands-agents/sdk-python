"""LiteLLM model provider.

- Docs: https://docs.litellm.ai/
"""

import json
import logging
from typing import Any, Generator, Optional, Type, TypedDict, TypeVar, Union, cast

import litellm
from litellm.utils import supports_response_schema
from pydantic import BaseModel
from typing_extensions import Unpack, override

from ..types.content import ContentBlock, Messages
from .openai import OpenAIModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LiteLLMModel(OpenAIModel):
    """LiteLLM model provider implementation."""

    class LiteLLMConfig(TypedDict, total=False):
        """Configuration options for LiteLLM models.

        Attributes:
            model_id: Model ID (e.g., "openai/gpt-4o", "anthropic/claude-3-sonnet").
                For a complete list of supported models, see https://docs.litellm.ai/docs/providers.
            params: Model parameters (e.g., max_tokens).
                For a complete list of supported parameters, see
                https://docs.litellm.ai/docs/completion/input#input-params-1.
        """

        model_id: str
        params: Optional[dict[str, Any]]

    def __init__(self, client_args: Optional[dict[str, Any]] = None, **model_config: Unpack[LiteLLMConfig]) -> None:
        """Initialize provider instance.

        Args:
            client_args: Arguments for the LiteLLM client.
                For a complete list of supported arguments, see
                https://github.com/BerriAI/litellm/blob/main/litellm/main.py.
            **model_config: Configuration options for the LiteLLM model.
        """
        self.config = dict(model_config)

        logger.debug("config=<%s> | initializing", self.config)

        client_args = client_args or {}
        self.client = litellm.LiteLLM(**client_args)

    @override
    def update_config(self, **model_config: Unpack[LiteLLMConfig]) -> None:  # type: ignore[override]
        """Update the LiteLLM model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)

    @override
    def get_config(self) -> LiteLLMConfig:
        """Get the LiteLLM model configuration.

        Returns:
            The LiteLLM model configuration.
        """
        return cast(LiteLLMModel.LiteLLMConfig, self.config)

    @override
    @classmethod
    def format_request_message_content(cls, content: ContentBlock) -> dict[str, Any]:
        """Format a LiteLLM content block.

        Args:
            content: Message content.

        Returns:
            LiteLLM formatted content block.

        Raises:
            TypeError: If the content block type cannot be converted to a LiteLLM-compatible format.
        """
        if "reasoningContent" in content:
            return {
                "signature": content["reasoningContent"]["reasoningText"]["signature"],
                "thinking": content["reasoningContent"]["reasoningText"]["text"],
                "type": "thinking",
            }

        if "video" in content:
            return {
                "type": "video_url",
                "video_url": {
                    "detail": "auto",
                    "url": content["video"]["source"]["bytes"],
                },
            }

        return super().format_request_message_content(content)

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
        # The LiteLLM `Client` inits with Chat().
        # Chat() inits with self.completions
        # completions() has a method `create()` which wraps the real completion API of Litellm
        response = self.client.chat.completions.create(
            model=self.get_config()["model_id"],
            messages=super().format_request(prompt)["messages"],
            response_format=output_model,
        )

        if not supports_response_schema(self.get_config()["model_id"]):
            raise ValueError("Model does not support response_format")
        if len(response.choices) > 1:
            raise ValueError("Multiple choices found in the response.")

        # Find the first choice with tool_calls
        for choice in response.choices:
            if choice.finish_reason == "tool_calls":
                try:
                    # Parse the tool call content as JSON
                    tool_call_data = json.loads(choice.message.content)
                    # Instantiate the output model with the parsed data
                    yield {"output": output_model(**tool_call_data)}
                    return
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    raise ValueError(f"Failed to parse or load content into model: {e}") from e

        # If no tool_calls found, raise an error
        raise ValueError("No tool_calls found in response")
