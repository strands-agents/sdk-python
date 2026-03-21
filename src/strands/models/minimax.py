"""MiniMax model provider.

- Docs: https://platform.minimaxi.com/document/introduction
"""

import json
import logging
import os
import re
from collections.abc import AsyncGenerator
from typing import Any, TypedDict, TypeVar

import openai
from pydantic import BaseModel
from typing_extensions import Unpack, override

from ..types.content import Messages
from ..types.exceptions import ContextWindowOverflowException, ModelThrottledException
from ..types.streaming import StreamEvent
from ..types.tools import ToolChoice, ToolSpec
from ._validation import validate_config_keys
from .openai import OpenAIModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Default MiniMax API base URL
_DEFAULT_MINIMAX_BASE_URL = "https://api.minimax.io/v1"


class MinimaxModel(OpenAIModel):
    """MiniMax model provider implementation.

    This provider extends OpenAIModel to work with MiniMax's OpenAI-compatible API.

    MiniMax provides large language models including MiniMax-M2.7 and MiniMax-M2.5-highspeed,
    accessible through an OpenAI-compatible chat completions endpoint.

    Example usage::

        from strands import Agent
        from strands.models.minimax import MinimaxModel

        model = MinimaxModel(model_id="MiniMax-M2.7")
        agent = Agent(model=model)
        response = agent("Tell me about AI")

    Attributes:
        client: The underlying OpenAI-compatible async client for MiniMax API.
    """

    class MinimaxConfig(TypedDict, total=False):
        """Configuration options for MiniMax models.

        Attributes:
            model_id: Model ID (e.g., "MiniMax-M2.7", "MiniMax-M2.5-highspeed").
                For a complete list of supported models, see https://platform.minimaxi.com/document/models.
            params: Model parameters (e.g., max_tokens, temperature).
                For a complete list of supported parameters, see
                https://platform.minimaxi.com/document/chat-completion-v2.
        """

        model_id: str
        params: dict[str, Any] | None

    def __init__(
        self,
        client: "OpenAIModel.Client | None" = None,
        client_args: dict[str, Any] | None = None,
        **model_config: Unpack[MinimaxConfig],
    ) -> None:
        """Initialize provider instance.

        If no client or client_args are provided, the provider will automatically configure
        the MiniMax API base URL and read the API key from the ``MINIMAX_API_KEY`` environment
        variable.

        Args:
            client: Pre-configured OpenAI-compatible client to reuse across requests.
                When provided, this client will be reused for all requests and will NOT be closed
                by the model. The caller is responsible for managing the client lifecycle.
            client_args: Arguments for the OpenAI client.
                Defaults to using the MiniMax API base URL and API key from environment.
            **model_config: Configuration options for the MiniMax model.

        Raises:
            ValueError: If both ``client`` and ``client_args`` are provided.
        """
        # Set default client_args for MiniMax if no client is provided
        if client is None and client_args is None:
            client_args = {}

        if client_args is not None:
            client_args.setdefault("base_url", _DEFAULT_MINIMAX_BASE_URL)
            client_args.setdefault("api_key", os.environ.get("MINIMAX_API_KEY", ""))

        super().__init__(client=client, client_args=client_args, **model_config)

    @override
    def update_config(self, **model_config: Unpack[MinimaxConfig]) -> None:  # type: ignore[override]
        """Update the MiniMax model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        validate_config_keys(model_config, self.MinimaxConfig)
        self.config.update(model_config)

    @override
    def get_config(self) -> MinimaxConfig:
        """Get the MiniMax model configuration.

        Returns:
            The MiniMax model configuration.
        """
        from typing import cast

        return cast(MinimaxModel.MinimaxConfig, self.config)

    @override
    def format_request(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Format a MiniMax-compatible chat streaming request.

        Extends the OpenAI format_request to remove empty tool lists, which
        are not accepted by the MiniMax API.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            tool_choice: Selection strategy for tool invocation.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            A MiniMax-compatible chat streaming request.
        """
        request = super().format_request(messages, tool_specs, system_prompt, tool_choice, **kwargs)

        # MiniMax does not accept empty tools list
        if not request.get("tools"):
            request.pop("tools", None)

        return request

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with the MiniMax model.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            tool_choice: Selection strategy for tool invocation.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Formatted message chunks from the model.

        Raises:
            ContextWindowOverflowException: If the input exceeds the model's context window.
            ModelThrottledException: If the request is throttled by MiniMax (rate limits).
        """
        async for event in super().stream(
            messages, tool_specs, system_prompt, tool_choice=tool_choice, **kwargs
        ):
            yield event

    @staticmethod
    def _clean_response_content(content: str) -> str:
        """Clean MiniMax model output for structured parsing.

        MiniMax models may include:
        - Reasoning content wrapped in ``<think>`` tags
        - JSON wrapped in markdown code blocks (````json ... ````)

        This method strips those wrappers and returns only the meaningful content.

        Args:
            content: Raw model response that may contain think tags or code blocks.

        Returns:
            Cleaned content ready for JSON parsing.
        """
        # Strip <think>...</think> tags
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        # Strip markdown code blocks (```json ... ``` or ``` ... ```)
        content = re.sub(r"^```(?:json)?\s*\n?", "", content)
        content = re.sub(r"\n?```\s*$", "", content)

        return content.strip()

    @override
    async def structured_output(
        self, output_model: type[T], prompt: Messages, system_prompt: str | None = None, **kwargs: Any
    ) -> AsyncGenerator[dict[str, T | Any], None]:
        """Get structured output from the MiniMax model.

        Uses a regular chat completion with ``response_format`` set to ``json_object``
        instead of the beta parse API, since MiniMax models may include ``<think>`` tags
        that interfere with the beta parser.

        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Model events with the last being the structured output.

        Raises:
            ContextWindowOverflowException: If the input exceeds the model's context window.
            ModelThrottledException: If the request is throttled by MiniMax (rate limits).
        """
        request = self.format_request(prompt, system_prompt=system_prompt)
        request["stream"] = False
        request["response_format"] = {"type": "json_object"}

        # Remove stream_options for non-streaming request
        request.pop("stream_options", None)

        # Add schema hint as a user message so the model knows the expected format
        # (MiniMax only allows system messages at the beginning of the conversation)
        schema_hint = {
            "role": "user",
            "content": f"Respond with a JSON object matching this schema: {json.dumps(output_model.model_json_schema())}",
        }
        request["messages"].append(schema_hint)

        async with self._get_client() as client:
            try:
                response = await client.chat.completions.create(**request)
            except openai.BadRequestError as e:
                if hasattr(e, "code") and e.code == "context_length_exceeded":
                    raise ContextWindowOverflowException(str(e)) from e
                raise
            except openai.RateLimitError as e:
                raise ModelThrottledException(str(e)) from e

        content = response.choices[0].message.content or ""
        content = self._clean_response_content(content)

        try:
            parsed = output_model.model_validate_json(content)
            yield {"output": parsed}
        except Exception as e:
            raise ValueError(f"Failed to parse MiniMax response into {output_model.__name__}: {e}") from e
