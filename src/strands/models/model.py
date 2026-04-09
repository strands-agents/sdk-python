"""Abstract base class for Agent model providers."""

import abc
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic import BaseModel

from ..hooks.events import AfterInvocationEvent
from ..plugins.plugin import Plugin
from ..types.content import ContentBlock, Messages, SystemContentBlock
from ..types.streaming import StreamEvent
from ..types.tools import ToolChoice, ToolSpec

if TYPE_CHECKING:
    from ..agent.agent import Agent

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_DEFAULT_ENCODING = "cl100k_base"
_cached_encoding: Any = None


def _get_encoding() -> Any:
    """Get the default tiktoken encoding, caching to avoid repeated lookups."""
    global _cached_encoding
    if _cached_encoding is None:
        try:
            import tiktoken
        except ImportError as err:
            raise ImportError(
                "tiktoken is required for token estimation. "
                "Install it with: pip install strands-agents[token-estimation]"
            ) from err
        _cached_encoding = tiktoken.get_encoding(_DEFAULT_ENCODING)
    return _cached_encoding


def _count_content_block_tokens(block: ContentBlock, encoding: Any) -> int:
    """Count tokens for a single content block."""
    total = 0

    if "text" in block:
        total += len(encoding.encode(block["text"]))

    if "toolUse" in block:
        tool_use = block["toolUse"]
        total += len(encoding.encode(tool_use.get("name", "")))
        try:
            total += len(encoding.encode(json.dumps(tool_use.get("input", {}))))
        except (TypeError, ValueError):
            logger.debug(
                "tool_name=<%s> | skipping non-serializable toolUse input for token estimation",
                tool_use.get("name", "unknown"),
            )

    if "toolResult" in block:
        tool_result = block["toolResult"]
        for item in tool_result.get("content", []):
            if "text" in item:
                total += len(encoding.encode(item["text"]))

    if "reasoningContent" in block:
        reasoning = block["reasoningContent"]
        if "reasoningText" in reasoning:
            reasoning_text = reasoning["reasoningText"]
            if "text" in reasoning_text:
                total += len(encoding.encode(reasoning_text["text"]))

    if "guardContent" in block:
        guard = block["guardContent"]
        if "text" in guard and "text" in guard["text"]:
            total += len(encoding.encode(guard["text"]["text"]))

    if "citationsContent" in block:
        citations = block["citationsContent"]
        if "content" in citations:
            for citation_item in citations["content"]:
                if "text" in citation_item:
                    total += len(encoding.encode(citation_item["text"]))

    return total


def _estimate_tokens_with_tiktoken(
    messages: Messages,
    tool_specs: list[ToolSpec] | None = None,
    system_prompt: str | None = None,
    system_prompt_content: list[SystemContentBlock] | None = None,
) -> int:
    """Estimate tokens by serializing messages/tools to text and counting with tiktoken.

    This is a best-effort fallback for providers that don't expose native counting.
    Accuracy varies by model but is sufficient for threshold-based decisions.
    """
    encoding = _get_encoding()
    total = 0

    # Prefer system_prompt_content (structured) over system_prompt (plain string) to avoid double-counting,
    # since providers wrap system_prompt into system_prompt_content when both are provided.
    if system_prompt_content:
        for block in system_prompt_content:
            if "text" in block:
                total += len(encoding.encode(block["text"]))
    elif system_prompt:
        total += len(encoding.encode(system_prompt))

    for message in messages:
        for block in message["content"]:
            total += _count_content_block_tokens(block, encoding)

    if tool_specs:
        for spec in tool_specs:
            try:
                total += len(encoding.encode(json.dumps(spec)))
            except (TypeError, ValueError):
                logger.debug(
                    "tool_name=<%s> | skipping non-serializable tool spec for token estimation",
                    spec.get("name", "unknown"),
                )

    return total


@dataclass
class CacheConfig:
    """Configuration for prompt caching.

    Attributes:
        strategy: Caching strategy to use.
            - "auto": Automatically detect model support and inject cachePoint to maximize cache coverage
            - "anthropic": Inject cachePoint in Anthropic-compatible format without model support check
    """

    strategy: Literal["auto", "anthropic"] = "auto"


class Model(abc.ABC):
    """Abstract base class for Agent model providers.

    This class defines the interface for all model implementations in the Strands Agents SDK. It provides a
    standardized way to configure and process requests for different AI model providers.
    """

    @property
    def stateful(self) -> bool:
        """Whether the model manages conversation state server-side.

        Returns:
            False by default. Model providers that support server-side state should override this.
        """
        return False

    @abc.abstractmethod
    # pragma: no cover
    def update_config(self, **model_config: Any) -> None:
        """Update the model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        pass

    @abc.abstractmethod
    # pragma: no cover
    def get_config(self) -> Any:
        """Return the model configuration.

        Returns:
            The model's configuration.
        """
        pass

    @abc.abstractmethod
    # pragma: no cover
    def structured_output(
        self, output_model: type[T], prompt: Messages, system_prompt: str | None = None, **kwargs: Any
    ) -> AsyncGenerator[dict[str, T | Any], None]:
        """Get structured output from the model.

        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Model events with the last being the structured output.

        Raises:
            ValidationException: The response format from the model does not match the output_model
        """
        pass

    @abc.abstractmethod
    # pragma: no cover
    def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        invocation_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """Stream conversation with the model.

        This method handles the full lifecycle of conversing with the model:

        1. Format the messages, tool specs, and configuration into a streaming request
        2. Send the request to the model
        3. Yield the formatted message chunks

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            tool_choice: Selection strategy for tool invocation.
            system_prompt_content: System prompt content blocks for advanced features like caching.
            invocation_state: Caller-provided state/context that was passed to the agent when it was invoked.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Formatted message chunks from the model.

        Raises:
            ModelThrottledException: When the model service is throttling requests from the client.
        """
        pass

    def _estimate_tokens(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
    ) -> int:
        """Estimate token count for the given input before sending to the model.

        Used for proactive context management (e.g., triggering compression at a
        threshold). This is a naive approximation using tiktoken's cl100k_base encoding.
        Accuracy varies by model provider but is estimated to be within 5-15% for most providers.
        Not intended for billing or precise quota calculations.

        Subclasses may override this method to provide model-specific token counting
        using native APIs for improved accuracy.

        Args:
            messages: List of message objects to estimate tokens for.
            tool_specs: List of tool specifications to include in the estimate.
            system_prompt: Plain string system prompt. Ignored if system_prompt_content is provided.
            system_prompt_content: Structured system prompt content blocks. Takes priority over system_prompt.

        Returns:
            Estimated total input tokens.
        """
        return _estimate_tokens_with_tiktoken(messages, tool_specs, system_prompt, system_prompt_content)


class _ModelPlugin(Plugin):
    """Plugin that manages model-related lifecycle hooks."""

    @property
    def name(self) -> str:
        """A stable string identifier for this plugin."""
        return "strands:model"

    @staticmethod
    def _on_after_invocation(event: AfterInvocationEvent) -> None:
        """Handle post-invocation model management tasks.

        Performs the following:
        - Clears messages when the model is managing conversation state server-side.
        """
        if event.agent.model.stateful:
            event.agent.messages.clear()
            logger.debug(
                "response_id=<%s> | cleared messages for server-managed conversation",
                event.agent._model_state.get("response_id"),
            )

    def init_agent(self, agent: "Agent") -> None:
        """Register model lifecycle hooks with the agent.

        Args:
            agent: The agent instance to register hooks with.
        """
        agent.add_hook(self._on_after_invocation, AfterInvocationEvent)
