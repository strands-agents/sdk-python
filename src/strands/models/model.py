"""Abstract base class for Agent model providers."""

import abc
import logging
from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypedDict, TypeVar

from pydantic import BaseModel

from ..hooks.events import AfterInvocationEvent
from ..plugins.plugin import Plugin
from ..types.content import Messages, SystemContentBlock
from ..types.streaming import StreamEvent
from ..types.tools import ToolChoice, ToolSpec

if TYPE_CHECKING:
    from ..agent.agent import Agent

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class BaseModelConfig(TypedDict, total=False):
    """Base configuration shared by all model providers.

    Attributes:
        context_window_limit: Maximum context window size in tokens for the model.
            This value represents the total token capacity shared between input and output.
    """

    context_window_limit: int | None


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

    @property
    def context_window_limit(self) -> int | None:
        """Maximum context window size in tokens, or None if not configured."""
        config = self.get_config()
        return (
            config.get("context_window_limit")
            if isinstance(config, dict)
            else getattr(config, "context_window_limit", None)
        )

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
