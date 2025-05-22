"""Model-related type definitions for the SDK."""

import abc
import logging
from typing import Any, Iterable, Optional

from ..content import Messages
from ..streaming import StreamEvent
from ..tools import ToolSpec

logger = logging.getLogger(__name__)


class Model(abc.ABC):
    """Abstract base class for AI model implementations.

    This class defines the interface for all model implementations in the Strands Agents SDK. It provides a
    standardized way to configure, format, and process requests for different AI model providers.
    """

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
    def format_request(
        self, messages: Messages, tool_specs: Optional[list[ToolSpec]] = None, system_prompt: Optional[str] = None
    ) -> Any:
        """Format a streaming request to the underlying model.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            The formatted request.
        """
        pass

    @abc.abstractmethod
    # pragma: no cover
    def format_chunk(self, event: Any) -> StreamEvent:
        """Format the model response events into standardized message chunks.

        Args:
            event: A response event from the model.

        Returns:
            The formatted chunk.
        """
        pass

    @abc.abstractmethod
    # pragma: no cover
    def stream(self, request: Any) -> Iterable[Any]:
        """Send the request to the model and get a streaming response.

        Args:
            request: The formatted request to send to the model.

        Returns:
            The model's response.

        Raises:
            ModelThrottledException: When the model service is throttling requests from the client.
        """
        pass

    def converse(
        self, messages: Messages, tool_specs: Optional[list[ToolSpec]] = None, system_prompt: Optional[str] = None
    ) -> Iterable[StreamEvent]:
        """Converse with the model.

        This method handles the full lifecycle of conversing with the model:
        1. Format the messages, tool specs, and configuration into a streaming request
        2. Send the request to the model
        3. Yield the formatted message chunks

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.

        Yields:
            Formatted message chunks from the model.

        Raises:
            ModelThrottledException: When the model service is throttling requests from the client.
        """
        logger.debug("formatting request")
        request = self.format_request(messages, tool_specs, system_prompt)

        logger.debug("invoking model")
        response = self.stream(request)

        logger.debug("got response from model")
        for event in response:
            yield self.format_chunk(event)

        logger.debug("finished streaming response from model")
