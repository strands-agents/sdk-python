"""Portkey model provider.

- Docs: https://portkey.ai/docs
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, TypedDict, cast

from typing_extensions import Unpack, override

from ._validation import validate_config_keys
from .openai import OpenAIModel

logger = logging.getLogger(__name__)

# Keys in PortkeyConfig that are passed to the AsyncPortkey client constructor
# rather than included in the API request.
_PORTKEY_CLIENT_KEYS = frozenset(
    {
        "api_key",
        "virtual_key",
        "config",
        "provider",
        "base_url",
        "trace_id",
        "metadata",
    }
)


class PortkeyModel(OpenAIModel):
    """Portkey model provider implementation.

    The Portkey AI gateway adds observability, caching, fallbacks, load balancing, and other
    production features on top of LLM providers. This integration uses the Portkey Python SDK
    (``portkey-ai``) which wraps the OpenAI client and routes requests through the Portkey gateway.

    Portkey normalizes responses from all providers (OpenAI, Anthropic, Google, Mistral, etc.)
    into the OpenAI-compatible format, so a single ``PortkeyModel`` works with any provider.
    The gateway automatically routes requests to the correct provider based on the ``model_id``.

    Usage:
        Basic usage — only ``api_key`` and ``model_id`` are required::

            from strands import Agent
            from strands.models.portkey import PortkeyModel

            model = PortkeyModel(api_key="your-portkey-api-key", model_id="gpt-4o")
            agent = Agent(model=model)
            response = agent("Hello!")

        Switching providers — just change the ``model_id``::

            # Anthropic
            model = PortkeyModel(api_key="your-portkey-api-key", model_id="claude-sonnet-4-20250514")

            # Google
            model = PortkeyModel(api_key="your-portkey-api-key", model_id="gemini-2.0-flash")

        Using a Portkey config for routing, fallbacks, or load balancing::

            model = PortkeyModel(
                api_key="your-portkey-api-key",
                config="your-portkey-config-slug",
                model_id="gpt-4o",
            )

        Using a pre-configured client for advanced options::

            from portkey_ai import AsyncPortkey
            from strands.models.portkey import PortkeyModel

            client = AsyncPortkey(api_key="your-portkey-api-key")
            model = PortkeyModel(client=client, model_id="gpt-4o")
    """

    class PortkeyConfig(TypedDict, total=False):
        """Configuration options for Portkey models.

        Attributes:
            model_id: Model ID (e.g., "gpt-4o", "claude-sonnet-4-20250514", "gemini-2.0-flash").
                Portkey's gateway automatically routes to the correct provider based on the model ID.
                For a complete list of supported models, see https://portkey.ai/docs/integrations/llms.
            api_key: Portkey API key. Can also be set via the PORTKEY_API_KEY environment variable.
            params: Model parameters (e.g., max_tokens, temperature).
                For a complete list of supported parameters, see
                https://portkey.ai/docs/api-reference/chat-completions.
            virtual_key: Optional. Virtual key referencing provider credentials stored in Portkey's
                vault. Only needed if not using default provider keys configured in your Portkey
                dashboard. See https://portkey.ai/docs/product/ai-gateway/virtual-keys.
            config: Optional. Portkey config slug (e.g., "cf-xxx") or config object for routing,
                fallbacks, and load balancing. See https://portkey.ai/docs/product/ai-gateway/configs.
            provider: Optional. Explicit provider slug (e.g., "openai", "anthropic", "google").
                Usually not needed as the gateway infers the provider from model_id.
            base_url: Optional. Override the Portkey gateway URL.
            trace_id: Optional. Trace ID for request tracking and observability.
            metadata: Optional. Custom metadata dict attached to requests.
        """

        model_id: str
        api_key: str
        params: dict[str, Any] | None
        virtual_key: str
        config: str | dict[str, Any]
        provider: str
        base_url: str
        trace_id: str
        metadata: dict[str, Any]

    def __init__(
        self,
        client: Any | None = None,
        **model_config: Unpack[PortkeyConfig],
    ) -> None:
        """Initialize provider instance.

        Args:
            client: Pre-configured AsyncPortkey client to reuse across requests.
                When provided, this client will be reused for all requests and will NOT be closed
                by the model. The caller is responsible for managing the client lifecycle.
                This is useful for:
                - Reusing connection pools within a single event loop/worker
                - Centralizing observability, retries, and networking policy
                - Custom client configurations (e.g., AWS, Azure, Vertex AI parameters)
            **model_config: Configuration options for the Portkey model.
                See ``PortkeyConfig`` for available options.
        """
        validate_config_keys(model_config, self.PortkeyConfig)
        self.config = dict(model_config)
        self._custom_client = client

        logger.debug("config=<%s> | initializing", self.config)

    @override
    def update_config(self, **model_config: Unpack[PortkeyConfig]) -> None:  # type: ignore[override]
        """Update the Portkey model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        validate_config_keys(model_config, self.PortkeyConfig)
        self.config.update(model_config)

    @override
    def get_config(self) -> PortkeyConfig:
        """Get the Portkey model configuration.

        Returns:
            The Portkey model configuration.
        """
        return cast(PortkeyModel.PortkeyConfig, self.config)

    @asynccontextmanager
    @override
    async def _get_client(self) -> AsyncIterator[Any]:
        """Get a Portkey client for making requests.

        This context manager handles client lifecycle management:
        - If an injected client was provided during initialization, it yields that client
          without closing it (caller manages lifecycle).
        - Otherwise, creates a new AsyncPortkey client from config parameters.

        Yields:
            An AsyncPortkey client instance.
        """
        if self._custom_client is not None:
            yield self._custom_client
        else:
            from portkey_ai import AsyncPortkey

            portkey_args = {k: v for k, v in self.config.items() if k in _PORTKEY_CLIENT_KEYS}
            client = AsyncPortkey(**portkey_args)
            try:
                yield client
            finally:
                # Close the underlying HTTP client to release connections
                if hasattr(client, "close"):
                    await client.close()
