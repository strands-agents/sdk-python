"""Cohere model provider using OpenAI compatibility API.

- Docs: https://docs.cohere.com/docs/compatibility-api
"""

import logging
from typing import Any, Optional, Iterable, cast
from typing_extensions import TypedDict, override, Unpack

from .openai import OpenAIModel

logger = logging.getLogger(__name__)


class CohereModel(OpenAIModel):
    """Cohere model provider implementation using OpenAI compatibility API."""

    class CohereConfig(OpenAIModel.OpenAIConfig):
        """Configuration options for Cohere models."""
        pass

    def __init__(self, api_key: str, **model_config: Unpack[CohereConfig]):
        """Initialize Cohere provider instance.

        Args:
            api_key: Cohere API key.
            **model_config: Configuration options for the Cohere model.
        """
        client_args = {
            "base_url": "https://api.cohere.ai/compatibility/v1",
            "api_key": api_key,
        }
        super().__init__(client_args=client_args, **model_config)
