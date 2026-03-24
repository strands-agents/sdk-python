"""Novita AI model provider.

- Docs: https://novita.ai/docs/guides/introduction.html
"""

import os
from typing import Any, TypedDict, cast
from typing_extensions import Unpack, override

from .openai import OpenAIModel

NOVITA_API_KEY_ENV_VAR = "NOVITA_API_KEY"
NOVITA_BASE_URL = "https://api.novita.ai/openai"
NOVITA_DEFAULT_MODEL_ID = "moonshotai/kimi-k2.5"


class NovitaModel(OpenAIModel):
    """Novita AI model provider implementation.

    Novita AI provides an OpenAI-compatible API endpoint. This provider extends
    OpenAIModel with Novita-specific defaults:

    - Base URL: https://api.novita.ai/openai
    - API Key: NOVITA_API_KEY environment variable
    - Default Model: moonshotai/kimi-k2.5

    Available models include:
    - moonshotai/kimi-k2.5 (default) - MoE, function calling, structured output, reasoning, vision
    - zai-org/glm-5 - MoE, function calling, structured output, reasoning
    - minimax/minimax-m2.5 - MoE, function calling, structured output, reasoning

    For a complete list of supported models, see https://novita.ai/docs/guides/models.html
    """

    class NovitaConfig(TypedDict, total=False):
        """Configuration options for Novita AI models.

        Attributes:
            model_id: Novita model ID (e.g., "moonshotai/kimi-k2.5").
                For a complete list of supported models, see https://novita.ai/docs/guides/models.html
            params: Model parameters (e.g., max_tokens, temperature).
                For a complete list of supported parameters, see
                https://platform.openai.com/docs/api-reference/chat/create.
        """

        model_id: str
        params: dict[str, Any] | None

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        client_args: dict[str, Any] | None = None,
        **model_config: Unpack[NovitaConfig],
    ) -> None:
        """Initialize Novita AI provider instance.

        Args:
            api_key: Novita AI API key. If not provided, will use NOVITA_API_KEY
                environment variable.
            base_url: Novita AI API base URL. Defaults to https://api.novita.ai/openai.
            client: Pre-configured OpenAI-compatible client to reuse across requests.
                When provided, this client will be reused for all requests and will NOT be closed
                by the model. The caller is responsible for managing the client lifecycle.
            client_args: Additional arguments for the OpenAI client.
                For a complete list of supported arguments, see https://pypi.org/project/openai/.
            **model_config: Configuration options for the Novita AI model.

        Raises:
            ValueError: If API key is not provided and NOVITA_API_KEY environment variable
                is not set.
        """
        # Set default model_id if not provided
        if "model_id" not in model_config:
            model_config = cast(NovitaModel.NovitaConfig, {**model_config, "model_id": NOVITA_DEFAULT_MODEL_ID})

        # Build client_args with Novita-specific settings
        novita_client_args: dict[str, Any] = {}

        # Determine API key
        resolved_api_key = api_key or os.environ.get(NOVITA_API_KEY_ENV_VAR)
        if resolved_api_key:
            novita_client_args["api_key"] = resolved_api_key

        # Set base URL for Novita
        novita_client_args["base_url"] = base_url or NOVITA_BASE_URL

        # Merge with user-provided client_args
        if client_args:
            novita_client_args.update(client_args)

        super().__init__(
            client=client,
            client_args=novita_client_args,
            **model_config,
        )

    @override
    def get_config(self) -> NovitaConfig:
        """Get the Novita AI model configuration.

        Returns:
            The Novita AI model configuration.
        """
        return cast(NovitaModel.NovitaConfig, self.config)
