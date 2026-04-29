"""Astraflow (UCloud ModelVerse) model provider.

Provides two regional endpoints as independent provider classes:

- ``AstraflowModel``   – Global node (US/CA): https://api-us-ca.umodelverse.ai/v1
- ``AstraflowCNModel`` – China node:          https://api.modelverse.cn/v1

Docs: https://www.umodelverse.ai/
"""

import logging
import os
from typing import Any

from typing_extensions import Unpack, override

from .openai import OpenAIModel

logger = logging.getLogger(__name__)

_ASTRAFLOW_BASE_URL = "https://api-us-ca.umodelverse.ai/v1"
_ASTRAFLOW_CN_BASE_URL = "https://api.modelverse.cn/v1"

_ASTRAFLOW_API_KEY_ENV = "ASTRAFLOW_API_KEY"
_ASTRAFLOW_CN_API_KEY_ENV = "ASTRAFLOW_CN_API_KEY"

_DEFAULT_MODEL_GLOBAL = "claude-3-5-haiku-20241022"
_DEFAULT_MODEL_CN = "deepseek-ai/DeepSeek-V3"


class AstraflowModel(OpenAIModel):
    """Astraflow model provider – Global node (US/CA).

    Connects to the Astraflow global endpoint at https://api-us-ca.umodelverse.ai/v1.
    Supports all models available on UCloud ModelVerse, including Claude, DeepSeek, GPT-4, and more.

    The API key is read from the ``ASTRAFLOW_API_KEY`` environment variable by default.

    Example::

        import os
        from strands import Agent
        from strands.models.astraflow import AstraflowModel

        os.environ["ASTRAFLOW_API_KEY"] = "<your-api-key>"

        model = AstraflowModel(model_id="claude-3-5-haiku-20241022")
        agent = Agent(model=model)
        agent("Hello!")
    """

    def __init__(
        self,
        client_args: dict[str, Any] | None = None,
        **model_config: Unpack[OpenAIModel.OpenAIConfig],
    ) -> None:
        """Initialize the Astraflow global provider.

        Args:
            client_args: Optional overrides for the underlying OpenAI client arguments
                (e.g., ``timeout``, ``max_retries``).  ``base_url`` and ``api_key`` are
                pre-configured from the environment and should not normally be overridden.
            **model_config: Configuration options forwarded to :class:`OpenAIModel`.
                ``model_id`` defaults to ``claude-3-5-haiku-20241022``.
        """
        if "model_id" not in model_config:
            model_config["model_id"] = _DEFAULT_MODEL_GLOBAL  # type: ignore[assignment]

        merged_client_args: dict[str, Any] = {
            "base_url": _ASTRAFLOW_BASE_URL,
            "api_key": os.environ.get(_ASTRAFLOW_API_KEY_ENV, "MISSING_ASTRAFLOW_API_KEY"),
        }
        if client_args:
            merged_client_args.update(client_args)

        logger.debug("base_url=<%s> model_id=<%s> | initializing AstraflowModel", _ASTRAFLOW_BASE_URL, model_config.get("model_id"))
        super().__init__(client_args=merged_client_args, **model_config)

    @override
    def update_config(self, **model_config: Unpack[OpenAIModel.OpenAIConfig]) -> None:  # type: ignore[override]
        """Update the Astraflow model configuration.

        Args:
            **model_config: Configuration overrides forwarded to :class:`OpenAIModel`.
        """
        super().update_config(**model_config)


class AstraflowCNModel(OpenAIModel):
    """Astraflow model provider – China node.

    Connects to the Astraflow China endpoint at https://api.modelverse.cn/v1.
    Supports all models available on UCloud ModelVerse China, including DeepSeek, Claude, and more.

    The API key is read from the ``ASTRAFLOW_CN_API_KEY`` environment variable by default.

    Example::

        import os
        from strands import Agent
        from strands.models.astraflow import AstraflowCNModel

        os.environ["ASTRAFLOW_CN_API_KEY"] = "<your-api-key>"

        model = AstraflowCNModel(model_id="deepseek-ai/DeepSeek-V3")
        agent = Agent(model=model)
        agent("你好！")
    """

    def __init__(
        self,
        client_args: dict[str, Any] | None = None,
        **model_config: Unpack[OpenAIModel.OpenAIConfig],
    ) -> None:
        """Initialize the Astraflow China provider.

        Args:
            client_args: Optional overrides for the underlying OpenAI client arguments
                (e.g., ``timeout``, ``max_retries``).  ``base_url`` and ``api_key`` are
                pre-configured from the environment and should not normally be overridden.
            **model_config: Configuration options forwarded to :class:`OpenAIModel`.
                ``model_id`` defaults to ``deepseek-ai/DeepSeek-V3``.
        """
        if "model_id" not in model_config:
            model_config["model_id"] = _DEFAULT_MODEL_CN  # type: ignore[assignment]

        merged_client_args: dict[str, Any] = {
            "base_url": _ASTRAFLOW_CN_BASE_URL,
            "api_key": os.environ.get(_ASTRAFLOW_CN_API_KEY_ENV, "MISSING_ASTRAFLOW_CN_API_KEY"),
        }
        if client_args:
            merged_client_args.update(client_args)

        logger.debug("base_url=<%s> model_id=<%s> | initializing AstraflowCNModel", _ASTRAFLOW_CN_BASE_URL, model_config.get("model_id"))
        super().__init__(client_args=merged_client_args, **model_config)

    @override
    def update_config(self, **model_config: Unpack[OpenAIModel.OpenAIConfig]) -> None:  # type: ignore[override]
        """Update the Astraflow CN model configuration.

        Args:
            **model_config: Configuration overrides forwarded to :class:`OpenAIModel`.
        """
        super().update_config(**model_config)
