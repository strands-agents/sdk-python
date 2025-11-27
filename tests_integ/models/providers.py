"""
Aggregates all providers for testing all providers in one go.
"""

import os
from typing import Callable, Optional

import requests
from pytest import mark

from strands.models import BedrockModel, Model
from strands.models.anthropic import AnthropicModel
from strands.models.gemini import GeminiModel
from strands.models.litellm import LiteLLMModel
from strands.models.llamaapi import LlamaAPIModel
from strands.models.mistral import MistralModel
from strands.models.ollama import OllamaModel
from strands.models.openai import OpenAIModel
from strands.models.sap_genai_hub import SAPGenAIHubModel
from strands.models.writer import WriterModel


class ProviderInfo:
    """Provider-based info for providers that require an APIKey via environment variables."""

    def __init__(
        self,
        id: str,
        factory: Callable[[], Model],
        environment_variable: Optional[str] = None,
    ) -> None:
        self.id = id
        self.model_factory = factory
        self.mark = mark.skipif(
            environment_variable is not None and environment_variable not in os.environ,
            reason=f"{environment_variable} environment variable missing",
        )

    def create_model(self) -> Model:
        return self.model_factory()


class OllamaProviderInfo(ProviderInfo):
    """Special case ollama as it's dependent on the server being available."""

    def __init__(self):
        super().__init__(
            id="ollama",
            factory=lambda: OllamaModel(
                host="http://localhost:11434", model_id="llama3.3:70b"
            ),
        )

        is_server_available = False
        try:
            is_server_available = requests.get("http://localhost:11434").ok
        except requests.exceptions.ConnectionError:
            pass

        self.mark = mark.skipif(
            not is_server_available,
            reason="Local Ollama endpoint not available at localhost:11434",
        )


class SAPGenAIHubProviderInfo(ProviderInfo):
    """Special case SAP GenAI Hub as it requires AI Core credentials to be configured."""

    def __init__(self):
        super().__init__(
            id="sap_genai_hub",
            factory=lambda: SAPGenAIHubModel(model_id="amazon--nova-lite"),
        )

        credentials_available = self._check_sap_credentials_available()
        self.mark = mark.skipif(
            not credentials_available,
            reason="SAP AI Core credentials not available - configure service key or set environment variables",
        )

    def _check_sap_credentials_available(self) -> bool:
        """Check if SAP GenAI Hub credentials are available."""
        try:
            # Try to import the SAP GenAI Hub SDK
            from gen_ai_hub.proxy.native.amazon.clients import Session

            # Try to create a session - this will fail if credentials are missing
            session = Session()
            # Try to create a client - this is where it would fail with missing base_url
            client = session.client(model_name="amazon--nova-lite")
            # If we got this far, credentials are available
            return True
        except (ImportError, TypeError, Exception) as e:
            # Common errors when credentials are missing:
            # - TypeError: AICoreV2Client.__init__() missing 1 required positional argument: 'base_url'
            # - ImportError: No module named 'gen_ai_hub'
            # - Other configuration-related exceptions
            error_msg = str(e)
            if "missing 1 required positional argument: 'base_url'" in error_msg:
                # This is the specific error we expect when AI Core credentials are missing
                return False
            # For any other errors, also assume credentials not available
            return False


anthropic = ProviderInfo(
    id="anthropic",
    environment_variable="ANTHROPIC_API_KEY",
    factory=lambda: AnthropicModel(
        client_args={
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
        },
        model_id="claude-3-7-sonnet-20250219",
        max_tokens=512,
    ),
)
bedrock = ProviderInfo(id="bedrock", factory=lambda: BedrockModel())
cohere = ProviderInfo(
    id="cohere",
    environment_variable="COHERE_API_KEY",
    factory=lambda: OpenAIModel(
        client_args={
            "base_url": "https://api.cohere.com/compatibility/v1",
            "api_key": os.getenv("COHERE_API_KEY"),
        },
        model_id="command-a-03-2025",
        params={"stream_options": None},
    ),
)
litellm = ProviderInfo(
    id="litellm",
    factory=lambda: LiteLLMModel(
        model_id="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    ),
)
llama = ProviderInfo(
    id="llama",
    environment_variable="LLAMA_API_KEY",
    factory=lambda: LlamaAPIModel(
        model_id="Llama-4-Maverick-17B-128E-Instruct-FP8",
        client_args={
            "api_key": os.getenv("LLAMA_API_KEY"),
        },
    ),
)
mistral = ProviderInfo(
    id="mistral",
    environment_variable="MISTRAL_API_KEY",
    factory=lambda: MistralModel(
        model_id="mistral-medium-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
        stream=True,
        temperature=0.7,
        max_tokens=1000,
        top_p=0.9,
    ),
)
openai = ProviderInfo(
    id="openai",
    environment_variable="OPENAI_API_KEY",
    factory=lambda: OpenAIModel(
        model_id="gpt-4o",
        client_args={
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
    ),
)
writer = ProviderInfo(
    id="writer",
    environment_variable="WRITER_API_KEY",
    factory=lambda: WriterModel(
        model_id="palmyra-x4",
        client_args={"api_key": os.getenv("WRITER_API_KEY", "")},
        stream_options={"include_usage": True},
    ),
)
gemini = ProviderInfo(
    id="gemini",
    environment_variable="GOOGLE_API_KEY",
    factory=lambda: GeminiModel(
        client_args={"api_key": os.getenv("GOOGLE_API_KEY")},
        model_id="gemini-2.5-flash",
        params={"temperature": 0.7},
    ),
)

ollama = OllamaProviderInfo()
sap_genai_hub = SAPGenAIHubProviderInfo()


all_providers = [
    bedrock,
    anthropic,
    cohere,
    gemini,
    llama,
    litellm,
    mistral,
    openai,
    sap_genai_hub,
    writer,
]
