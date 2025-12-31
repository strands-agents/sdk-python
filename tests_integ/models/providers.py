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
from strands.models.sglang import SGLangModel
from strands.models.vllm import VLLMModel
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
            id="ollama", factory=lambda: OllamaModel(host="http://localhost:11434", model_id="llama3.3:70b")
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


class VLLMProviderInfo(ProviderInfo):
    """Special case vLLM as it's dependent on the server being available."""

    def __init__(self):
        super().__init__(
            id="vllm",
            factory=lambda: VLLMModel(
                client_args={
                    "api_key": "EMPTY",
                    "base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
                },
                model_id=os.getenv("VLLM_MODEL_ID", "AMead10/Llama-3.2-3B-Instruct-AWQ"),
                params={"max_tokens": 64, "temperature": 0},
                return_token_ids=True,
            ),
        )

        base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
        is_server_available = False
        try:
            # OpenAI-compatible discovery endpoint.
            is_server_available = requests.get(f"{base_url}/models", timeout=2).ok
        except requests.exceptions.RequestException:
            pass

        self.mark = mark.skipif(
            not is_server_available,
            reason=f"Local vLLM endpoint not available at {base_url}",
        )


class SGLangProviderInfo(ProviderInfo):
    """Special case SGLang as it's dependent on the server being available."""

    def __init__(self):
        super().__init__(
            id="sglang",
            factory=lambda: SGLangModel(
                base_url=os.getenv("SGLANG_BASE_URL", "http://localhost:30000"),
                model_id=os.getenv("SGLANG_MODEL_ID"),
                params={"max_new_tokens": 64, "temperature": 0},
                return_token_ids=True,
            ),
        )

        base_url = os.getenv("SGLANG_BASE_URL", "http://localhost:30000").rstrip("/")
        is_server_available = False
        try:
            is_server_available = requests.get(f"{base_url}/health", timeout=2).ok
        except requests.exceptions.RequestException:
            pass

        self.mark = mark.skipif(
            not is_server_available,
            reason=f"Local SGLang endpoint not available at {base_url}",
        )


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
    id="litellm", factory=lambda: LiteLLMModel(model_id="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0")
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
vllm = VLLMProviderInfo()
sglang = SGLangProviderInfo()


all_providers = [
    bedrock,
    anthropic,
    cohere,
    gemini,
    llama,
    litellm,
    mistral,
    openai,
    writer,
]
