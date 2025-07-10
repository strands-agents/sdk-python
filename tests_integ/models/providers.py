import os
from dataclasses import dataclass

import requests
from pytest import mark


@dataclass
class ApiKeyProviderInfo:
    """Provider-based info for providers that require an APIKey via environment variables."""

    def __init__(self, id: str, environment_variable: str) -> None:
        self.id = id
        self.environment_variable = environment_variable
        self.mark = mark.skipif(
            self.environment_variable not in os.environ,
            reason=f"{self.environment_variable} environment variable missing",
        )


class OllamaProviderInfo:
    """Special case ollama as it's dependent on the server being available."""

    def __init__(self):
        self.id = "ollama"

        is_server_available = False
        try:
            is_server_available = requests.get("http://localhost:11434").ok
        except requests.exceptions.ConnectionError:
            pass

        self.mark = mark.skipif(
            not is_server_available,
            reason="Local Ollama endpoint not available at localhost:11434",
        )


anthropic = ApiKeyProviderInfo(id="anthropic", environment_variable="ANTHROPIC_API_KEY")
cohere = ApiKeyProviderInfo(id="cohere", environment_variable="CO_API_KEY")
llama = ApiKeyProviderInfo(id="cohere", environment_variable="LLAMA_API_KEY")
mistral = ApiKeyProviderInfo(id="mistral", environment_variable="MISTRAL_API_KEY")
openai = ApiKeyProviderInfo(id="openai", environment_variable="OPENAI_API_KEY")
writer = ApiKeyProviderInfo(id="writer", environment_variable="WRITER_API_KEY")

ollama = OllamaProviderInfo()
