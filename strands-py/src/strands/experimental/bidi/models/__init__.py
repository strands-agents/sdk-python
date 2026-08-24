"""Bidirectional model interfaces and implementations."""

from typing import Any

from .model import BidiModel, BidiModelTimeoutError

__all__ = [
    "BidiModel",
    "BidiModelTimeoutError",
]


def __getattr__(name: str) -> Any:
    """Lazy load bidi model implementations only when accessed.

    This defers the import of optional dependencies until actually needed.
    """
    if name == "BedrockNovaSonicModel":
        from .bedrock import BedrockNovaSonicModel

        return BedrockNovaSonicModel
    if name == "GoogleLiveModel":
        from .google import GoogleLiveModel

        return GoogleLiveModel
    if name == "GoogleModel":
        from .google import GoogleModel

        return GoogleModel
    if name == "OpenAIRealtimeModel":
        from .openai import OpenAIRealtimeModel

        return OpenAIRealtimeModel
    if name == "OpenAIModel":
        from .openai import OpenAIModel

        return OpenAIModel
    raise AttributeError(f"cannot import name '{name}' from '{__name__}' ({__file__})")
