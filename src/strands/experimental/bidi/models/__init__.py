"""Bidirectional model interfaces and implementations."""

from typing import Any

from .model import BidiModel, BidiModelTimeoutError
from .nova_sonic import BidiNovaSonicModel

__all__ = [
    "BidiModel",
    "BidiModelTimeoutError",
    "BidiNovaSonicModel",
]


def __getattr__(name: str) -> Any:
    """
    Lazy load bidi model implementations only when accessed.
    
    This defers the import of optional dependencies until actually needed:
    - BidiGeminiLiveModel requires google-generativeai (lazy loaded)
    - BidiOpenAIRealtimeModel requires openai (lazy loaded)
    """
    if name == "BidiGeminiLiveModel":
        from .gemini_live import BidiGeminiLiveModel

        return BidiGeminiLiveModel
    if name == "BidiOpenAIRealtimeModel":
        from .openai_realtime import BidiOpenAIRealtimeModel

        return BidiOpenAIRealtimeModel
    raise AttributeError(f"cannot import name '{name}' from '{__name__}' ({__file__})")
