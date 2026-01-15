"""Bidirectional model interfaces and implementations."""

from typing import Any

from .model import BidiModel, BidiModelTimeoutError

__all__ = [
    "BidiModel",
    "BidiModelTimeoutError",
    "NOVA_SONIC_V1_MODEL_ID",
    "NOVA_SONIC_V2_MODEL_ID",
]


def __getattr__(name: str) -> Any:
    """Lazy load bidi model implementations only when accessed.

    This defers the import of optional dependencies until actually needed.
    """
    if name == "BidiGeminiLiveModel":
        from .gemini_live import BidiGeminiLiveModel

        return BidiGeminiLiveModel
    if name == "BidiNovaSonicModel":
        from .nova_sonic import BidiNovaSonicModel

        return BidiNovaSonicModel
    if name == "BidiOpenAIRealtimeModel":
        from .openai_realtime import BidiOpenAIRealtimeModel

        return BidiOpenAIRealtimeModel
    if name == "NOVA_SONIC_V1_MODEL_ID":
        from .nova_sonic import NOVA_SONIC_V1_MODEL_ID

        return NOVA_SONIC_V1_MODEL_ID
    if name == "NOVA_SONIC_V2_MODEL_ID":
        from .nova_sonic import NOVA_SONIC_V2_MODEL_ID

        return NOVA_SONIC_V2_MODEL_ID
    raise AttributeError(f"cannot import name '{name}' from '{__name__}' ({__file__})")
