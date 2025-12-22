"""Bidirectional model interfaces and implementations."""

from typing import TYPE_CHECKING

from .model import BidiModel, BidiModelTimeoutError
from .nova_sonic import BidiNovaSonicModel

# Type checking imports for static analysis
if TYPE_CHECKING:
    from .gemini_live import BidiGeminiLiveModel
    from .openai_realtime import BidiOpenAIRealtimeModel

__all__ = [
    "BidiModel",
    "BidiModelTimeoutError",
    "BidiNovaSonicModel",
    "BidiGeminiLiveModel",
    "BidiOpenAIRealtimeModel",
]


def __getattr__(name: str):
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
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
