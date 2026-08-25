"""IO channel implementations for bidirectional streaming."""

from typing import Any

from .text import BidiTextIO

__all__ = ["AudioProcessorConfig", "BidiAudioIO", "BidiAudioIOConfig", "BidiTextIO"]


def __getattr__(name: str) -> Any:
    """Lazy load the audio IO implementation only when accessed."""
    if name == "AudioProcessorConfig":
        from .audio import AudioProcessorConfig

        return AudioProcessorConfig
    if name == "BidiAudioIO":
        from .audio import BidiAudioIO

        return BidiAudioIO
    if name == "BidiAudioIOConfig":
        from .audio import BidiAudioIOConfig

        return BidiAudioIOConfig
    raise AttributeError(f"cannot import name '{name}' from '{__name__}' ({__file__})")
