"""IO channel implementations for bidirectional streaming."""

from typing import Any

from .text import BidiTextIO

__all__ = ["BidiAudioIO", "BidiTextIO"]


def __getattr__(name: str) -> Any:
    """Lazy load the audio IO implementation only when accessed."""
    if name == "BidiAudioIO":
        from .audio import BidiAudioIO

        return BidiAudioIO
    raise AttributeError(f"cannot import name '{name}' from '{__name__}' ({__file__})")
