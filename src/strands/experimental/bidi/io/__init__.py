"""IO channel implementations for bidirectional streaming.

These IO implementations require additional dependencies that can be installed with:
    pip install strands-agents[bidi-io]
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .audio import BidiAudioIO
    from .text import BidiTextIO


def __getattr__(name: str) -> type:
    """Lazy import IO classes with helpful error messages if dependencies are missing."""
    if name == "BidiAudioIO":
        try:
            from .audio import BidiAudioIO

            return BidiAudioIO
        except ImportError as e:
            raise ImportError(
                "BidiAudioIO requires pyaudio. Install it with: pip install strands-agents[bidi-io]"
            ) from e
    elif name == "BidiTextIO":
        try:
            from .text import BidiTextIO

            return BidiTextIO
        except ImportError as e:
            raise ImportError(
                "BidiTextIO requires prompt_toolkit. Install it with: pip install strands-agents[bidi-io]"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BidiAudioIO", "BidiTextIO"]
