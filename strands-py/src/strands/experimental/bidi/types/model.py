"""Model-related type definitions for bidirectional streaming.

Defines types and configurations that are central to model providers,
including audio configuration that models use to specify their audio
processing requirements.
"""

from typing import TypedDict

from .events import AudioChannel, AudioFormat, AudioSampleRate


class AudioConfig(TypedDict, total=False):
    """Audio configuration for bidirectional streaming models.

    Defines standard audio parameters that model providers use to specify
    their audio processing requirements. All fields are optional to support
    models that may not use audio or only need specific parameters.

    Model providers build this configuration by merging user-provided values
    with their own defaults. The resulting configuration is then used by
    audio I/O implementations to configure hardware appropriately.

    Attributes:
        input_rate: Input sample rate in Hz (e.g., 8000, 16000, 24000, 48000)
        output_rate: Output sample rate in Hz (e.g., 8000, 16000, 24000, 48000)
        channels: Number of audio channels (1=mono, 2=stereo)
        format: Audio encoding format
        voice: Voice identifier for text-to-speech (e.g., "alloy", "matthew")
    """

    input_rate: AudioSampleRate
    output_rate: AudioSampleRate
    channels: AudioChannel
    format: AudioFormat
    voice: str


class BidiConnectionConfig(TypedDict, total=False):
    """Declared reconnect timing for a bidirectional model.

    Providers declare this so the agent loop can reconnect proactively, before the provider
    terminates the connection on its own limit. A provider that declares nothing (empty config)
    keeps reactive-only behavior: no proactive timer, reconnect only after the provider reports
    a timeout.

    All fields are optional. The proactive timer arms only when ``restart_after_s`` is declared.

    Attributes:
        restart_after_s: Seconds after a connection is established at which to proactively
            reconnect. Set it at least ~10s below the provider's own connection limit: the
            reconnect may wait briefly for the current turn to finish (aligning the swap to a
            turn boundary), and that wait plus the swap must complete before the provider's limit.
        auto_reconnect: Whether the loop reconnects automatically (default True).
    """

    restart_after_s: int
    auto_reconnect: bool
