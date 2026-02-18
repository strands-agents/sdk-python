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
        input_rate: Input sample rate in Hz (e.g., 16000, 24000, 48000)
        output_rate: Output sample rate in Hz (e.g., 16000, 24000, 48000)
        channels: Number of audio channels (1=mono, 2=stereo)
        format: Audio encoding format
        voice: Voice identifier for text-to-speech (e.g., "alloy", "matthew")
    """

    input_rate: AudioSampleRate
    output_rate: AudioSampleRate
    channels: AudioChannel
    format: AudioFormat
    voice: str
