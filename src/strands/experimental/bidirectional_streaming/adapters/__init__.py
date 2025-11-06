"""Adapters for BidirectionalAgent.

Provides clean separation of concerns by moving hardware-specific functionality
(audio, video, sensors, etc.) into separate adapter classes that work with
the core BidirectionalAgent through the run() pattern.
"""

from .audio_adapter import AudioAdapter

__all__ = ["AudioAdapter"]
