"""Agent-related type definitions for bidirectional streaming.

This module defines the types used for BidiAgent.
"""

from .events import BidiAudioInputEvent, BidiImageInputEvent, BidiTextInputEvent

type BidiAgentInput = str | BidiTextInputEvent | BidiAudioInputEvent | BidiImageInputEvent
