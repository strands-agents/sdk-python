"""Agent-related type definitions for bidirectional streaming.

This module defines the types used for BidiAgent.
"""

from typing import TypeAlias

from .events import BidiAudioInputEvent, BidiImageInputEvent, BidiTextInputEvent

BidiAgentInput: TypeAlias = str | BidiTextInputEvent | BidiAudioInputEvent | BidiImageInputEvent
