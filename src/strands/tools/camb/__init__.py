"""CAMB.AI tools for Strands Agents.

Provides 9 audio/speech tools powered by camb.ai:
- Text-to-Speech (TTS)
- Translation
- Transcription
- Translated TTS
- Voice Cloning
- Voice Listing
- Text-to-Sound generation
- Audio Separation
- Voice from Description

Usage::

    from strands import Agent
    from strands.tools.camb import CambAIToolProvider

    provider = CambAIToolProvider(api_key="your-api-key")
    agent = Agent(tools=[provider])
"""

from .camb_tools import CambAIToolProvider

__all__ = ["CambAIToolProvider"]
