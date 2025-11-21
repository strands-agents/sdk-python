"""Multi-agent related type definitions for the SDK."""

from typing import TypeAlias

from .content import ContentBlock

MultiAgentInput: TypeAlias = str | list[ContentBlock]
