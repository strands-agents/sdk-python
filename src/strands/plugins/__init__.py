"""Plugin system for extending agent functionality.

This module provides a composable mechanism for building objects that can
extend agent behavior through automatic hook and tool registration.
"""

from .decorator import hook
from .plugin import Plugin
from .skills import AgentSkills, Skill

__all__ = [
    "AgentSkills",
    "Plugin",
    "Skill",
    "hook",
]
