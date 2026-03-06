"""AgentSkills.io integration for Strands Agents.

This module provides the AgentSkills plugin for integrating AgentSkills.io skills
into Strands agents. Skills enable progressive disclosure of instructions:
metadata is injected into the system prompt upfront, and full instructions
are loaded on demand via a tool.

Example Usage:
    ```python
    from strands import Agent
    from strands.plugins.skills import Skill, AgentSkills

    plugin = AgentSkills(skills=["./skills/pdf-processing"])
    agent = Agent(plugins=[plugin])
    ```
"""

from .agent_skills import AgentSkills
from .loader import load_skill, load_skills
from .skill import Skill

__all__ = [
    "AgentSkills",
    "Skill",
    "load_skill",
    "load_skills",
]
