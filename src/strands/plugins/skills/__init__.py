"""AgentSkills.io integration for Strands Agents.

This module provides the SkillsPlugin for integrating AgentSkills.io skills
into Strands agents. Skills enable progressive disclosure of instructions:
metadata is injected into the system prompt upfront, and full instructions
are loaded on demand via a tool.

Example Usage:
    ```python
    from strands import Agent
    from strands.plugins.skills import Skill, SkillsPlugin

    plugin = SkillsPlugin(skills=["./skills/pdf-processing"])
    agent = Agent(plugins=[plugin])
    ```
"""

from .loader import load_skill, load_skills
from .skill import Skill
from .skills_plugin import SkillsPlugin

__all__ = [
    "Skill",
    "SkillsPlugin",
    "load_skill",
    "load_skills",
]
