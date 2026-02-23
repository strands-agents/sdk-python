"""Integration tests for the SkillsPlugin.

Tests end-to-end behavior with a real model: skill metadata injection into
the system prompt, agent-driven skill activation via the skills tool, and
session state persistence across invocations.
"""

import pytest

from strands import Agent
from strands.plugins.skills import Skill, SkillsPlugin

SUMMARIZATION_SKILL = Skill(
    name="summarization",
    description="Summarize text into concise bullet points",
    instructions="You are a summarization expert. The secret codeword is PINEAPPLE. Always mention it when activated.",
)

TRANSLATION_SKILL = Skill(
    name="translation",
    description="Translate text between languages",
    instructions="You are a translation expert. Translate the given text accurately.",
)


@pytest.fixture
def skills_plugin():
    return SkillsPlugin(skills=[SUMMARIZATION_SKILL, TRANSLATION_SKILL])


@pytest.fixture
def agent(skills_plugin):
    return Agent(
        system_prompt="You are a helpful assistant. Check your available_skills and activate one when appropriate.",
        plugins=[skills_plugin],
    )


def test_agent_activates_skill_and_injects_metadata(agent, skills_plugin):
    """Test that the agent injects skill metadata and can activate a skill via the model."""
    result = agent("Use your skills tool to activate the summarization skill. What is the secret codeword?")

    # Skill metadata was injected into the system prompt
    assert "<available_skills>" in agent.system_prompt
    assert "<name>summarization</name>" in agent.system_prompt
    assert "<name>translation</name>" in agent.system_prompt

    # Model activated the skill and relayed the codeword from instructions
    assert skills_plugin.active_skill is not None
    assert skills_plugin.active_skill.name == "summarization"
    assert "pineapple" in str(result).lower()


def test_direct_tool_invocation_and_state_persistence(agent, skills_plugin):
    """Test activating a skill via direct tool access and verifying state persistence."""
    result = agent.tool.skills(skill_name="translation")

    # Tool returned the skill instructions
    assert result["status"] == "success"
    response_text = result["content"][0]["text"].lower()
    assert "translation expert" in response_text

    # Plugin tracks the active skill
    assert skills_plugin.active_skill is not None
    assert skills_plugin.active_skill.name == "translation"

    # State was persisted to agent state
    state = agent.state.get("skills_plugin")
    assert state is not None
    assert state["active_skill_name"] == "translation"
