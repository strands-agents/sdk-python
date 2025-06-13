"""Tests for the A2AAgent class."""

import pytest
from a2a.types import AgentCard
from fastapi import FastAPI
from starlette.applications import Starlette

from strands import Agent
from strands.multiagent.a2a import A2AAgent


@pytest.fixture
def strands_agent():
    """Create a Strands agent for testing."""
    return Agent()


@pytest.fixture
def a2a_agent(strands_agent):
    """Create an A2A agent for testing."""
    return A2AAgent(
        agent=strands_agent,
        name="Test Agent",
        description="A test agent",
        host="localhost",
        port=9000,
    )


def test_a2a_agent_initialization(a2a_agent, strands_agent):
    """Test that the A2AAgent initializes correctly."""
    assert a2a_agent.name == "Test Agent"
    assert a2a_agent.description == "A test agent"
    assert a2a_agent.host == "localhost"
    assert a2a_agent.port == 9000
    assert a2a_agent.http_url == "http://localhost:9000/"
    assert a2a_agent.version == "0.0.1"
    assert a2a_agent.strands_agent == strands_agent


def test_public_agent_card(a2a_agent):
    """Test that the public agent card is created correctly."""
    card = a2a_agent.public_agent_card
    assert isinstance(card, AgentCard)
    assert card.name == "Test Agent"
    assert card.description == "A test agent"
    assert card.url == "http://localhost:9000/"
    assert card.version == "0.0.1"
    assert card.defaultInputModes == ["text"]
    assert card.defaultOutputModes == ["text"]
    assert len(card.skills) == 0  # No skills defined yet


def test_agent_skills(a2a_agent):
    """Test that agent skills are returned correctly."""
    skills = a2a_agent.agent_skills
    assert isinstance(skills, list)
    assert len(skills) == 0  # No skills defined yet


def test_to_starlette_app(a2a_agent):
    """Test that a Starlette app is created correctly."""
    app = a2a_agent.to_starlette_app()
    assert isinstance(app, Starlette)


def test_to_fastapi_app(a2a_agent):
    """Test that a FastAPI app is created correctly."""
    app = a2a_agent.to_fastapi_app()
    assert isinstance(app, FastAPI)
