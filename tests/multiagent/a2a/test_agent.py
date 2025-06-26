"""Tests for the A2AAgent class."""

from unittest.mock import patch

import pytest
from a2a.types import AgentCapabilities, AgentCard
from fastapi import FastAPI
from starlette.applications import Starlette

from strands.multiagent.a2a.agent import A2AAgent


def test_a2a_agent_initialization(mock_strands_agent):
    """Test that A2AAgent initializes correctly with default values."""
    a2a_agent = A2AAgent(mock_strands_agent)

    assert a2a_agent.strands_agent == mock_strands_agent
    assert a2a_agent.name == "Test Agent"
    assert a2a_agent.description == "A test agent for unit testing"
    assert a2a_agent.host == "0.0.0"
    assert a2a_agent.port == 9000
    assert a2a_agent.http_url == "http://0.0.0:9000/"
    assert a2a_agent.version == "0.0.1"
    assert isinstance(a2a_agent.capabilities, AgentCapabilities)


def test_a2a_agent_initialization_with_custom_values(mock_strands_agent):
    """Test that A2AAgent initializes correctly with custom values."""
    a2a_agent = A2AAgent(
        mock_strands_agent,
        host="127.0.0.1",
        port=8080,
        version="1.0.0",
    )

    assert a2a_agent.host == "127.0.0.1"
    assert a2a_agent.port == 8080
    assert a2a_agent.http_url == "http://127.0.0.1:8080/"
    assert a2a_agent.version == "1.0.0"


def test_public_agent_card(mock_strands_agent):
    """Test that public_agent_card returns a valid AgentCard."""
    a2a_agent = A2AAgent(mock_strands_agent)

    card = a2a_agent.public_agent_card

    assert isinstance(card, AgentCard)
    assert card.name == "Test Agent"
    assert card.description == "A test agent for unit testing"
    assert card.url == "http://0.0.0:9000/"
    assert card.version == "0.0.1"
    assert card.defaultInputModes == ["text"]
    assert card.defaultOutputModes == ["text"]
    assert card.skills == []
    assert card.capabilities == a2a_agent.capabilities


def test_public_agent_card_with_missing_name(mock_strands_agent):
    """Test that public_agent_card raises ValueError when name is missing."""
    mock_strands_agent.name = ""
    a2a_agent = A2AAgent(mock_strands_agent)

    with pytest.raises(ValueError, match="A2A agent name cannot be None or empty"):
        _ = a2a_agent.public_agent_card


def test_public_agent_card_with_missing_description(mock_strands_agent):
    """Test that public_agent_card raises ValueError when description is missing."""
    mock_strands_agent.description = ""
    a2a_agent = A2AAgent(mock_strands_agent)

    with pytest.raises(ValueError, match="A2A agent description cannot be None or empty"):
        _ = a2a_agent.public_agent_card


def test_agent_skills(mock_strands_agent):
    """Test that agent_skills returns an empty list (current implementation)."""
    a2a_agent = A2AAgent(mock_strands_agent)

    skills = a2a_agent.agent_skills

    assert isinstance(skills, list)
    assert len(skills) == 0


def test_to_starlette_app(mock_strands_agent):
    """Test that to_starlette_app returns a Starlette application."""
    a2a_agent = A2AAgent(mock_strands_agent)

    app = a2a_agent.to_starlette_app()

    assert isinstance(app, Starlette)


def test_to_fastapi_app(mock_strands_agent):
    """Test that to_fastapi_app returns a FastAPI application."""
    a2a_agent = A2AAgent(mock_strands_agent)

    app = a2a_agent.to_fastapi_app()

    assert isinstance(app, FastAPI)


@patch("uvicorn.run")
def test_serve_with_starlette(mock_run, mock_strands_agent):
    """Test that serve starts a Starlette server by default."""
    a2a_agent = A2AAgent(mock_strands_agent)

    a2a_agent.serve()

    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    assert isinstance(args[0], Starlette)
    assert kwargs["host"] == "0.0.0"
    assert kwargs["port"] == 9000


@patch("uvicorn.run")
def test_serve_with_fastapi(mock_run, mock_strands_agent):
    """Test that serve starts a FastAPI server when specified."""
    a2a_agent = A2AAgent(mock_strands_agent)

    a2a_agent.serve(app_type="fastapi")

    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    assert isinstance(args[0], FastAPI)
    assert kwargs["host"] == "0.0.0"
    assert kwargs["port"] == 9000


@patch("uvicorn.run")
def test_serve_with_custom_kwargs(mock_run, mock_strands_agent):
    """Test that serve passes additional kwargs to uvicorn.run."""
    a2a_agent = A2AAgent(mock_strands_agent)

    a2a_agent.serve(log_level="debug", reload=True)

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["log_level"] == "debug"
    assert kwargs["reload"] is True


@patch("uvicorn.run", side_effect=KeyboardInterrupt)
def test_serve_handles_keyboard_interrupt(mock_run, mock_strands_agent, caplog):
    """Test that serve handles KeyboardInterrupt gracefully."""
    a2a_agent = A2AAgent(mock_strands_agent)

    a2a_agent.serve()

    assert "Strands A2A server shutdown requested (KeyboardInterrupt)" in caplog.text
    assert "Strands A2A server has shutdown" in caplog.text


@patch("uvicorn.run", side_effect=Exception("Test exception"))
def test_serve_handles_general_exception(mock_run, mock_strands_agent, caplog):
    """Test that serve handles general exceptions gracefully."""
    a2a_agent = A2AAgent(mock_strands_agent)

    a2a_agent.serve()

    assert "Strands A2A server encountered exception" in caplog.text
    assert "Strands A2A server has shutdown" in caplog.text
