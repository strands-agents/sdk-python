"""Tests for AgentDelegationException.

This module tests the exception that enables clean agent delegation control flow.
"""

import pytest
from typing import Any

from strands.types.exceptions import AgentDelegationException


class TestAgentDelegationException:
    """Test AgentDelegationException functionality."""

    def test_initialization(self):
        """Test exception with all parameters."""
        exc = AgentDelegationException(
            target_agent="SubAgent",
            message="Test msg",
            context={"key": "value"},
            delegation_chain=["Orchestrator"],
            transfer_state=False,
            transfer_messages=True
        )
        assert exc.target_agent == "SubAgent"
        assert exc.message == "Test msg"
        assert exc.context == {"key": "value"}
        assert exc.delegation_chain == ["Orchestrator"]
        assert exc.transfer_state is False
        assert exc.transfer_messages is True

    def test_chain_appending(self):
        """Test delegation chain updates."""
        exc = AgentDelegationException(
            target_agent="B",
            message="Test",
            delegation_chain=["A"]
        )
        exc.delegation_chain.append("B")
        assert exc.delegation_chain == ["A", "B"]

    def test_default_values(self):
        """Test default parameter values."""
        exc = AgentDelegationException(
            target_agent="Agent",
            message="Test"
        )
        assert exc.context == {}
        assert exc.delegation_chain == []
        assert exc.transfer_state is True
        assert exc.transfer_messages is True

    def test_exception_message_format(self):
        """Test exception string representation."""
        exc = AgentDelegationException(
            target_agent="TestAgent",
            message="Delegation message"
        )
        assert str(exc) == "Delegating to agent: TestAgent"

    def test_context_isolation(self):
        """Test context dict is properly isolated."""
        original_context = {"data": [1, 2, 3]}
        exc = AgentDelegationException(
            target_agent="Agent",
            message="Test",
            context=original_context
        )

        # Modify original context
        original_context["new_key"] = "new_value"

        # Exception context should be unchanged
        assert exc.context == {"data": [1, 2, 3]}
        assert "new_key" not in exc.context

    def test_delegation_chain_copy(self):
        """Test delegation chain is properly isolated."""
        original_chain = ["Agent1", "Agent2"]
        exc = AgentDelegationException(
            target_agent="Agent3",
            message="Test",
            delegation_chain=original_chain
        )

        # Modify original chain
        original_chain.append("Agent4")

        # Exception delegation chain should be unchanged
        assert exc.delegation_chain == ["Agent1", "Agent2"]
        assert "Agent4" not in exc.delegation_chain

    def test_minimal_initialization(self):
        """Test exception with minimal required parameters."""
        exc = AgentDelegationException(
            target_agent="MinimalAgent",
            message="Minimal message"
        )

        assert exc.target_agent == "MinimalAgent"
        assert exc.message == "Minimal message"
        assert isinstance(exc.context, dict)
        assert isinstance(exc.delegation_chain, list)
        assert isinstance(exc.transfer_state, bool)
        assert isinstance(exc.transfer_messages, bool)