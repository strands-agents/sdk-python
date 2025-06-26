"""Tests for SessionManager interface."""

from abc import ABC

import pytest

from strands.session.session_manager import SessionManager


def test_is_abstract_base_class():
    """Test that SessionManager is an abstract base class."""
    assert issubclass(SessionManager, ABC)

    # Should not be able to instantiate directly
    with pytest.raises(TypeError):
        SessionManager()


def test_abstract_methods_defined():
    """Test that all required abstract methods are defined."""
    expected_methods = ["append_message_to_agent_session", "initialize_agent"]

    for method_name in expected_methods:
        assert hasattr(SessionManager, method_name)
        method = getattr(SessionManager, method_name)
        assert callable(method)


def test_concrete_implementation_required():
    """Test that concrete implementations must implement all abstract methods."""

    class IncompleteManager(SessionManager):
        """Incomplete manager implementation for testing."""

        pass

    # Should not be able to instantiate incomplete implementation
    with pytest.raises(TypeError):
        IncompleteManager()


def test_method_signatures():
    """Test that abstract methods have correct signatures."""
    assert hasattr(SessionManager.append_message_to_agent_session, "__annotations__")
    assert hasattr(SessionManager.initialize_agent, "__annotations__")


class MockSessionManager(SessionManager):
    """Mock implementation for testing abstract method behavior."""

    def append_message_to_agent_session(self, agent, message):
        raise NotImplementedError("Test implementation")

    def initialize_agent(self, agent):
        raise NotImplementedError("Test implementation")


def test_mock_manager_instantiation():
    """Test that mock manager can be instantiated."""
    manager = MockSessionManager()
    assert isinstance(manager, SessionManager)


def test_abstract_methods_raise_not_implemented():
    """Test that abstract methods raise NotImplementedError."""
    manager = MockSessionManager()

    with pytest.raises(NotImplementedError):
        manager.append_message_to_agent_session(None, None)

    with pytest.raises(NotImplementedError):
        manager.initialize_agent(None)
