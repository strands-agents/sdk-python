"""Tests for delegation tracing functionality.

This module tests OpenTelemetry tracing integration for delegation operations.

Tests actual start_delegation_span() method implementation.
"""

from unittest.mock import Mock, patch

import pytest

from strands.telemetry.tracer import Tracer


@pytest.mark.delegation
class TestDelegationTracing:
    """Test delegation tracing and OpenTelemetry integration."""

    def test_delegation_span_attributes_complete(self):
        """Test delegation span created with all required attributes."""
        tracer = Tracer()

        with patch.object(tracer, "_start_span") as mock_start:
            mock_span = Mock()
            mock_start.return_value = mock_span

            # Use ACTUAL start_delegation_span() method that exists
            tracer.start_delegation_span(
                from_agent="Orchestrator",
                to_agent="SubAgent",
                message="Test delegation",
                delegation_depth=2,
                transfer_state=True,
                transfer_messages=False,
            )

            # Verify span was created with correct name
            mock_start.assert_called_once_with("delegation.Orchestrator.SubAgent", parent_span=None)

            # Verify all 8 attributes were set via set_attributes
            mock_span.set_attributes.assert_called_once()
            attrs = mock_span.set_attributes.call_args[0][0]

            assert attrs["delegation.from"] == "Orchestrator"
            assert attrs["delegation.to"] == "SubAgent"
            assert attrs["delegation.message"] == "Test delegation"
            assert attrs["delegation.depth"] == 2
            assert attrs["delegation.state_transferred"] is True
            assert attrs["delegation.messages_transferred"] is False
            assert attrs["gen_ai.operation.name"] == "agent_delegation"
            assert attrs["gen_ai.system"] == "strands_agents"

    def test_delegation_span_parent_child_relationship(self):
        """Test parent-child span relationships for nested delegation."""
        tracer = Tracer()

        with patch.object(tracer, "_start_span") as mock_start:
            parent_span = Mock()
            child_span = Mock()
            mock_start.side_effect = [parent_span, child_span]

            # Create parent delegation span
            parent = tracer.start_delegation_span(
                from_agent="Root", to_agent="Level1", message="First", delegation_depth=1
            )

            # Create child delegation span with parent
            tracer.start_delegation_span(
                from_agent="Level1", to_agent="Level2", message="Nested", delegation_depth=2, parent_span=parent
            )

            # Verify both spans were created
            assert mock_start.call_count == 2

            # Verify parent span has no parent
            first_call = mock_start.call_args_list[0]
            assert first_call[0][0] == "delegation.Root.Level1"
            assert first_call[1]["parent_span"] is None

            # Verify child span has parent
            second_call = mock_start.call_args_list[1]
            assert second_call[0][0] == "delegation.Level1.Level2"
            assert second_call[1]["parent_span"] == parent

    def test_delegation_span_naming_convention(self):
        """Test span names follow delegation.{from}.{to} pattern."""
        tracer = Tracer()

        with patch.object(tracer, "_start_span") as mock_start:
            mock_span = Mock()
            mock_start.return_value = mock_span

            # Use actual start_delegation_span method
            tracer.start_delegation_span(
                from_agent="Orchestrator", to_agent="Specialist", message="Test", delegation_depth=1
            )

            # Verify span name follows convention
            span_name = mock_start.call_args[0][0]
            assert span_name == "delegation.Orchestrator.Specialist"
            assert "delegation." in span_name
            assert "Orchestrator" in span_name
            assert "Specialist" in span_name

    def test_delegation_span_with_minimal_attributes(self):
        """Test delegation span with minimal required parameters."""
        tracer = Tracer()

        with patch.object(tracer, "_start_span") as mock_start:
            mock_span = Mock()
            mock_start.return_value = mock_span

            # Create span with minimal parameters (defaults for transfer flags)
            span = tracer.start_delegation_span(from_agent="A", to_agent="B", message="test", delegation_depth=1)

            # Should succeed and use default values
            mock_start.assert_called_once()
            assert span == mock_span

            # Verify defaults were used
            attrs = mock_span.set_attributes.call_args[0][0]
            assert attrs["delegation.state_transferred"] is True  # Default
            assert attrs["delegation.messages_transferred"] is True  # Default

    def test_delegation_span_error_handling(self):
        """Test delegation span handles errors gracefully."""
        tracer = Tracer()

        with patch.object(tracer, "_start_span") as mock_start:
            # Simulate an error during span creation
            mock_start.side_effect = Exception("Span creation failed")

            # Should propagate the exception
            with pytest.raises(Exception, match="Span creation failed"):
                tracer.start_delegation_span(from_agent="A", to_agent="B", message="test", delegation_depth=1)

    def test_delegation_depth_tracking(self):
        """Test delegation depth is properly tracked in spans."""
        tracer = Tracer()

        with patch.object(tracer, "_start_span") as mock_start:
            mock_span = Mock()
            mock_start.return_value = mock_span

            # Create spans with different depths
            for depth in [1, 2, 3]:
                tracer.start_delegation_span(
                    from_agent=f"Agent{depth - 1}", to_agent=f"Agent{depth}", message="test", delegation_depth=depth
                )

            # Verify all spans were created
            assert mock_start.call_count == 3

            # Verify depth attribute for each call
            for idx, depth in enumerate([1, 2, 3]):
                attrs = mock_span.set_attributes.call_args_list[idx][0][0]
                assert attrs["delegation.depth"] == depth

    def test_delegation_state_transfer_tracking(self):
        """Test state and message transfer flags are tracked."""
        tracer = Tracer()

        with patch.object(tracer, "_start_span") as mock_start:
            mock_span = Mock()
            mock_start.return_value = mock_span

            # Test all combinations of transfer flags
            test_cases = [
                (True, True),
                (True, False),
                (False, True),
                (False, False),
            ]

            for state_transfer, message_transfer in test_cases:
                tracer.start_delegation_span(
                    from_agent="A",
                    to_agent="B",
                    message="test",
                    delegation_depth=1,
                    transfer_state=state_transfer,
                    transfer_messages=message_transfer,
                )

            # Verify all combinations were tracked
            assert mock_start.call_count == 4

            # Verify each combination was properly set
            for idx, (state_transfer, message_transfer) in enumerate(test_cases):
                attrs = mock_span.set_attributes.call_args_list[idx][0][0]
                assert attrs["delegation.state_transferred"] == state_transfer
                assert attrs["delegation.messages_transferred"] == message_transfer

    def test_delegation_span_with_gen_ai_attributes(self):
        """Test delegation spans include gen_ai standard attributes."""
        tracer = Tracer()

        with patch.object(tracer, "_start_span") as mock_start:
            mock_span = Mock()
            mock_start.return_value = mock_span

            tracer.start_delegation_span(
                from_agent="Orchestrator", to_agent="SubAgent", message="test", delegation_depth=1
            )

            # Verify gen_ai attributes were set
            attrs = mock_span.set_attributes.call_args[0][0]

            assert attrs["gen_ai.operation.name"] == "agent_delegation"
            assert attrs["gen_ai.system"] == "strands_agents"
            # Note: gen_ai.agent.name is not set by start_delegation_span
