"""Unit tests for ledger context providers."""

from unittest.mock import Mock, patch

from strands.experimental.steering.context_providers.ledger_provider import (
    LedgerAfterToolCall,
    LedgerBeforeToolCall,
    LedgerProvider,
)
from strands.experimental.steering.core.context import SteeringContext
from strands.hooks.events import AfterToolCallEvent, BeforeToolCallEvent


def test_context_providers_method():
    """Test context_providers method returns correct callbacks."""
    provider = LedgerProvider()

    callbacks = provider.context_providers()

    assert len(callbacks) == 2
    assert isinstance(callbacks[0], LedgerBeforeToolCall)
    assert isinstance(callbacks[1], LedgerAfterToolCall)


@patch("strands.experimental.steering.context_providers.ledger_provider.datetime")
def test_ledger_before_tool_call_new_ledger(mock_datetime):
    """Test LedgerBeforeToolCall with new ledger."""
    mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"

    callback = LedgerBeforeToolCall()
    steering_context = SteeringContext()

    tool_use = {"name": "test_tool", "arguments": {"param": "value"}}
    event = Mock(spec=BeforeToolCallEvent)
    event.tool_use = tool_use

    callback(event, steering_context)

    ledger = steering_context.data.get("ledger")
    assert ledger is not None
    assert "session_start" in ledger
    assert "tool_calls" in ledger
    assert len(ledger["tool_calls"]) == 1

    tool_call = ledger["tool_calls"][0]
    assert tool_call["tool_name"] == "test_tool"
    assert tool_call["tool_args"] == {"param": "value"}
    assert tool_call["status"] == "pending"


@patch("strands.experimental.steering.context_providers.ledger_provider.datetime")
def test_ledger_before_tool_call_existing_ledger(mock_datetime):
    """Test LedgerBeforeToolCall with existing ledger."""
    mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"

    callback = LedgerBeforeToolCall()
    steering_context = SteeringContext()

    # Set up existing ledger
    existing_ledger = {
        "session_start": "2024-01-01T10:00:00",
        "tool_calls": [{"name": "previous_tool"}],
        "conversation_history": [],
        "session_metadata": {},
    }
    steering_context.data.set("ledger", existing_ledger)

    tool_use = {"name": "new_tool", "arguments": {"param": "value"}}
    event = Mock(spec=BeforeToolCallEvent)
    event.tool_use = tool_use

    callback(event, steering_context)

    ledger = steering_context.data.get("ledger")
    assert len(ledger["tool_calls"]) == 2
    assert ledger["tool_calls"][0]["name"] == "previous_tool"
    assert ledger["tool_calls"][1]["tool_name"] == "new_tool"


@patch("strands.experimental.steering.context_providers.ledger_provider.datetime")
def test_ledger_after_tool_call_success(mock_datetime):
    """Test LedgerAfterToolCall with successful completion."""
    mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:05:00"

    callback = LedgerAfterToolCall()
    steering_context = SteeringContext()

    # Set up existing ledger with pending call
    existing_ledger = {
        "tool_calls": [{"tool_name": "test_tool", "status": "pending", "timestamp": "2024-01-01T12:00:00"}]
    }
    steering_context.data.set("ledger", existing_ledger)

    event = Mock(spec=AfterToolCallEvent)
    event.result = {"status": "success", "content": ["success_result"]}
    event.exception = None

    callback(event, steering_context)

    ledger = steering_context.data.get("ledger")
    tool_call = ledger["tool_calls"][0]
    assert tool_call["status"] == "success"
    assert tool_call["result"] == ["success_result"]
    assert tool_call["error"] is None
    assert tool_call["completion_timestamp"] == "2024-01-01T12:05:00"


def test_ledger_after_tool_call_no_calls():
    """Test LedgerAfterToolCall when no tool calls exist."""
    callback = LedgerAfterToolCall()
    steering_context = SteeringContext()

    # Set up ledger with no tool calls
    existing_ledger = {"tool_calls": []}
    steering_context.data.set("ledger", existing_ledger)

    event = Mock(spec=AfterToolCallEvent)
    event.result = {"status": "success", "content": ["test"]}
    event.exception = None

    # Should not crash when no tool calls exist
    callback(event, steering_context)

    ledger = steering_context.data.get("ledger")
    assert ledger["tool_calls"] == []


def test_session_start_persistence():
    """Test that session_start is set during initialization and persists."""
    with patch("strands.experimental.steering.context_providers.ledger_provider.datetime") as mock_datetime:
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T10:00:00"

        callback = LedgerBeforeToolCall()

        assert callback.session_start == "2024-01-01T10:00:00"
