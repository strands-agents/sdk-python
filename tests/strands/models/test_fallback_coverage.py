"""Additional tests to improve FallbackModel coverage."""

from unittest.mock import AsyncMock

import pytest

from strands.models import Model
from strands.models.fallback import FallbackModel
from strands.types.exceptions import ModelThrottledException


@pytest.fixture
def mock_primary():
    """Create a mock primary model."""
    model = AsyncMock(spec=Model)
    model.get_config.return_value = {"model": "primary"}
    return model


@pytest.fixture
def mock_fallback():
    """Create a mock fallback model."""
    model = AsyncMock(spec=Model)
    model.get_config.return_value = {"model": "fallback"}
    return model


async def async_generator(items):
    """Helper function to create async generators for test data."""
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_track_stats_disabled_primary_failure(mock_primary, mock_fallback, alist):
    """Test that primary failures are not tracked when track_stats=False."""
    # Create FallbackModel with track_stats=False
    fallback_model = FallbackModel(primary=mock_primary, fallback=mock_fallback, track_stats=False)

    # Mock primary.stream to raise ModelThrottledException
    mock_primary.stream.side_effect = ModelThrottledException("Primary throttled")

    # Mock fallback.stream to return test events
    fallback_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Fallback response"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    mock_fallback.stream.return_value = async_generator(fallback_events)

    # Call stream
    messages = [{"role": "user", "content": [{"text": "Hi"}]}]
    result = await alist(fallback_model.stream(messages))

    # Assert fallback events received
    assert result == fallback_events

    # This should cover the track_stats=False branch in _handle_primary_failure
    # The primary_failures counter should not be incremented


def test_get_model_name_with_get_config_exception(mock_primary, mock_fallback):
    """Test _get_model_name when get_config() raises an exception."""
    # Configure mock to raise exception when get_config is called
    mock_primary.get_config.side_effect = Exception("Config error")

    fallback_model = FallbackModel(primary=mock_primary, fallback=mock_fallback)

    # This should cover the exception handling branch in _get_model_name
    model_name = fallback_model._get_model_name(mock_primary)

    # Should fall back to class name
    assert model_name == "Model"


def test_get_model_name_with_empty_config(mock_primary, mock_fallback):
    """Test _get_model_name when config has no model identifiers."""
    # Configure mock to return empty config
    mock_primary.get_config.return_value = {}

    fallback_model = FallbackModel(primary=mock_primary, fallback=mock_fallback)

    # This should cover the fallback to class name branch
    model_name = fallback_model._get_model_name(mock_primary)

    # Should fall back to class name
    assert model_name == "Model"


def test_get_model_name_with_none_values(mock_primary, mock_fallback):
    """Test _get_model_name when config has None values for model identifiers."""
    # Configure mock to return config with None values
    mock_primary.get_config.return_value = {"model_id": None, "model": None, "name": None, "other": "value"}

    fallback_model = FallbackModel(primary=mock_primary, fallback=mock_fallback)

    # This should cover the branch where identifiers exist but are None/empty
    model_name = fallback_model._get_model_name(mock_primary)

    # Should fall back to class name
    assert model_name == "Model"


def test_get_model_name_with_empty_string_values(mock_primary, mock_fallback):
    """Test _get_model_name when config has empty string values."""
    # Configure mock to return config with empty string values
    mock_primary.get_config.return_value = {"model_id": "", "model": "", "name": "", "other": "value"}

    fallback_model = FallbackModel(primary=mock_primary, fallback=mock_fallback)

    # This should cover the branch where identifiers exist but are empty strings
    model_name = fallback_model._get_model_name(mock_primary)

    # Should fall back to class name
    assert model_name == "Model"


def test_should_fallback_with_non_dict_config(mock_primary, mock_fallback):
    """Test _should_fallback with various error types."""
    fallback_model = FallbackModel(primary=mock_primary, fallback=mock_fallback)

    # Test with generic Exception containing connection keywords
    error = Exception("Connection refused by server")
    assert fallback_model._should_fallback(error) is True

    # Test with generic Exception containing network keywords
    error = Exception("Network timeout occurred")
    assert fallback_model._should_fallback(error) is True

    # Test with generic Exception containing timeout keywords
    error = Exception("Request timeout")
    assert fallback_model._should_fallback(error) is True

    # Test with generic Exception containing unavailable keywords
    error = Exception("Service unavailable")
    assert fallback_model._should_fallback(error) is True

    # Test with generic Exception containing closed keywords
    error = Exception("Connection closed")
    assert fallback_model._should_fallback(error) is True

    # Test with generic Exception containing aborted keywords
    error = Exception("Connection aborted")
    assert fallback_model._should_fallback(error) is True

    # Test with generic Exception containing reset keywords
    error = Exception("Connection reset")
    assert fallback_model._should_fallback(error) is True

    # Test with generic Exception that doesn't match any keywords
    error = Exception("Some other error")
    assert fallback_model._should_fallback(error) is False


def test_update_config_all_parameters(mock_primary, mock_fallback):
    """Test update_config with all possible parameters to cover all branches."""
    fallback_model = FallbackModel(primary=mock_primary, fallback=mock_fallback)

    # Custom function for testing
    def custom_should_fallback(error):
        return True

    # Update all possible config parameters
    fallback_model.update_config(
        circuit_failure_threshold=10,
        circuit_time_window=300.0,
        circuit_cooldown_seconds=120,
        should_fallback=custom_should_fallback,
        track_stats=False,
    )

    # Verify all parameters were updated
    assert fallback_model.circuit_failure_threshold == 10
    assert fallback_model.circuit_time_window == 300.0
    assert fallback_model.circuit_cooldown_seconds == 120
    assert fallback_model.should_fallback == custom_should_fallback
    assert fallback_model.track_stats is False


def test_update_config_partial_parameters(mock_primary, mock_fallback):
    """Test update_config with only some parameters to ensure others remain unchanged."""
    fallback_model = FallbackModel(primary=mock_primary, fallback=mock_fallback)

    # Store original values
    original_time_window = fallback_model.circuit_time_window
    original_cooldown = fallback_model.circuit_cooldown_seconds
    original_should_fallback = fallback_model.should_fallback
    original_track_stats = fallback_model.track_stats

    # Update only circuit_failure_threshold
    fallback_model.update_config(circuit_failure_threshold=15)

    # Verify only the specified parameter changed
    assert fallback_model.circuit_failure_threshold == 15
    assert fallback_model.circuit_time_window == original_time_window
    assert fallback_model.circuit_cooldown_seconds == original_cooldown
    assert fallback_model.should_fallback == original_should_fallback
    assert fallback_model.track_stats == original_track_stats

    # Update only circuit_time_window
    fallback_model.update_config(circuit_time_window=180.0)

    # Verify only the specified parameter changed
    assert fallback_model.circuit_failure_threshold == 15  # Previous change
    assert fallback_model.circuit_time_window == 180.0
    assert fallback_model.circuit_cooldown_seconds == original_cooldown
    assert fallback_model.should_fallback == original_should_fallback
    assert fallback_model.track_stats == original_track_stats

    # Update only circuit_cooldown_seconds
    fallback_model.update_config(circuit_cooldown_seconds=90)

    # Verify only the specified parameter changed
    assert fallback_model.circuit_failure_threshold == 15  # Previous change
    assert fallback_model.circuit_time_window == 180.0  # Previous change
    assert fallback_model.circuit_cooldown_seconds == 90
    assert fallback_model.should_fallback == original_should_fallback
    assert fallback_model.track_stats == original_track_stats

    # Update only should_fallback
    def new_should_fallback(error):
        return False

    fallback_model.update_config(should_fallback=new_should_fallback)

    # Verify only the specified parameter changed
    assert fallback_model.circuit_failure_threshold == 15  # Previous change
    assert fallback_model.circuit_time_window == 180.0  # Previous change
    assert fallback_model.circuit_cooldown_seconds == 90  # Previous change
    assert fallback_model.should_fallback == new_should_fallback
    assert fallback_model.track_stats == original_track_stats

    # Update only track_stats
    fallback_model.update_config(track_stats=False)

    # Verify only the specified parameter changed
    assert fallback_model.circuit_failure_threshold == 15  # Previous change
    assert fallback_model.circuit_time_window == 180.0  # Previous change
    assert fallback_model.circuit_cooldown_seconds == 90  # Previous change
    assert fallback_model.should_fallback == new_should_fallback  # Previous change
    assert fallback_model.track_stats is False


@pytest.mark.asyncio
async def test_circuit_skip_with_track_stats_disabled(mock_primary, mock_fallback, alist):
    """Test circuit skip behavior when track_stats is disabled."""
    # Create FallbackModel with track_stats=False and low threshold
    fallback_model = FallbackModel(
        primary=mock_primary, fallback=mock_fallback, circuit_failure_threshold=1, track_stats=False
    )

    # Mock primary.stream to raise ModelThrottledException to open circuit
    mock_primary.stream.side_effect = ModelThrottledException("Primary throttled")

    # Mock fallback.stream to return test events
    fallback_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Fallback response"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    mock_fallback.stream.return_value = async_generator(fallback_events)

    messages = [{"role": "user", "content": [{"text": "Hi"}]}]

    # First request to open circuit
    await alist(fallback_model.stream(messages))

    # Reset mocks to track next call
    mock_primary.stream.reset_mock()
    mock_fallback.stream.reset_mock()
    mock_fallback.stream.return_value = async_generator(fallback_events)

    # Second request should skip primary (circuit is open) with track_stats=False
    result = await alist(fallback_model.stream(messages))

    # Assert primary was not called (circuit skip)
    mock_primary.stream.assert_not_called()
    mock_fallback.stream.assert_called_once()
    assert result == fallback_events

    # This should cover the track_stats=False branch in circuit skip logic


@pytest.mark.asyncio
async def test_structured_output_circuit_skip_with_track_stats_disabled(mock_primary, mock_fallback, alist):
    """Test structured_output circuit skip behavior when track_stats is disabled."""
    from pydantic import BaseModel

    class TestResponse(BaseModel):
        answer: str
        confidence: float

    # Create FallbackModel with track_stats=False and low threshold
    fallback_model = FallbackModel(
        primary=mock_primary, fallback=mock_fallback, circuit_failure_threshold=1, track_stats=False
    )

    # Mock primary.structured_output to raise ModelThrottledException to open circuit
    mock_primary.structured_output.side_effect = ModelThrottledException("Primary throttled")

    # Mock fallback.structured_output to return test events
    fallback_events = [
        {"chunk_type": "structured_output_complete", "data": TestResponse(answer="temp", confidence=0.5)}
    ]
    mock_fallback.structured_output.return_value = async_generator(fallback_events)

    prompt = [{"role": "user", "content": [{"text": "Hi"}]}]

    # First request to open circuit
    await alist(fallback_model.structured_output(TestResponse, prompt))

    # Reset mocks to track next call
    mock_primary.structured_output.reset_mock()
    mock_fallback.structured_output.reset_mock()
    mock_fallback.structured_output.return_value = async_generator(fallback_events)

    # Second request should skip primary (circuit is open) with track_stats=False
    result = await alist(fallback_model.structured_output(TestResponse, prompt))

    # Assert primary was not called (circuit skip)
    mock_primary.structured_output.assert_not_called()
    mock_fallback.structured_output.assert_called_once()
    assert result == fallback_events

    # This should cover the track_stats=False branch in structured_output circuit skip logic


@pytest.mark.asyncio
async def test_both_models_fail_structured_output(mock_primary, mock_fallback):
    """Test that fallback exception is raised when both models fail in structured_output."""
    from pydantic import BaseModel

    class TestResponse(BaseModel):
        answer: str
        confidence: float

    fallback_model = FallbackModel(primary=mock_primary, fallback=mock_fallback)

    # Mock primary.structured_output to raise ModelThrottledException
    mock_primary.structured_output.side_effect = ModelThrottledException("Primary throttled")

    # Mock fallback.structured_output to raise RuntimeError
    mock_fallback.structured_output.side_effect = RuntimeError("Fallback failed")

    # Call structured_output and expect RuntimeError (not ModelThrottledException)
    prompt = [{"role": "user", "content": [{"text": "Hi"}]}]

    with pytest.raises(RuntimeError, match="Fallback failed"):
        async for _ in fallback_model.structured_output(TestResponse, prompt):
            pass

    # This should cover the "both models failed during structured output" error path (lines 666-677)


@pytest.mark.asyncio
async def test_should_fallback_returns_false_for_custom_error(mock_primary, mock_fallback):
    """Test that non-fallback-eligible errors are re-raised without fallback."""
    fallback_model = FallbackModel(primary=mock_primary, fallback=mock_fallback)

    # Mock primary.stream to raise a custom error that should not trigger fallback
    custom_error = ValueError("This is a custom error that should not trigger fallback")
    mock_primary.stream.side_effect = custom_error

    # Call stream and expect the custom error to be re-raised
    messages = [{"role": "user", "content": [{"text": "Hi"}]}]

    with pytest.raises(ValueError, match="This is a custom error that should not trigger fallback"):
        async for _ in fallback_model.stream(messages):
            pass

    # Assert fallback was not called
    mock_fallback.stream.assert_not_called()

    # Assert fallback_count is 0
    stats = fallback_model.get_stats()
    assert stats["fallback_count"] == 0

    # This should cover the "error is not fallback-eligible, re-raising" path


@pytest.mark.asyncio
async def test_should_fallback_returns_false_for_custom_error_structured_output(mock_primary, mock_fallback):
    """Test that non-fallback-eligible errors are re-raised without fallback in structured_output."""
    from pydantic import BaseModel

    class TestResponse(BaseModel):
        answer: str
        confidence: float

    fallback_model = FallbackModel(primary=mock_primary, fallback=mock_fallback)

    # Mock primary.structured_output to raise a custom error that should not trigger fallback
    custom_error = ValueError("This is a custom error that should not trigger fallback")
    mock_primary.structured_output.side_effect = custom_error

    # Call structured_output and expect the custom error to be re-raised
    prompt = [{"role": "user", "content": [{"text": "Hi"}]}]

    with pytest.raises(ValueError, match="This is a custom error that should not trigger fallback"):
        async for _ in fallback_model.structured_output(TestResponse, prompt):
            pass

    # Assert fallback was not called
    mock_fallback.structured_output.assert_not_called()

    # Assert fallback_count is 0
    stats = fallback_model.get_stats()
    assert stats["fallback_count"] == 0

    # This should cover the "error is not fallback-eligible, re-raising" path in structured_output


def test_get_model_name_with_non_dict_config(mock_primary, mock_fallback):
    """Test _get_model_name when get_config() returns non-dict."""
    # Configure mock to return non-dict config
    mock_primary.get_config.return_value = "not a dict"

    fallback_model = FallbackModel(primary=mock_primary, fallback=mock_fallback)

    # This should cover the isinstance(config, dict) check returning False
    model_name = fallback_model._get_model_name(mock_primary)

    # Should fall back to class name
    assert model_name == "Model"


@pytest.mark.asyncio
async def test_track_stats_false_in_fallback_scenarios(mock_primary, mock_fallback, alist):
    """Test track_stats=False in various fallback scenarios to cover missing branches."""
    # Create FallbackModel with track_stats=False
    fallback_model = FallbackModel(primary=mock_primary, fallback=mock_fallback, track_stats=False)

    # Mock primary.stream to raise ModelThrottledException
    mock_primary.stream.side_effect = ModelThrottledException("Primary throttled")

    # Mock fallback.stream to return test events
    fallback_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Fallback response"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    mock_fallback.stream.return_value = async_generator(fallback_events)

    # Call stream
    messages = [{"role": "user", "content": [{"text": "Hi"}]}]
    result = await alist(fallback_model.stream(messages))

    # Assert fallback events received
    assert result == fallback_events

    # This should cover the track_stats=False branches in the fallback logic (lines 635-636)


@pytest.mark.asyncio
async def test_track_stats_false_in_structured_output_fallback(mock_primary, mock_fallback, alist):
    """Test track_stats=False in structured_output fallback scenarios."""
    from pydantic import BaseModel

    class TestResponse(BaseModel):
        answer: str
        confidence: float

    # Create FallbackModel with track_stats=False
    fallback_model = FallbackModel(primary=mock_primary, fallback=mock_fallback, track_stats=False)

    # Mock primary.structured_output to raise ModelThrottledException
    mock_primary.structured_output.side_effect = ModelThrottledException("Primary throttled")

    # Mock fallback.structured_output to return test events
    fallback_events = [
        {"chunk_type": "structured_output_complete", "data": TestResponse(answer="Fallback", confidence=0.85)}
    ]
    mock_fallback.structured_output.return_value = async_generator(fallback_events)

    # Call structured_output
    prompt = [{"role": "user", "content": [{"text": "Hi"}]}]
    result = await alist(fallback_model.structured_output(TestResponse, prompt))

    # Assert fallback events received
    assert result == fallback_events

    # This should cover the track_stats=False branches in structured_output fallback logic
