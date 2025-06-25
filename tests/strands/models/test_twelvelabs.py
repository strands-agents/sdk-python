import unittest.mock
from unittest.mock import Mock

import pytest

from strands.models.twelvelabs import TwelveLabsModel
from strands.types.exceptions import ModelThrottledException


@pytest.fixture
def mock_twelvelabs_client():
    """Mock TwelveLabs client for testing."""
    mock_client = Mock()
    mock_context_manager = Mock()
    mock_context_manager.__enter__ = Mock(return_value=mock_client)
    mock_context_manager.__exit__ = Mock(return_value=None)

    with unittest.mock.patch("strands.models.twelvelabs.TwelveLabs", return_value=mock_context_manager):
        yield mock_client


@pytest.fixture
def model_config():
    """Base model configuration for testing."""
    return {
        "model_id": "Marengo-retrieval-2.7",
        "index_id": "test-index-123",
        "api_key": "test-api-key",
    }


@pytest.fixture
def model(model_config):
    """TwelveLabs model instance for testing."""
    return TwelveLabsModel(**model_config)


@pytest.fixture
def messages():
    """Sample messages for testing."""
    return [{"role": "user", "content": [{"text": "find videos with rabbits"}]}]


@pytest.fixture
def search_result():
    """Mock search result from TwelveLabs API."""
    mock_result = Mock()
    mock_result.pool = Mock()
    mock_result.pool.total_count = 5

    # Mock search result items
    mock_item1 = Mock()
    mock_item1.score = 0.95
    mock_item1.start = 10.5
    mock_item1.end = 15.2
    mock_item1.confidence = "high"
    mock_item1.video_id = "video_123"

    mock_item2 = Mock()
    mock_item2.score = 0.87
    mock_item2.start = 25.1
    mock_item2.end = 30.8
    mock_item2.confidence = "medium"
    mock_item2.video_id = "video_456"

    mock_result.data = [mock_item1, mock_item2]
    return mock_result


class TestTwelveLabsModel:
    """Test suite for TwelveLabsModel."""

    def test_init_with_valid_config(self, model_config):
        """Test model initialization with valid configuration."""
        model = TwelveLabsModel(**model_config)

        assert model.config["model_id"] == "Marengo-retrieval-2.7"
        assert model.config["index_id"] == "test-index-123"
        assert model._api_key == "test-api-key"

    def test_init_missing_api_key(self):
        """Test model initialization fails without API key."""
        with unittest.mock.patch("strands.models.twelvelabs.os.getenv", return_value=None):
            with pytest.raises(ValueError, match="TwelveLabs API key required"):
                TwelveLabsModel(model_id="test", index_id="test")

    def test_update_config(self, model):
        """Test updating model configuration."""
        model.update_config(page_limit=20, threshold="high")

        assert model.config["page_limit"] == 20
        assert model.config["threshold"] == "high"

    def test_get_config(self, model):
        """Test getting model configuration."""
        config = model.get_config()

        assert config["model_id"] == "Marengo-retrieval-2.7"
        assert config["index_id"] == "test-index-123"

    def test_format_request_with_text_message(self, model, messages):
        """Test formatting request with text message."""
        request = model.format_request(messages)

        expected_request = {
            "query_text": "find videos with rabbits",
            "index_id": "test-index-123",
            "search_options": ["visual"],
            "group_by": "clip",
            "threshold": None,
            "page_limit": 10,
        }

        assert request == expected_request

    def test_format_request_with_string_content(self, model):
        """Test formatting request with string content."""
        messages = [{"role": "user", "content": "search for cats"}]
        request = model.format_request(messages)

        assert request["query_text"] == "search for cats"

    def test_format_request_missing_index_id(self, model_config):
        """Test formatting request fails without index_id."""
        config_without_index = {**model_config}
        del config_without_index["index_id"]

        model = TwelveLabsModel(**config_without_index)
        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(ValueError, match="index_id must be configured"):
            model.format_request(messages)

    def test_format_chunk(self, model):
        """Test formatting chunk events."""
        event = {"type": "test_event", "data": "test_data"}
        chunk = model.format_chunk(event)

        assert chunk == event

    def test_stream_successful_search(self, mock_twelvelabs_client, model, search_result):
        """Test successful video search streaming."""
        mock_twelvelabs_client.search.query.return_value = search_result

        request = {
            "query_text": "find rabbits",
            "index_id": "test-index-123",
            "search_options": ["visual"],
            "group_by": "clip",
            "threshold": None,
            "page_limit": 10,
        }

        events = list(model.stream(request))

        # Verify stream events structure
        assert len(events) >= 4  # messageStart, contentBlockStart, contentBlockDelta, contentBlockStop, messageStop
        assert events[0]["messageStart"]["role"] == "assistant"
        assert "contentBlockStart" in events[1]
        assert "contentBlockDelta" in events[2]
        assert "Video Search Results" in events[2]["contentBlockDelta"]["delta"]["text"]

        # Verify API call
        mock_twelvelabs_client.search.query.assert_called_once_with(
            index_id="test-index-123",
            options=["visual"],
            query_text="find rabbits",
            group_by="clip",
            threshold=None,
            page_limit=10,
        )

    def test_stream_no_query_text(self, model):
        """Test streaming with missing query text."""
        request = {"query_text": "", "index_id": "test-index-123"}
        events = list(model.stream(request))

        # Should return error response
        assert any("No search query provided" in str(event) for event in events)

    def test_stream_no_index_id(self, model):
        """Test streaming with missing index ID."""
        request = {"query_text": "test", "index_id": None}
        events = list(model.stream(request))

        # Should return error response
        assert any("No index ID specified" in str(event) for event in events)

    def test_stream_throttling_exception(self, mock_twelvelabs_client, model):
        """Test handling of throttling exceptions."""
        mock_twelvelabs_client.search.query.side_effect = Exception("usage_limit_exceeded")

        request = {"query_text": "test", "index_id": "test-index-123"}

        with pytest.raises(ModelThrottledException, match="TwelveLabs API throttling"):
            list(model.stream(request))

    def test_stream_authentication_error(self, mock_twelvelabs_client, model):
        """Test handling of authentication errors."""
        mock_twelvelabs_client.search.query.side_effect = Exception("api_key_invalid")

        request = {"query_text": "test", "index_id": "test-index-123"}
        events = list(model.stream(request))

        # Should return error response, not raise exception
        assert any("Authentication failed" in str(event) for event in events)

    def test_stream_search_error(self, mock_twelvelabs_client, model):
        """Test handling of search-specific errors."""
        mock_twelvelabs_client.search.query.side_effect = Exception("search_option_not_supported")

        request = {"query_text": "test", "index_id": "test-index-123"}
        events = list(model.stream(request))

        # Should return error response, not raise exception
        assert any("Search error" in str(event) for event in events)

    def test_stream_generic_error(self, mock_twelvelabs_client, model):
        """Test handling of generic errors."""
        mock_twelvelabs_client.search.query.side_effect = Exception("unknown error")

        request = {"query_text": "test", "index_id": "test-index-123"}
        events = list(model.stream(request))

        # Should return error response
        assert any("Search failed" in str(event) for event in events)

    def test_search_videos_successful(self, mock_twelvelabs_client, model, search_result):
        """Test successful direct video search."""
        mock_twelvelabs_client.search.query.return_value = search_result

        result = model.search_videos("find cats", "test-index-123")

        assert result["total_results"] == 5
        assert len(result["results"]) == 2
        assert result["query"] == "find cats"
        assert result["index_id"] == "test-index-123"
        assert result["model_id"] == "Marengo-retrieval-2.7"

        # Verify result structure
        first_result = result["results"][0]
        assert first_result["score"] == 0.95
        assert first_result["video_id"] == "video_123"

    def test_search_videos_with_default_index(self, mock_twelvelabs_client, model, search_result):
        """Test direct search using default index ID."""
        mock_twelvelabs_client.search.query.return_value = search_result

        result = model.search_videos("find dogs")

        assert result["index_id"] == "test-index-123"  # Uses default from config

    def test_search_videos_missing_index(self, model_config):
        """Test direct search fails without index ID."""
        config_without_index = {**model_config}
        del config_without_index["index_id"]

        model = TwelveLabsModel(**config_without_index)

        with pytest.raises(ValueError, match="index_id required"):
            model.search_videos("test")

    def test_search_videos_throttling_exception(self, mock_twelvelabs_client, model):
        """Test direct search throttling exception handling."""
        mock_twelvelabs_client.search.query.side_effect = Exception("usage_limit_exceeded")

        with pytest.raises(ModelThrottledException, match="TwelveLabs API throttling"):
            model.search_videos("test")

    def test_search_videos_authentication_error(self, mock_twelvelabs_client, model):
        """Test direct search authentication error handling."""
        mock_twelvelabs_client.search.query.side_effect = Exception("api_key_invalid")

        with pytest.raises(ValueError, match="TwelveLabs authentication error"):
            model.search_videos("test")

    def test_search_videos_search_error(self, mock_twelvelabs_client, model):
        """Test direct search error handling."""
        mock_twelvelabs_client.search.query.side_effect = Exception("search_option_not_supported")

        with pytest.raises(ValueError, match="TwelveLabs search error"):
            model.search_videos("test")

    def test_error_response_format(self, model):
        """Test error response formatting."""
        events = list(model._error_response("Test error message"))

        assert len(events) == 5
        assert events[0]["messageStart"]["role"] == "assistant"
        assert events[1]["contentBlockStart"]["start"] == {}
        assert "Error: Test error message" in events[2]["contentBlockDelta"]["delta"]["text"]
        assert "contentBlockStop" in events[3]
        assert events[4]["messageStop"]["stopReason"] == "end_turn"

    def test_video_grouped_results(self, mock_twelvelabs_client, model):
        """Test handling of video-grouped search results."""
        # Mock video-grouped result
        mock_result = Mock()
        mock_result.pool = Mock()
        mock_result.pool.total_count = 1

        mock_video = Mock()
        mock_video.id = "video_789"
        mock_video.clips = [
            Mock(score=0.9, start=5.0, end=10.0, confidence="high", video_id="video_789"),
            Mock(score=0.8, start=15.0, end=20.0, confidence="medium", video_id="video_789"),
        ]

        mock_result.data = [mock_video]
        mock_twelvelabs_client.search.query.return_value = mock_result

        result = model.search_videos("test", group_by="video")

        assert len(result["results"]) == 1
        video_result = result["results"][0]
        assert video_result["video_id"] == "video_789"
        assert len(video_result["clips"]) == 2
        assert video_result["clips"][0]["score"] == 0.9

    @pytest.mark.parametrize(
        "error_message,expected_exception",
        [
            ("usage_limit_exceeded", ModelThrottledException),
            ("api_key_invalid", ValueError),
            ("search_option_not_supported", ValueError),
        ],
    )
    def test_error_classification(self, mock_twelvelabs_client, model, error_message, expected_exception):
        """Test proper classification of different error types."""
        mock_twelvelabs_client.search.query.side_effect = Exception(error_message)

        with pytest.raises(expected_exception):
            model.search_videos("test")
