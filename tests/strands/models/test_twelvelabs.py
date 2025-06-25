import unittest.mock
from unittest.mock import Mock

import pytest

from strands.models.twelvelabs import TwelveLabsModel, TwelveLabsPegasusModel, TwelveLabsSearchModel
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
def search_model_config():
    """Search model configuration for testing."""
    return {
        "model_id": "Marengo-retrieval-2.7",
        "index_id": "test-index-123",
        "api_key": "test-api-key",
        "search_options": ["visual", "audio"],
        "group_by": "clip",
        "threshold": "medium",
        "page_limit": 15,
    }


@pytest.fixture
def pegasus_model_config():
    """Pegasus model configuration for testing."""
    return {
        "model_id": "pegasus1.3",
        "index_id": "test-index-456",
        "api_key": "test-api-key",
        "default_video_id": "test-video-789",
        "temperature": 0.8,
        "engine_options": ["visual", "audio"],
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
    """Test suite for TwelveLabsModel (legacy alias for TwelveLabsSearchModel)."""

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


class TestTwelveLabsSearchModel:
    """Test suite for TwelveLabsSearchModel."""

    def test_init_with_valid_config(self, search_model_config):
        """Test search model initialization with valid configuration."""
        model = TwelveLabsSearchModel(**search_model_config)

        assert model.config["model_id"] == "Marengo-retrieval-2.7"
        assert model.config["index_id"] == "test-index-123"
        assert model.config["search_options"] == ["visual", "audio"]
        assert model.config["group_by"] == "clip"
        assert model.config["threshold"] == "medium"
        assert model.config["page_limit"] == 15
        assert model._api_key == "test-api-key"

    def test_init_with_defaults(self):
        """Test search model initialization with default values."""
        with unittest.mock.patch("strands.models.twelvelabs.os.getenv", return_value="env-api-key"):
            model = TwelveLabsSearchModel(index_id="test-index")

        assert model.config["model_id"] == "Marengo-retrieval-2.7"
        assert model.config["search_options"] == ["visual"]
        assert model.config["group_by"] == "clip"
        assert model.config["page_limit"] == 10
        assert model._api_key == "env-api-key"

    def test_init_missing_api_key(self):
        """Test search model initialization fails without API key."""
        with unittest.mock.patch("strands.models.twelvelabs.os.getenv", return_value=None):
            with pytest.raises(ValueError, match="TwelveLabs API key required"):
                TwelveLabsSearchModel(index_id="test")

    def test_update_config_filters_invalid_keys(self, search_model_config):
        """Test that update_config only accepts valid configuration keys."""
        model = TwelveLabsSearchModel(**search_model_config)

        model.update_config(page_limit=25, invalid_key="should_be_ignored", threshold="high")

        assert model.config["page_limit"] == 25
        assert model.config["threshold"] == "high"
        assert "invalid_key" not in model.config

    def test_format_request_extracts_text_from_list_content(self, search_model_config):
        """Test format_request extracts text from list-based content."""
        model = TwelveLabsSearchModel(**search_model_config)
        messages = [{"role": "user", "content": [{"text": "search for dolphins"}]}]

        request = model.format_request(messages)

        assert request["query_text"] == "search for dolphins"
        assert request["index_id"] == "test-index-123"
        assert request["search_options"] == ["visual", "audio"]
        assert request["group_by"] == "clip"
        assert request["threshold"] == "medium"
        assert request["page_limit"] == 15

    def test_format_request_handles_mixed_content_blocks(self, search_model_config):
        """Test format_request handles mixed content blocks correctly."""
        model = TwelveLabsSearchModel(**search_model_config)
        messages = [{"role": "user", "content": [{"type": "image", "data": "..."}, {"text": "find this object"}]}]

        request = model.format_request(messages)

        assert request["query_text"] == "find this object"

    def test_structured_output_not_implemented(self, search_model_config):
        """Test that structured_output raises NotImplementedError."""
        model = TwelveLabsSearchModel(**search_model_config)

        with pytest.raises(NotImplementedError, match="TwelveLabs search models do not support structured output"):
            model.structured_output(None, None)


class TestTwelveLabsPegasusModel:
    """Test suite for TwelveLabsPegasusModel."""

    def test_init_with_valid_config(self, pegasus_model_config):
        """Test Pegasus model initialization with valid configuration."""
        model = TwelveLabsPegasusModel(**pegasus_model_config)

        assert model.config["model_id"] == "pegasus1.3"
        assert model.config["index_id"] == "test-index-456"
        assert model.config["default_video_id"] == "test-video-789"
        assert model.config["temperature"] == 0.8
        assert model.config["engine_options"] == ["visual", "audio"]
        assert model._api_key == "test-api-key"
        assert hasattr(model, "video_cache")

    def test_init_with_defaults(self):
        """Test Pegasus model initialization with default values."""
        with unittest.mock.patch("strands.models.twelvelabs.os.getenv", return_value="env-api-key"):
            model = TwelveLabsPegasusModel()

        assert model.config["model_id"] == "pegasus1.3"
        assert model.config["temperature"] == 0.7
        assert model.config["engine_options"] == ["visual", "audio"]
        assert model._api_key == "env-api-key"

    def test_init_missing_api_key(self):
        """Test Pegasus model initialization fails without API key."""
        with unittest.mock.patch("strands.models.twelvelabs.os.getenv", return_value=None):
            with pytest.raises(ValueError, match="TwelveLabs API key required"):
                TwelveLabsPegasusModel()

    def test_update_config_filters_invalid_keys(self, pegasus_model_config):
        """Test that update_config only accepts valid configuration keys."""
        model = TwelveLabsPegasusModel(**pegasus_model_config)

        model.update_config(temperature=0.9, invalid_key="should_be_ignored", default_video_id="new-video")

        assert model.config["temperature"] == 0.9
        assert model.config["default_video_id"] == "new-video"
        assert "invalid_key" not in model.config

    def test_format_request_with_valid_messages(self, pegasus_model_config):
        """Test format_request with valid messages and video_id."""
        model = TwelveLabsPegasusModel(**pegasus_model_config)
        messages = [{"role": "user", "content": [{"text": "What happens in this video?"}]}]

        request = model.format_request(messages)

        assert request["prompt"] == "What happens in this video?"
        assert request["video_id"] == "test-video-789"
        assert request["temperature"] == 0.8
        assert request["engine_options"] == ["visual", "audio"]

    def test_format_request_with_string_content(self, pegasus_model_config):
        """Test format_request with string content."""
        model = TwelveLabsPegasusModel(**pegasus_model_config)
        messages = [{"role": "user", "content": "Describe the video content"}]

        request = model.format_request(messages)

        assert request["prompt"] == "Describe the video content"

    def test_format_request_no_messages(self, pegasus_model_config):
        """Test format_request fails with no messages."""
        model = TwelveLabsPegasusModel(**pegasus_model_config)

        with pytest.raises(ValueError, match="No messages provided"):
            model.format_request([])

    def test_format_request_empty_prompt(self, pegasus_model_config):
        """Test format_request fails with empty prompt."""
        model = TwelveLabsPegasusModel(**pegasus_model_config)
        messages = [{"role": "user", "content": [{"text": "   "}]}]

        with pytest.raises(ValueError, match="Pegasus requires a text prompt"):
            model.format_request(messages)

    def test_format_request_no_video_id(self, pegasus_model_config):
        """Test format_request fails without video_id."""
        config = {**pegasus_model_config}
        del config["default_video_id"]
        model = TwelveLabsPegasusModel(**config)
        messages = [{"role": "user", "content": "test prompt"}]

        with pytest.raises(ValueError, match="No video_id provided in configuration"):
            model.format_request(messages)

    @unittest.mock.patch("strands.models.twelvelabs.hashlib.sha256")
    @unittest.mock.patch("builtins.open", new_callable=unittest.mock.mock_open, read_data=b"fake_video_data")
    def test_upload_video(self, mock_open, mock_hash, pegasus_model_config):
        """Test video upload functionality."""
        mock_hash.return_value.hexdigest.return_value = "fake_hash"

        with unittest.mock.patch("strands.models.twelvelabs.TwelveLabs") as mock_twelvelabs:
            mock_client = Mock()
            mock_context = Mock()
            mock_context.__enter__ = Mock(return_value=mock_client)
            mock_context.__exit__ = Mock(return_value=None)
            mock_twelvelabs.return_value = mock_context

            mock_task = Mock()
            mock_task.id = "task_123"
            mock_task.status = "ready"
            mock_task.video_id = "uploaded_video_456"
            mock_task.wait_for_done = Mock()
            mock_client.task.create.return_value = mock_task

            model = TwelveLabsPegasusModel(**pegasus_model_config)
            video_id = model.upload_video("/path/to/video.mp4")

            assert video_id == "uploaded_video_456"
            assert model.config["default_video_id"] == "uploaded_video_456"
            assert model.video_cache["fake_hash"] == "uploaded_video_456"
            mock_client.task.create.assert_called_once_with(index_id="test-index-456", file=b"fake_video_data")

    def test_analyze_video_with_video_id(self, pegasus_model_config):
        """Test direct video analysis with specific video_id."""
        with unittest.mock.patch("strands.models.twelvelabs.TwelveLabs") as mock_twelvelabs:
            mock_client = Mock()
            mock_context = Mock()
            mock_context.__enter__ = Mock(return_value=mock_client)
            mock_context.__exit__ = Mock(return_value=None)
            mock_twelvelabs.return_value = mock_context

            mock_response = Mock()
            mock_response.data = "This video shows a cat playing."
            mock_client.generate.text.return_value = mock_response

            model = TwelveLabsPegasusModel(**pegasus_model_config)
            result = model.analyze_video("What animal is in the video?", "specific_video_123")

            assert result == "This video shows a cat playing."
            mock_client.generate.text.assert_called_once_with(
                video_id="specific_video_123", prompt="What animal is in the video?", temperature=0.8
            )

    def test_analyze_video_with_default_video_id(self, pegasus_model_config):
        """Test direct video analysis using default video_id."""
        with unittest.mock.patch("strands.models.twelvelabs.TwelveLabs") as mock_twelvelabs:
            mock_client = Mock()
            mock_context = Mock()
            mock_context.__enter__ = Mock(return_value=mock_client)
            mock_context.__exit__ = Mock(return_value=None)
            mock_twelvelabs.return_value = mock_context

            mock_response = Mock()
            mock_response.data = "Analysis result"
            mock_client.generate.text.return_value = mock_response

            model = TwelveLabsPegasusModel(**pegasus_model_config)
            result = model.analyze_video("Analyze this video")

            assert result == "Analysis result"
            mock_client.generate.text.assert_called_once_with(
                video_id="test-video-789", prompt="Analyze this video", temperature=0.8
            )

    def test_analyze_video_no_video_id(self, pegasus_model_config):
        """Test analyze_video fails without video_id."""
        config = {**pegasus_model_config}
        del config["default_video_id"]
        model = TwelveLabsPegasusModel(**config)

        with pytest.raises(ValueError, match="video_id required"):
            model.analyze_video("test prompt")

    @unittest.mock.patch("strands.models.twelvelabs.TwelveLabs")
    def test_stream_successful_generation(self, mock_twelvelabs, pegasus_model_config):
        """Test successful Pegasus generation streaming."""
        mock_client = Mock()
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_client)
        mock_context.__exit__ = Mock(return_value=None)
        mock_twelvelabs.return_value = mock_context

        mock_response = Mock()
        mock_response.data = "Generated response about the video"
        mock_client.generate.text.return_value = mock_response

        model = TwelveLabsPegasusModel(**pegasus_model_config)
        request = {
            "video_id": "test-video-789",
            "prompt": "What is happening?",
            "temperature": 0.7,
            "engine_options": ["visual"],
        }

        events = list(model.stream(request))

        assert len(events) >= 4
        assert events[0]["messageStart"]["role"] == "assistant"
        assert "contentBlockStart" in events[1]
        assert "contentBlockDelta" in events[2]
        assert "Generated response about the video" in events[2]["contentBlockDelta"]["delta"]["text"]

        mock_client.generate.text.assert_called_once_with(
            video_id="test-video-789", prompt="What is happening?", temperature=0.7
        )

    def test_stream_no_prompt(self, pegasus_model_config):
        """Test streaming with missing prompt."""
        model = TwelveLabsPegasusModel(**pegasus_model_config)
        request = {"video_id": "test-video", "prompt": ""}

        events = list(model.stream(request))

        assert any("No prompt provided" in str(event) for event in events)

    def test_stream_no_video_id(self, pegasus_model_config):
        """Test streaming with missing video_id."""
        model = TwelveLabsPegasusModel(**pegasus_model_config)
        request = {"prompt": "test prompt", "video_id": ""}

        events = list(model.stream(request))

        assert any("No video ID specified" in str(event) for event in events)

    @unittest.mock.patch("strands.models.twelvelabs.TwelveLabs")
    def test_stream_throttling_exception(self, mock_twelvelabs, pegasus_model_config):
        """Test handling of throttling exceptions in streaming."""
        mock_client = Mock()
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_client)
        mock_context.__exit__ = Mock(return_value=None)
        mock_twelvelabs.return_value = mock_context
        mock_client.generate.text.side_effect = Exception("usage_limit_exceeded")

        model = TwelveLabsPegasusModel(**pegasus_model_config)
        request = {"video_id": "test", "prompt": "test"}

        with pytest.raises(ModelThrottledException, match="TwelveLabs API throttling"):
            list(model.stream(request))

    @unittest.mock.patch("strands.models.twelvelabs.TwelveLabs")
    def test_stream_processing_error(self, mock_twelvelabs, pegasus_model_config):
        """Test handling of processing errors in streaming."""
        mock_client = Mock()
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_client)
        mock_context.__exit__ = Mock(return_value=None)
        mock_twelvelabs.return_value = mock_context
        mock_client.generate.text.side_effect = Exception("video_processing_failed")

        model = TwelveLabsPegasusModel(**pegasus_model_config)
        request = {"video_id": "test", "prompt": "test"}

        events = list(model.stream(request))

        assert any("Video processing error" in str(event) for event in events)

    def test_structured_output_not_implemented(self, pegasus_model_config):
        """Test that structured_output raises NotImplementedError."""
        model = TwelveLabsPegasusModel(**pegasus_model_config)

        with pytest.raises(NotImplementedError, match="TwelveLabs Pegasus models do not support structured output"):
            model.structured_output(None, None)

    @unittest.mock.patch("strands.models.twelvelabs.TwelveLabs")
    def test_upload_and_index_video_caching(self, mock_twelvelabs, pegasus_model_config):
        """Test video upload caching functionality."""
        mock_client = Mock()
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_client)
        mock_context.__exit__ = Mock(return_value=None)
        mock_twelvelabs.return_value = mock_context

        mock_task = Mock()
        mock_task.id = "task_123"
        mock_task.status = "ready"
        mock_task.video_id = "video_456"
        mock_task.wait_for_done = Mock()
        mock_client.task.create.return_value = mock_task

        model = TwelveLabsPegasusModel(**pegasus_model_config)

        with unittest.mock.patch("strands.models.twelvelabs.hashlib.sha256") as mock_hash:
            mock_hash.return_value.hexdigest.return_value = "test_hash"

            # First upload
            video_id1 = model._upload_and_index_video(b"test_video_data")
            assert video_id1 == "video_456"
            assert mock_client.task.create.call_count == 1

            # Second upload with same data should use cache
            video_id2 = model._upload_and_index_video(b"test_video_data")
            assert video_id2 == "video_456"
            assert mock_client.task.create.call_count == 1  # No additional call
