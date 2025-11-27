"""Tests for SAP GenAI Hub model provider."""

from unittest.mock import MagicMock, patch

import pytest

from strands.models.sap_genai_hub import SAPGenAIHubModel


class TestSAPGenAIHubModel:
    """Test suite for SAP GenAI Hub model provider."""

    def test_initialization_with_default_config(self):
        """Test model initialization with default configuration."""
        with patch("strands.models.sap_genai_hub.Session") as mock_session:
            mock_client = MagicMock()
            mock_session.return_value.client.return_value = mock_client

            model = SAPGenAIHubModel()

            assert model.config["model_id"] == "amazon--nova-lite"
            mock_session.return_value.client.assert_called_once_with(
                model_name="amazon--nova-lite"
            )

    def test_initialization_with_custom_config(self):
        """Test model initialization with custom configuration."""
        with patch("strands.models.sap_genai_hub.Session") as mock_session:
            mock_client = MagicMock()
            mock_session.return_value.client.return_value = mock_client

            model = SAPGenAIHubModel(
                model_id="anthropic--claude-3-sonnet",
                temperature=0.7,
                max_tokens=1000,
            )

            assert model.config["model_id"] == "anthropic--claude-3-sonnet"
            assert model.config["temperature"] == 0.7
            assert model.config["max_tokens"] == 1000

    def test_update_config(self):
        """Test updating model configuration."""
        with patch("strands.models.sap_genai_hub.Session") as mock_session:
            mock_client = MagicMock()
            mock_session.return_value.client.return_value = mock_client

            model = SAPGenAIHubModel()
            model.update_config(temperature=0.5, max_tokens=2000)

            assert model.config["temperature"] == 0.5
            assert model.config["max_tokens"] == 2000

    def test_get_config(self):
        """Test getting model configuration."""
        with patch("strands.models.sap_genai_hub.Session") as mock_session:
            mock_client = MagicMock()
            mock_session.return_value.client.return_value = mock_client

            model = SAPGenAIHubModel(temperature=0.8)
            config = model.get_config()

            assert config["temperature"] == 0.8
            assert "model_id" in config

    def test_is_nova_model(self):
        """Test Nova model detection."""
        with patch("strands.models.sap_genai_hub.Session") as mock_session:
            mock_client = MagicMock()
            mock_session.return_value.client.return_value = mock_client

            # Test Nova models
            for model_id in [
                "amazon--nova-pro",
                "amazon--nova-micro",
                "amazon--nova-lite",
            ]:
                model = SAPGenAIHubModel(model_id=model_id)
                assert model._is_nova_model() is True
                assert model._is_claude_model() is False
                assert model._is_titan_embed_model() is False

    def test_is_claude_model(self):
        """Test Claude model detection."""
        with patch("strands.models.sap_genai_hub.Session") as mock_session:
            mock_client = MagicMock()
            mock_session.return_value.client.return_value = mock_client

            model = SAPGenAIHubModel(model_id="anthropic--claude-3-sonnet")
            assert model._is_claude_model() is True
            assert model._is_nova_model() is False
            assert model._is_titan_embed_model() is False

    def test_is_titan_embed_model(self):
        """Test Titan Embedding model detection."""
        with patch("strands.models.sap_genai_hub.Session") as mock_session:
            mock_client = MagicMock()
            mock_session.return_value.client.return_value = mock_client

            model = SAPGenAIHubModel(model_id="amazon--titan-embed-text")
            assert model._is_titan_embed_model() is True
            assert model._is_nova_model() is False
            assert model._is_claude_model() is False

    def test_format_nova_request(self):
        """Test request formatting for Nova models."""
        with patch("strands.models.sap_genai_hub.Session") as mock_session:
            mock_client = MagicMock()
            mock_session.return_value.client.return_value = mock_client

            model = SAPGenAIHubModel(
                model_id="amazon--nova-lite", temperature=0.7, max_tokens=1000
            )

            messages = [{"role": "user", "content": [{"text": "Hello"}]}]
            system_prompt_content = [{"text": "You are a helpful assistant"}]

            request = model._format_nova_request(
                messages=messages, system_prompt_content=system_prompt_content
            )

            assert request["messages"] == messages
            assert request["system"] == system_prompt_content
            assert request["inferenceConfig"]["temperature"] == 0.7
            assert request["inferenceConfig"]["maxTokens"] == 1000

    def test_format_nova_request_with_tools(self):
        """Test request formatting for Nova models with tools."""
        with patch("strands.models.sap_genai_hub.Session") as mock_session:
            mock_client = MagicMock()
            mock_session.return_value.client.return_value = mock_client

            model = SAPGenAIHubModel(model_id="amazon--nova-lite")

            messages = [{"role": "user", "content": [{"text": "Hello"}]}]
            tool_specs = [
                {
                    "name": "test_tool",
                    "description": "A test tool",
                    "inputSchema": {"type": "object", "properties": {}},
                }
            ]

            request = model._format_nova_request(
                messages=messages, tool_specs=tool_specs
            )

            assert "toolConfig" in request
            assert len(request["toolConfig"]["tools"]) == 1
            assert request["toolConfig"]["tools"][0]["toolSpec"]["name"] == "test_tool"

    def test_format_titan_embed_request(self):
        """Test request formatting for Titan Embedding models."""
        with patch("strands.models.sap_genai_hub.Session") as mock_session:
            mock_client = MagicMock()
            mock_session.return_value.client.return_value = mock_client

            model = SAPGenAIHubModel(model_id="amazon--titan-embed-text")

            messages = [
                {"role": "user", "content": [{"text": "Text to embed"}]},
            ]

            request = model._format_titan_embed_request(messages)

            assert request["inputText"] == "Text to embed"

    def test_unsupported_model(self):
        """Test that unsupported model raises ValueError."""
        with patch("strands.models.sap_genai_hub.Session") as mock_session:
            mock_client = MagicMock()
            mock_session.return_value.client.return_value = mock_client

            model = SAPGenAIHubModel(model_id="unsupported--model")

            messages = [{"role": "user", "content": [{"text": "Hello"}]}]

            with pytest.raises(ValueError, match="unsupported model"):
                model._format_request(messages)

    def test_format_chunk_with_string(self):
        """Test chunk formatting with string input."""
        with patch("strands.models.sap_genai_hub.Session") as mock_session:
            mock_client = MagicMock()
            mock_session.return_value.client.return_value = mock_client

            model = SAPGenAIHubModel()
            chunk = model._format_chunk("test text")

            assert chunk == {"contentBlockDelta": {"delta": {"text": "test text"}}}

    def test_format_chunk_with_dict(self):
        """Test chunk formatting with dict input."""
        with patch("strands.models.sap_genai_hub.Session") as mock_session:
            mock_client = MagicMock()
            mock_session.return_value.client.return_value = mock_client

            model = SAPGenAIHubModel()
            input_chunk = {"contentBlockDelta": {"delta": {"text": "test"}}}
            chunk = model._format_chunk(input_chunk)

            assert chunk == input_chunk
