"""Tests for BedrockModelInvoke."""

import json
from unittest.mock import Mock, patch

import pytest

from strands.models.bedrock_invoke import BedrockModelInvoke


class TestBedrockModelInvoke:
    """Test BedrockModelInvoke functionality."""

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        with patch("boto3.Session"):
            model = BedrockModelInvoke(model_id="test-model")
            assert model.config["model_id"] == "test-model"
            assert model.config["streaming"] is True

    def test_update_config(self):
        """Test configuration updates."""
        with patch("boto3.Session"):
            model = BedrockModelInvoke(model_id="test-model")
            model.update_config(temperature=0.7, max_tokens=1000)
            assert model.config["temperature"] == 0.7
            assert model.config["max_tokens"] == 1000

    def test_get_model_family_anthropic(self):
        """Test model family detection for Anthropic models."""
        with patch("boto3.Session"):
            model = BedrockModelInvoke(model_id="anthropic.claude-3-sonnet")
            assert model._get_model_family() == "anthropic"

    def test_get_model_family_unknown_defaults_anthropic(self):
        """Test model family detection defaults to anthropic for unknown models."""
        with patch("boto3.Session"):
            model = BedrockModelInvoke(model_id="arn:aws:bedrock:us-east-1:123:imported-model/abc123")
            assert model._get_model_family() == "anthropic"

    def test_format_openai_request_basic(self):
        """Test OpenAI request formatting."""
        with patch("boto3.Session"):
            model = BedrockModelInvoke(model_id="test-model")
            messages = [{"role": "user", "content": [{"text": "Hello"}]}]
            
            request = model._format_openai_request(messages)
            
            assert request["model"] == "test-model"
            assert request["messages"] == [{"role": "user", "content": "Hello"}]
            assert request["max_tokens"] == 4096
            assert request["stream"] is True

    def test_format_openai_request_with_system_prompt(self):
        """Test OpenAI request formatting with system prompt."""
        with patch("boto3.Session"):
            model = BedrockModelInvoke(model_id="test-model")
            messages = [{"role": "user", "content": [{"text": "Hello"}]}]
            system_prompt_content = [{"text": "You are a helpful assistant"}]
            
            request = model._format_openai_request(messages, system_prompt_content=system_prompt_content)
            
            assert request["messages"][0] == {"role": "system", "content": "You are a helpful assistant"}
            assert request["messages"][1] == {"role": "user", "content": "Hello"}

    def test_format_openai_request_with_tools(self):
        """Test OpenAI request formatting with tools."""
        with patch("boto3.Session"):
            model = BedrockModelInvoke(model_id="test-model")
            messages = [{"role": "user", "content": [{"text": "Hello"}]}]
            tool_specs = [{
                "name": "test_tool",
                "description": "A test tool",
                "inputSchema": {"type": "object", "properties": {}}
            }]
            
            request = model._format_openai_request(messages, tool_specs=tool_specs)
            
            assert "tools" in request
            assert request["tools"][0]["type"] == "function"
            assert request["tools"][0]["function"]["name"] == "test_tool"

    def test_parse_openai_streaming_chunk(self):
        """Test parsing OpenAI streaming chunks."""
        with patch("boto3.Session"):
            model = BedrockModelInvoke(model_id="test-model")
            chunk = {
                "choices": [{
                    "delta": {"content": "Hello"},
                    "finish_reason": None
                }]
            }
            
            event = model._parse_anthropic_streaming_chunk(chunk)
            
            assert event == {"contentBlockDelta": {"delta": {"text": "Hello"}}}

    def test_extract_text_from_response_completion(self):
        """Test text extraction from completion field."""
        with patch("boto3.Session"):
            model = BedrockModelInvoke(model_id="test-model")
            response = {"completion": "Hello world"}
            
            text = model._extract_text_from_response(response)
            
            assert text == "Hello world"

    def test_extract_text_from_response_anthropic_content(self):
        """Test text extraction from Anthropic content format."""
        with patch("boto3.Session"):
            model = BedrockModelInvoke(model_id="test-model")
            response = {
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "world"}
                ]
            }
            
            text = model._extract_text_from_response(response)
            
            assert text == "Hello world"

    def test_extract_usage_from_response_openai_format(self):
        """Test usage extraction from OpenAI format."""
        with patch("boto3.Session"):
            model = BedrockModelInvoke(model_id="test-model")
            response_body = {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            }
            
            usage = model._extract_usage_from_response({}, response_body)
            
            assert usage["inputTokens"] == 10
            assert usage["outputTokens"] == 20
            assert usage["totalTokens"] == 30