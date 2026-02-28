import copy
import logging
import os
import sys
import traceback
import unittest.mock
from unittest.mock import ANY

import boto3
import pydantic
import pytest
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError, EventStreamError

import strands
from strands import _exception_notes
from strands.models import BedrockModel, CacheConfig
from strands.models.bedrock import (
    _DEFAULT_BEDROCK_MODEL_ID,
    DEFAULT_BEDROCK_MODEL_ID,
    DEFAULT_BEDROCK_REGION,
    DEFAULT_READ_TIMEOUT,
)
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException
from strands.types.tools import ToolSpec

FORMATTED_DEFAULT_MODEL_ID = DEFAULT_BEDROCK_MODEL_ID.format("us")


@pytest.fixture
def session_cls():
    # Mock the creation of a Session so that we don't depend on environment variables or profiles
    with unittest.mock.patch.object(strands.models.bedrock.boto3, "Session") as mock_session_cls:
        mock_session = unittest.mock.Mock()
        mock_session.region_name = None
        mock_session_cls.return_value = mock_session
        yield mock_session_cls


@pytest.fixture
def mock_client_method(session_cls):
    # the boto3.Session().client(...) method
    return session_cls.return_value.client


@pytest.fixture
def bedrock_client(session_cls):
    mock_client = session_cls.return_value.client.return_value
    mock_client.meta = unittest.mock.MagicMock()
    mock_client.meta.region_name = "us-west-2"
    yield mock_client


@pytest.fixture
def model_id():
    return "m1"


@pytest.fixture
def model(bedrock_client, model_id):
    _ = bedrock_client

    return BedrockModel(model_id=model_id)


@pytest.fixture
def messages():
    return [{"role": "user", "content": [{"text": "test"}]}]


@pytest.fixture
def system_prompt():
    return "s1"


@pytest.fixture
def additional_request_fields():
    return {"a": 1}


@pytest.fixture
def additional_response_field_paths():
    return ["p1"]


@pytest.fixture
def guardrail_config():
    return {
        "guardrail_id": "g1",
        "guardrail_version": "v1",
        "guardrail_stream_processing_mode": "async",
        "guardrail_trace": "enabled",
    }


@pytest.fixture
def inference_config():
    return {
        "max_tokens": 1,
        "stop_sequences": ["stop"],
        "temperature": 1,
        "top_p": 1,
    }


@pytest.fixture
def tool_spec() -> ToolSpec:
    return {
        "description": "description",
        "name": "name",
        "inputSchema": {"key": "val"},
    }


@pytest.fixture
def cache_type():
    return "default"


@pytest.fixture
def test_output_model_cls():
    class TestOutputModel(pydantic.BaseModel):
        name: str
        age: int

    return TestOutputModel


def test__init__default_model_id(bedrock_client):
    """Test that BedrockModel uses DEFAULT_MODEL_ID when no model_id is provided."""
    _ = bedrock_client
    model = BedrockModel()

    tru_model_id = model.get_config().get("model_id")
    exp_model_id = FORMATTED_DEFAULT_MODEL_ID

    assert tru_model_id == exp_model_id


def test__init__with_default_region(session_cls, mock_client_method):
    """Test that BedrockModel uses the provided region."""
    with unittest.mock.patch.object(os, "environ", {}):
        BedrockModel()
        session_cls.return_value.client.assert_called_with(
            region_name=DEFAULT_BEDROCK_REGION, config=ANY, service_name=ANY, endpoint_url=None
        )


def test__init__with_session_region(session_cls, mock_client_method):
    """Test that BedrockModel uses the provided region."""
    session_cls.return_value.region_name = "eu-blah-1"

    BedrockModel()

    mock_client_method.assert_called_with(region_name="eu-blah-1", config=ANY, service_name=ANY, endpoint_url=None)


def test__init__with_custom_region(mock_client_method):
    """Test that BedrockModel uses the provided region."""
    custom_region = "us-east-1"
    BedrockModel(region_name=custom_region)
    mock_client_method.assert_called_with(region_name=custom_region, config=ANY, service_name=ANY, endpoint_url=None)


def test__init__with_default_environment_variable_region(mock_client_method):
    """Test that BedrockModel uses the AWS_REGION since we code that in."""
    with unittest.mock.patch.object(os, "environ", {"AWS_REGION": "eu-west-2"}):
        BedrockModel()

    mock_client_method.assert_called_with(region_name="eu-west-2", config=ANY, service_name=ANY, endpoint_url=None)


def test__init__region_precedence(mock_client_method, session_cls):
    """Test that BedrockModel uses the correct ordering of precedence when determining region."""
    with unittest.mock.patch.object(os, "environ", {"AWS_REGION": "us-environment-1"}) as mock_os_environ:
        session_cls.return_value.region_name = "us-session-1"

        # specifying a region always wins out
        BedrockModel(region_name="us-specified-1")
        mock_client_method.assert_called_with(
            region_name="us-specified-1", config=ANY, service_name=ANY, endpoint_url=None
        )

        # other-wise uses the session's
        BedrockModel()
        mock_client_method.assert_called_with(
            region_name="us-session-1", config=ANY, service_name=ANY, endpoint_url=None
        )

        # environment variable next
        session_cls.return_value.region_name = None
        BedrockModel()
        mock_client_method.assert_called_with(
            region_name="us-environment-1", config=ANY, service_name=ANY, endpoint_url=None
        )

        mock_os_environ.pop("AWS_REGION")
        session_cls.return_value.region_name = None  # No session region
        BedrockModel()
        mock_client_method.assert_called_with(
            region_name=DEFAULT_BEDROCK_REGION, config=ANY, service_name=ANY, endpoint_url=None
        )


def test__init__with_endpoint_url(mock_client_method):
    """Test that BedrockModel uses the provided endpoint_url for VPC endpoints."""
    custom_endpoint = "https://vpce-12345-abcde.bedrock-runtime.us-west-2.vpce.amazonaws.com"
    with unittest.mock.patch.object(os, "environ", {}):
        BedrockModel(endpoint_url=custom_endpoint)
        mock_client_method.assert_called_with(
            region_name=DEFAULT_BEDROCK_REGION, config=ANY, service_name=ANY, endpoint_url=custom_endpoint
        )


def test__init__with_region_and_session_raises_value_error():
    """Test that BedrockModel raises ValueError when both region and session are provided."""
    with pytest.raises(ValueError):
        _ = BedrockModel(region_name="us-east-1", boto_session=boto3.Session(region_name="us-east-1"))


def test__init__default_user_agent(session_cls, bedrock_client):
    """Set user agent when no boto_client_config is provided."""
    _ = BedrockModel()

    # Verify the client was created with the correct config
    client = session_cls.return_value.client
    client.assert_called_once()
    args, kwargs = client.call_args
    assert kwargs["service_name"] == "bedrock-runtime"
    assert isinstance(kwargs["config"], BotocoreConfig)
    assert kwargs["config"].user_agent_extra == "strands-agents"
    assert kwargs["config"].read_timeout == DEFAULT_READ_TIMEOUT


def test__init__default_read_timeout(session_cls, bedrock_client):
    """Set default read timeout when no boto_client_config is provided."""

    _ = BedrockModel()

    # Verify the client was created with the correct read timeout
    client = session_cls.return_value.client
    client.assert_called_once()
    args, kwargs = client.call_args
    assert isinstance(kwargs["config"], BotocoreConfig)
    assert kwargs["config"].read_timeout == DEFAULT_READ_TIMEOUT


def test__init__with_custom_boto_client_config_no_user_agent(session_cls, bedrock_client):
    """Set user agent when boto_client_config is provided without user_agent_extra."""
    custom_config = BotocoreConfig(read_timeout=900)

    _ = BedrockModel(boto_client_config=custom_config)

    # Verify the client was created with the correct config
    client = session_cls.return_value.client
    client.assert_called_once()
    args, kwargs = client.call_args
    assert kwargs["service_name"] == "bedrock-runtime"
    assert isinstance(kwargs["config"], BotocoreConfig)
    assert kwargs["config"].user_agent_extra == "strands-agents"
    assert kwargs["config"].read_timeout == 900


def test__init__with_custom_boto_client_config_with_user_agent(session_cls, bedrock_client):
    """Append to existing user agent when boto_client_config is provided with user_agent_extra."""
    custom_config = BotocoreConfig(user_agent_extra="existing-agent", read_timeout=900)

    _ = BedrockModel(boto_client_config=custom_config)

    # Verify the client was created with the correct config
    client = session_cls.return_value.client
    client.assert_called_once()
    args, kwargs = client.call_args
    assert kwargs["service_name"] == "bedrock-runtime"
    assert isinstance(kwargs["config"], BotocoreConfig)
    assert kwargs["config"].user_agent_extra == "existing-agent strands-agents"
    assert kwargs["config"].read_timeout == 900


def test__init__model_config(bedrock_client):
    _ = bedrock_client

    model = BedrockModel(max_tokens=1)

    tru_max_tokens = model.get_config().get("max_tokens")
    exp_max_tokens = 1

    assert tru_max_tokens == exp_max_tokens


def test_update_config(model, model_id):
    model.update_config(model_id=model_id)

    tru_model_id = model.get_config().get("model_id")
    exp_model_id = model_id

    assert tru_model_id == exp_model_id


def test_format_request_default(model, messages, model_id):
    tru_request = model._format_request(messages)
    exp_request = {
        "inferenceConfig": {},
        "modelId": model_id,
        "messages": messages,
        "system": [],
    }

    assert tru_request == exp_request


def test_format_request_additional_request_fields(model, messages, model_id, additional_request_fields):
    model.update_config(additional_request_fields=additional_request_fields)
    tru_request = model._format_request(messages)
    exp_request = {
        "additionalModelRequestFields": additional_request_fields,
        "inferenceConfig": {},
        "modelId": model_id,
        "messages": messages,
        "system": [],
    }

    assert tru_request == exp_request


def test_format_request_additional_response_field_paths(model, messages, model_id, additional_response_field_paths):
    model.update_config(additional_response_field_paths=additional_response_field_paths)
    tru_request = model._format_request(messages)
    exp_request = {
        "additionalModelResponseFieldPaths": additional_response_field_paths,
        "inferenceConfig": {},
        "modelId": model_id,
        "messages": messages,
        "system": [],
    }

    assert tru_request == exp_request


def test_format_request_guardrail_config(model, messages, model_id, guardrail_config):
    model.update_config(**guardrail_config)
    tru_request = model._format_request(messages)
    exp_request = {
        "guardrailConfig": {
            "guardrailIdentifier": guardrail_config["guardrail_id"],
            "guardrailVersion": guardrail_config["guardrail_version"],
            "trace": guardrail_config["guardrail_trace"],
            "streamProcessingMode": guardrail_config["guardrail_stream_processing_mode"],
        },
        "inferenceConfig": {},
        "modelId": model_id,
        "messages": messages,
        "system": [],
    }

    assert tru_request == exp_request


def test_format_request_guardrail_config_without_trace_or_stream_processing_mode(model, messages, model_id):
    model.update_config(
        **{
            "guardrail_id": "g1",
            "guardrail_version": "v1",
        }
    )
    tru_request = model._format_request(messages)
    exp_request = {
        "guardrailConfig": {
            "guardrailIdentifier": "g1",
            "guardrailVersion": "v1",
            "trace": "enabled",
        },
        "inferenceConfig": {},
        "modelId": model_id,
        "messages": messages,
        "system": [],
    }

    assert tru_request == exp_request


def test_format_request_inference_config(model, messages, model_id, inference_config):
    model.update_config(**inference_config)
    tru_request = model._format_request(messages)
    exp_request = {
        "inferenceConfig": {
            "maxTokens": inference_config["max_tokens"],
            "stopSequences": inference_config["stop_sequences"],
            "temperature": inference_config["temperature"],
            "topP": inference_config["top_p"],
        },
        "modelId": model_id,
        "messages": messages,
        "system": [],
    }

    assert tru_request == exp_request


def test_format_request_system_prompt(model, messages, model_id, system_prompt):
    tru_request = model._format_request(messages, system_prompt_content=[{"text": system_prompt}])
    exp_request = {
        "inferenceConfig": {},
        "modelId": model_id,
        "messages": messages,
        "system": [{"text": system_prompt}],
    }

    assert tru_request == exp_request


def test_format_request_system_prompt_content(model, messages, model_id):
    """Test _format_request with SystemContentBlock input."""
    system_prompt_content = [{"text": "You are a helpful assistant."}, {"cachePoint": {"type": "default"}}]

    tru_request = model._format_request(messages, system_prompt_content=system_prompt_content)
    exp_request = {
        "inferenceConfig": {},
        "modelId": model_id,
        "messages": messages,
        "system": system_prompt_content,
    }

    assert tru_request == exp_request


def test_format_request_system_prompt_content_with_cache_prompt_config(model, messages, model_id):
    """Test _format_request with SystemContentBlock and cache_prompt config (backwards compatibility)."""
    system_prompt_content = [{"text": "You are a helpful assistant."}]
    model.update_config(cache_prompt="default")

    with pytest.warns(UserWarning, match="cache_prompt is deprecated"):
        tru_request = model._format_request(messages, system_prompt_content=system_prompt_content)

    exp_request = {
        "inferenceConfig": {},
        "modelId": model_id,
        "messages": messages,
        "system": [{"text": "You are a helpful assistant."}, {"cachePoint": {"type": "default"}}],
    }

    assert tru_request == exp_request


def test_format_request_empty_system_prompt_content(model, messages, model_id):
    """Test _format_request with empty SystemContentBlock list."""
    tru_request = model._format_request(messages, system_prompt_content=[])
    exp_request = {
        "inferenceConfig": {},
        "modelId": model_id,
        "messages": messages,
        "system": [],
    }

    assert tru_request == exp_request


def test_format_request_tool_specs(model, messages, model_id, tool_spec):
    tru_request = model._format_request(messages, tool_specs=[tool_spec])
    exp_request = {
        "inferenceConfig": {},
        "modelId": model_id,
        "messages": messages,
        "system": [],
        "toolConfig": {
            "tools": [{"toolSpec": tool_spec}],
            "toolChoice": {"auto": {}},
        },
    }

    assert tru_request == exp_request


def test_format_request_tool_choice_auto(model, messages, model_id, tool_spec):
    tool_choice = {"auto": {}}
    tru_request = model._format_request(messages, [tool_spec], tool_choice=tool_choice)
    exp_request = {
        "inferenceConfig": {},
        "modelId": model_id,
        "messages": messages,
        "system": [],
        "toolConfig": {
            "tools": [{"toolSpec": tool_spec}],
            "toolChoice": tool_choice,
        },
    }

    assert tru_request == exp_request


def test_format_request_tool_choice_any(model, messages, model_id, tool_spec):
    tool_choice = {"any": {}}
    tru_request = model._format_request(messages, [tool_spec], tool_choice=tool_choice)
    exp_request = {
        "inferenceConfig": {},
        "modelId": model_id,
        "messages": messages,
        "system": [],
        "toolConfig": {
            "tools": [{"toolSpec": tool_spec}],
            "toolChoice": tool_choice,
        },
    }

    assert tru_request == exp_request


def test_format_request_tool_choice_tool(model, messages, model_id, tool_spec):
    tool_choice = {"tool": {"name": "test_tool"}}
    tru_request = model._format_request(messages, [tool_spec], tool_choice=tool_choice)
    exp_request = {
        "inferenceConfig": {},
        "modelId": model_id,
        "messages": messages,
        "system": [],
        "toolConfig": {
            "tools": [{"toolSpec": tool_spec}],
            "toolChoice": tool_choice,
        },
    }

    assert tru_request == exp_request


def test_format_request_cache(model, messages, model_id, tool_spec, cache_type):
    model.update_config(cache_prompt=cache_type, cache_tools=cache_type)

    with pytest.warns(UserWarning, match="cache_prompt is deprecated"):
        tru_request = model._format_request(messages, tool_specs=[tool_spec])

    exp_request = {
        "inferenceConfig": {},
        "modelId": model_id,
        "messages": messages,
        "system": [{"cachePoint": {"type": cache_type}}],
        "toolConfig": {
            "tools": [
                {"toolSpec": tool_spec},
                {"cachePoint": {"type": cache_type}},
            ],
            "toolChoice": {"auto": {}},
        },
    }

    assert tru_request == exp_request


@pytest.mark.asyncio
async def test_stream_throttling_exception_from_event_stream_error(bedrock_client, model, messages, alist):
    error_message = "Rate exceeded"
    bedrock_client.converse_stream.side_effect = EventStreamError(
        {"Error": {"Message": error_message, "Code": "ThrottlingException"}}, "ConverseStream"
    )

    with pytest.raises(ModelThrottledException) as excinfo:
        await alist(model.stream(messages))

    assert error_message in str(excinfo.value)
    bedrock_client.converse_stream.assert_called_once_with(
        modelId="m1", messages=messages, system=[], inferenceConfig={}
    )


@pytest.mark.asyncio
async def test_stream_with_invalid_content_throws(bedrock_client, model, alist):
    # We used to hang on None, so ensure we don't regress: https://github.com/strands-agents/sdk-python/issues/642
    messages = [{"role": "user", "content": None}]

    with pytest.raises(TypeError):
        await alist(model.stream(messages))


@pytest.mark.asyncio
async def test_stream_throttling_exception_from_general_exception(bedrock_client, model, messages, alist):
    error_message = "ThrottlingException: Rate exceeded for ConverseStream"
    bedrock_client.converse_stream.side_effect = ClientError(
        {"Error": {"Message": error_message, "Code": "ThrottlingException"}}, "Any"
    )

    with pytest.raises(ModelThrottledException) as excinfo:
        await alist(model.stream(messages))

    assert error_message in str(excinfo.value)
    bedrock_client.converse_stream.assert_called_once_with(
        modelId="m1", messages=messages, system=[], inferenceConfig={}
    )


@pytest.mark.asyncio
async def test_stream_throttling_exception_lowercase(bedrock_client, model, messages, alist):
    """Test that lowercase throttlingException is converted to ModelThrottledException."""
    error_message = "throttlingException: Rate exceeded for ConverseStream"
    bedrock_client.converse_stream.side_effect = ClientError(
        {"Error": {"Message": error_message, "Code": "throttlingException"}}, "Any"
    )

    with pytest.raises(ModelThrottledException) as excinfo:
        await alist(model.stream(messages))

    assert error_message in str(excinfo.value)
    bedrock_client.converse_stream.assert_called_once_with(
        modelId="m1", messages=messages, system=[], inferenceConfig={}
    )


@pytest.mark.asyncio
async def test_stream_throttling_exception_lowercase_non_streaming(bedrock_client, messages, alist):
    """Test that lowercase throttlingException is converted to ModelThrottledException in non-streaming mode."""
    error_message = "throttlingException: Rate exceeded for Converse"
    bedrock_client.converse.side_effect = ClientError(
        {"Error": {"Message": error_message, "Code": "throttlingException"}}, "Any"
    )

    model = BedrockModel(model_id="test-model", streaming=False)
    with pytest.raises(ModelThrottledException) as excinfo:
        await alist(model.stream(messages))

    assert error_message in str(excinfo.value)
    bedrock_client.converse.assert_called_once()
    bedrock_client.converse_stream.assert_not_called()


@pytest.mark.asyncio
async def test_general_exception_is_raised(bedrock_client, model, messages, alist):
    error_message = "Should be raised up"
    bedrock_client.converse_stream.side_effect = ValueError(error_message)

    with pytest.raises(ValueError) as excinfo:
        await alist(model.stream(messages))

    assert error_message in str(excinfo.value)
    bedrock_client.converse_stream.assert_called_once_with(
        modelId="m1", messages=messages, system=[], inferenceConfig={}
    )


@pytest.mark.asyncio
async def test_stream(bedrock_client, model, messages, tool_spec, model_id, additional_request_fields, alist):
    bedrock_client.converse_stream.return_value = {"stream": ["e1", "e2"]}

    request = {
        "additionalModelRequestFields": additional_request_fields,
        "inferenceConfig": {},
        "modelId": model_id,
        "messages": messages,
        "system": [],
        "toolConfig": {
            "tools": [{"toolSpec": tool_spec}],
            "toolChoice": {"auto": {}},
        },
    }

    model.update_config(additional_request_fields=additional_request_fields)
    response = model.stream(messages, [tool_spec])

    tru_chunks = await alist(response)
    exp_chunks = ["e1", "e2"]

    assert tru_chunks == exp_chunks
    bedrock_client.converse_stream.assert_called_once_with(**request)


@pytest.mark.asyncio
async def test_stream_with_system_prompt_content(bedrock_client, model, messages, alist):
    """Test stream method with system_prompt_content parameter."""
    bedrock_client.converse_stream.return_value = {"stream": ["e1", "e2"]}

    system_prompt_content = [{"text": "You are a helpful assistant."}, {"cachePoint": {"type": "default"}}]

    response = model.stream(messages, system_prompt_content=system_prompt_content)
    tru_chunks = await alist(response)
    exp_chunks = ["e1", "e2"]

    assert tru_chunks == exp_chunks

    # Verify the request was formatted with system_prompt_content
    expected_request = {
        "inferenceConfig": {},
        "modelId": "m1",
        "messages": messages,
        "system": system_prompt_content,
    }
    bedrock_client.converse_stream.assert_called_once_with(**expected_request)


@pytest.mark.asyncio
async def test_stream_backwards_compatibility_single_text_block(bedrock_client, model, messages, alist):
    """Test that single text block in system_prompt_content works with legacy system_prompt."""
    bedrock_client.converse_stream.return_value = {"stream": ["e1", "e2"]}

    system_prompt_content = [{"text": "You are a helpful assistant."}]

    response = model.stream(
        messages, system_prompt="You are a helpful assistant.", system_prompt_content=system_prompt_content
    )
    await alist(response)

    # Verify the request was formatted with system_prompt_content
    expected_request = {
        "inferenceConfig": {},
        "modelId": "m1",
        "messages": messages,
        "system": system_prompt_content,
    }
    bedrock_client.converse_stream.assert_called_once_with(**expected_request)


@pytest.mark.asyncio
async def test_stream_stream_input_guardrails(
    bedrock_client, model, messages, tool_spec, model_id, additional_request_fields, alist
):
    metadata_event = {
        "metadata": {
            "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
            "metrics": {"latencyMs": 245},
            "trace": {
                "guardrail": {
                    "inputAssessment": {
                        "3e59qlue4hag": {
                            "wordPolicy": {
                                "customWords": [
                                    {
                                        "match": "CACTUS",
                                        "action": "BLOCKED",
                                        "detected": True,
                                    }
                                ]
                            }
                        }
                    }
                }
            },
        }
    }
    bedrock_client.converse_stream.return_value = {"stream": [metadata_event]}

    request = {
        "additionalModelRequestFields": additional_request_fields,
        "inferenceConfig": {},
        "modelId": model_id,
        "messages": messages,
        "system": [],
        "toolConfig": {
            "tools": [{"toolSpec": tool_spec}],
            "toolChoice": {"auto": {}},
        },
    }

    model.update_config(additional_request_fields=additional_request_fields)
    response = model.stream(messages, [tool_spec])

    tru_chunks = await alist(response)
    exp_chunks = [
        {"redactContent": {"redactUserContentMessage": "[User input redacted.]"}},
        metadata_event,
    ]

    assert tru_chunks == exp_chunks
    bedrock_client.converse_stream.assert_called_once_with(**request)


@pytest.mark.asyncio
async def test_stream_stream_input_guardrails_full_trace(
    bedrock_client, model, messages, tool_spec, model_id, additional_request_fields, alist
):
    """Test guardrails are correctly detected also with guardrail_trace="enabled_full".
    In that case bedrock returns all filters, including those not detected/blocked."""
    metadata_event = {
        "metadata": {
            "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
            "metrics": {"latencyMs": 245},
            "trace": {
                "guardrail": {
                    "inputAssessment": {
                        "jrv9qlue4hag": {
                            "contentPolicy": {
                                "filters": [
                                    {
                                        "action": "NONE",
                                        "confidence": "NONE",
                                        "detected": False,
                                        "filterStrength": "HIGH",
                                        "type": "SEXUAL",
                                    },
                                    {
                                        "action": "BLOCKED",
                                        "confidence": "LOW",
                                        "detected": True,
                                        "filterStrength": "HIGH",
                                        "type": "VIOLENCE",
                                    },
                                    {
                                        "action": "NONE",
                                        "confidence": "NONE",
                                        "detected": False,
                                        "filterStrength": "HIGH",
                                        "type": "HATE",
                                    },
                                    {
                                        "action": "NONE",
                                        "confidence": "NONE",
                                        "detected": False,
                                        "filterStrength": "HIGH",
                                        "type": "INSULTS",
                                    },
                                    {
                                        "action": "NONE",
                                        "confidence": "NONE",
                                        "detected": False,
                                        "filterStrength": "HIGH",
                                        "type": "PROMPT_ATTACK",
                                    },
                                    {
                                        "action": "NONE",
                                        "confidence": "NONE",
                                        "detected": False,
                                        "filterStrength": "HIGH",
                                        "type": "MISCONDUCT",
                                    },
                                ]
                            }
                        }
                    }
                }
            },
        }
    }
    bedrock_client.converse_stream.return_value = {"stream": [metadata_event]}

    request = {
        "additionalModelRequestFields": additional_request_fields,
        "inferenceConfig": {},
        "modelId": model_id,
        "messages": messages,
        "system": [],
        "toolConfig": {
            "tools": [{"toolSpec": tool_spec}],
            "toolChoice": {"auto": {}},
        },
    }

    model.update_config(additional_request_fields=additional_request_fields)
    response = model.stream(messages, [tool_spec])

    tru_chunks = await alist(response)
    exp_chunks = [
        {"redactContent": {"redactUserContentMessage": "[User input redacted.]"}},
        metadata_event,
    ]

    assert tru_chunks == exp_chunks
    bedrock_client.converse_stream.assert_called_once_with(**request)


@pytest.mark.asyncio
async def test_stream_stream_output_guardrails(
    bedrock_client, model, messages, tool_spec, model_id, additional_request_fields, alist
):
    model.update_config(guardrail_redact_input=False, guardrail_redact_output=True)
    metadata_event = {
        "metadata": {
            "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
            "metrics": {"latencyMs": 245},
            "trace": {
                "guardrail": {
                    "outputAssessments": {
                        "3e59qlue4hag": [
                            {
                                "wordPolicy": {
                                    "customWords": [
                                        {
                                            "match": "CACTUS",
                                            "action": "BLOCKED",
                                            "detected": True,
                                        }
                                    ]
                                },
                            }
                        ]
                    },
                }
            },
        }
    }
    bedrock_client.converse_stream.return_value = {"stream": [metadata_event]}

    request = {
        "additionalModelRequestFields": additional_request_fields,
        "inferenceConfig": {},
        "modelId": model_id,
        "messages": messages,
        "system": [],
        "toolConfig": {
            "tools": [{"toolSpec": tool_spec}],
            "toolChoice": {"auto": {}},
        },
    }

    model.update_config(additional_request_fields=additional_request_fields)
    response = model.stream(messages, [tool_spec])

    tru_chunks = await alist(response)
    exp_chunks = [
        {"redactContent": {"redactAssistantContentMessage": "[Assistant output redacted.]"}},
        metadata_event,
    ]

    assert tru_chunks == exp_chunks
    bedrock_client.converse_stream.assert_called_once_with(**request)


@pytest.mark.asyncio
async def test_stream_output_guardrails_redacts_input_and_output(
    bedrock_client, model, messages, tool_spec, model_id, additional_request_fields, alist
):
    model.update_config(guardrail_redact_output=True)
    metadata_event = {
        "metadata": {
            "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
            "metrics": {"latencyMs": 245},
            "trace": {
                "guardrail": {
                    "outputAssessments": {
                        "3e59qlue4hag": [
                            {
                                "wordPolicy": {
                                    "customWords": [
                                        {
                                            "match": "CACTUS",
                                            "action": "BLOCKED",
                                            "detected": True,
                                        }
                                    ]
                                },
                            }
                        ]
                    },
                }
            },
        }
    }
    bedrock_client.converse_stream.return_value = {"stream": [metadata_event]}

    request = {
        "additionalModelRequestFields": additional_request_fields,
        "inferenceConfig": {},
        "modelId": model_id,
        "messages": messages,
        "system": [],
        "toolConfig": {
            "tools": [{"toolSpec": tool_spec}],
            "toolChoice": {"auto": {}},
        },
    }

    model.update_config(additional_request_fields=additional_request_fields)
    response = model.stream(messages, [tool_spec])

    tru_chunks = await alist(response)
    exp_chunks = [
        {"redactContent": {"redactUserContentMessage": "[User input redacted.]"}},
        {"redactContent": {"redactAssistantContentMessage": "[Assistant output redacted.]"}},
        metadata_event,
    ]

    assert tru_chunks == exp_chunks
    bedrock_client.converse_stream.assert_called_once_with(**request)


@pytest.mark.asyncio
async def test_stream_output_no_blocked_guardrails_doesnt_redact(
    bedrock_client, model, messages, tool_spec, model_id, additional_request_fields, alist
):
    metadata_event = {
        "metadata": {
            "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
            "metrics": {"latencyMs": 245},
            "trace": {
                "guardrail": {
                    "outputAssessments": {
                        "3e59qlue4hag": [
                            {
                                "wordPolicy": {
                                    "customWords": [
                                        {
                                            "match": "CACTUS",
                                            "action": "NONE",
                                            "detected": True,
                                        }
                                    ]
                                },
                            }
                        ]
                    },
                }
            },
        }
    }
    bedrock_client.converse_stream.return_value = {"stream": [metadata_event]}

    request = {
        "additionalModelRequestFields": additional_request_fields,
        "inferenceConfig": {},
        "modelId": model_id,
        "messages": messages,
        "system": [],
        "toolConfig": {
            "tools": [{"toolSpec": tool_spec}],
            "toolChoice": {"auto": {}},
        },
    }

    model.update_config(additional_request_fields=additional_request_fields)
    response = model.stream(messages, [tool_spec])

    tru_chunks = await alist(response)
    exp_chunks = [metadata_event]

    assert tru_chunks == exp_chunks
    bedrock_client.converse_stream.assert_called_once_with(**request)


@pytest.mark.asyncio
async def test_stream_output_no_guardrail_redact(
    bedrock_client, model, messages, tool_spec, model_id, additional_request_fields, alist
):
    metadata_event = {
        "metadata": {
            "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
            "metrics": {"latencyMs": 245},
            "trace": {
                "guardrail": {
                    "outputAssessments": {
                        "3e59qlue4hag": [
                            {
                                "wordPolicy": {
                                    "customWords": [
                                        {
                                            "match": "CACTUS",
                                            "action": "BLOCKED",
                                            "detected": True,
                                        }
                                    ]
                                },
                            }
                        ]
                    },
                }
            },
        }
    }
    bedrock_client.converse_stream.return_value = {"stream": [metadata_event]}

    request = {
        "additionalModelRequestFields": additional_request_fields,
        "inferenceConfig": {},
        "modelId": model_id,
        "messages": messages,
        "system": [],
        "toolConfig": {
            "tools": [{"toolSpec": tool_spec}],
            "toolChoice": {"auto": {}},
        },
    }

    model.update_config(
        additional_request_fields=additional_request_fields,
        guardrail_redact_output=False,
        guardrail_redact_input=False,
    )
    response = model.stream(messages, [tool_spec])

    tru_chunks = await alist(response)
    exp_chunks = [metadata_event]

    assert tru_chunks == exp_chunks
    bedrock_client.converse_stream.assert_called_once_with(**request)


@pytest.mark.asyncio
async def test_stream_with_streaming_false(bedrock_client, alist, messages):
    """Test stream method with streaming=False."""
    bedrock_client.converse.return_value = {
        "output": {"message": {"role": "assistant", "content": [{"text": "test"}]}},
        "stopReason": "end_turn",
    }

    # Create model and call stream
    model = BedrockModel(model_id="test-model", streaming=False)
    response = model.stream(messages)

    tru_events = await alist(response)
    exp_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "test"}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn", "additionalModelResponseFields": None}},
    ]
    assert tru_events == exp_events

    bedrock_client.converse.assert_called_once()
    bedrock_client.converse_stream.assert_not_called()


@pytest.mark.asyncio
async def test_stream_with_streaming_false_and_tool_use(bedrock_client, alist, messages):
    """Test stream method with streaming=False."""
    bedrock_client.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "123", "name": "dummyTool", "input": {"hello": "world!"}}}],
            }
        },
        "stopReason": "tool_use",
    }

    # Create model and call stream
    model = BedrockModel(model_id="test-model", streaming=False)
    response = model.stream(messages)

    tru_events = await alist(response)
    exp_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "123", "name": "dummyTool"}}}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"hello": "world!"}'}}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "tool_use", "additionalModelResponseFields": None}},
    ]
    assert tru_events == exp_events

    bedrock_client.converse.assert_called_once()
    bedrock_client.converse_stream.assert_not_called()


@pytest.mark.asyncio
async def test_stream_with_streaming_false_and_reasoning(bedrock_client, alist, messages):
    """Test stream method with streaming=False."""
    bedrock_client.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {"text": "Thinking really hard....", "signature": "123"},
                        }
                    }
                ],
            }
        },
        "stopReason": "tool_use",
    }

    # Create model and call stream
    model = BedrockModel(model_id="test-model", streaming=False)
    response = model.stream(messages)

    tru_events = await alist(response)
    exp_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "Thinking really hard...."}}}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"signature": "123"}}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "tool_use", "additionalModelResponseFields": None}},
    ]
    assert tru_events == exp_events

    # Verify converse was called
    bedrock_client.converse.assert_called_once()
    bedrock_client.converse_stream.assert_not_called()


@pytest.mark.asyncio
async def test_stream_and_reasoning_no_signature(bedrock_client, alist, messages):
    """Test stream method with streaming=False."""
    bedrock_client.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {"text": "Thinking really hard...."},
                        }
                    }
                ],
            }
        },
        "stopReason": "tool_use",
    }

    # Create model and call stream
    model = BedrockModel(model_id="test-model", streaming=False)
    response = model.stream(messages)

    tru_events = await alist(response)
    exp_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "Thinking really hard...."}}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "tool_use", "additionalModelResponseFields": None}},
    ]
    assert tru_events == exp_events

    bedrock_client.converse.assert_called_once()
    bedrock_client.converse_stream.assert_not_called()


@pytest.mark.asyncio
async def test_stream_with_streaming_false_with_metrics_and_usage(bedrock_client, alist, messages):
    """Test stream method with streaming=False."""
    bedrock_client.converse.return_value = {
        "output": {"message": {"role": "assistant", "content": [{"text": "test"}]}},
        "usage": {"inputTokens": 1234, "outputTokens": 1234, "totalTokens": 2468},
        "metrics": {"latencyMs": 1234},
        "stopReason": "tool_use",
    }

    # Create model and call stream
    model = BedrockModel(model_id="test-model", streaming=False)
    response = model.stream(messages)

    tru_events = await alist(response)
    exp_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "test"}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "tool_use", "additionalModelResponseFields": None}},
        {
            "metadata": {
                "usage": {"inputTokens": 1234, "outputTokens": 1234, "totalTokens": 2468},
                "metrics": {"latencyMs": 1234},
            }
        },
    ]
    assert tru_events == exp_events

    # Verify converse was called
    bedrock_client.converse.assert_called_once()
    bedrock_client.converse_stream.assert_not_called()


@pytest.mark.asyncio
async def test_stream_input_guardrails(bedrock_client, alist, messages):
    """Test stream method with streaming=False."""
    bedrock_client.converse.return_value = {
        "output": {"message": {"role": "assistant", "content": [{"text": "test"}]}},
        "trace": {
            "guardrail": {
                "inputAssessment": {
                    "3e59qlue4hag": {
                        "wordPolicy": {"customWords": [{"match": "CACTUS", "action": "BLOCKED", "detected": True}]}
                    }
                }
            }
        },
        "stopReason": "end_turn",
    }

    # Create model and call stream
    model = BedrockModel(model_id="test-model", streaming=False)
    response = model.stream(messages)

    tru_events = await alist(response)
    exp_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "test"}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn", "additionalModelResponseFields": None}},
        {
            "metadata": {
                "trace": {
                    "guardrail": {
                        "inputAssessment": {
                            "3e59qlue4hag": {
                                "wordPolicy": {
                                    "customWords": [{"match": "CACTUS", "action": "BLOCKED", "detected": True}]
                                }
                            }
                        }
                    }
                }
            }
        },
        {"redactContent": {"redactUserContentMessage": "[User input redacted.]"}},
    ]
    assert tru_events == exp_events

    bedrock_client.converse.assert_called_once()
    bedrock_client.converse_stream.assert_not_called()


@pytest.mark.asyncio
async def test_stream_output_guardrails(bedrock_client, alist, messages):
    """Test stream method with streaming=False."""
    bedrock_client.converse.return_value = {
        "output": {"message": {"role": "assistant", "content": [{"text": "test"}]}},
        "trace": {
            "guardrail": {
                "outputAssessments": {
                    "3e59qlue4hag": [
                        {
                            "wordPolicy": {"customWords": [{"match": "CACTUS", "action": "BLOCKED", "detected": True}]},
                        }
                    ]
                },
            }
        },
        "stopReason": "end_turn",
    }

    model = BedrockModel(model_id="test-model", streaming=False)
    response = model.stream(messages)

    tru_events = await alist(response)
    exp_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "test"}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn", "additionalModelResponseFields": None}},
        {
            "metadata": {
                "trace": {
                    "guardrail": {
                        "outputAssessments": {
                            "3e59qlue4hag": [
                                {
                                    "wordPolicy": {
                                        "customWords": [{"match": "CACTUS", "action": "BLOCKED", "detected": True}]
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        },
        {"redactContent": {"redactUserContentMessage": "[User input redacted.]"}},
    ]
    assert tru_events == exp_events

    bedrock_client.converse.assert_called_once()
    bedrock_client.converse_stream.assert_not_called()


@pytest.mark.asyncio
async def test_stream_output_guardrails_redacts_output(bedrock_client, alist, messages):
    """Test stream method with streaming=False."""
    bedrock_client.converse.return_value = {
        "output": {"message": {"role": "assistant", "content": [{"text": "test"}]}},
        "trace": {
            "guardrail": {
                "outputAssessments": {
                    "3e59qlue4hag": [
                        {
                            "wordPolicy": {"customWords": [{"match": "CACTUS", "action": "BLOCKED", "detected": True}]},
                        }
                    ]
                },
            }
        },
        "stopReason": "end_turn",
    }

    model = BedrockModel(model_id="test-model", streaming=False)
    response = model.stream(messages)

    tru_events = await alist(response)
    exp_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "test"}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn", "additionalModelResponseFields": None}},
        {
            "metadata": {
                "trace": {
                    "guardrail": {
                        "outputAssessments": {
                            "3e59qlue4hag": [
                                {
                                    "wordPolicy": {
                                        "customWords": [{"match": "CACTUS", "action": "BLOCKED", "detected": True}]
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        },
        {"redactContent": {"redactUserContentMessage": "[User input redacted.]"}},
    ]
    assert tru_events == exp_events

    bedrock_client.converse.assert_called_once()
    bedrock_client.converse_stream.assert_not_called()


@pytest.mark.asyncio
async def test_structured_output(bedrock_client, model, test_output_model_cls, alist):
    messages = [{"role": "user", "content": [{"text": "Generate a person"}]}]

    bedrock_client.converse_stream.return_value = {
        "stream": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "123", "name": "TestOutputModel"}}}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"name": "John", "age": 30}'}}}},
            {"contentBlockStop": {}},
            {"messageStop": {"stopReason": "tool_use"}},
        ]
    }

    stream = model.structured_output(test_output_model_cls, messages)
    events = await alist(stream)

    tru_output = events[-1]
    exp_output = {"output": test_output_model_cls(name="John", age=30)}
    assert tru_output == exp_output


@pytest.mark.skipif(sys.version_info < (3, 11), reason="This test requires Python 3.11 or higher (need add_note)")
@pytest.mark.asyncio
async def test_add_note_on_client_error(bedrock_client, model, alist, messages):
    """Test that add_note is called on ClientError with region and model ID information."""
    # Mock the client error response
    error_response = {"Error": {"Code": "ValidationException", "Message": "Some error message"}}
    bedrock_client.converse_stream.side_effect = ClientError(error_response, "ConversationStream")

    # Call the stream method which should catch and add notes to the exception
    with pytest.raises(ClientError) as err:
        await alist(model.stream(messages))

    assert err.value.__notes__ == ["└ Bedrock region: us-west-2", "└ Model id: m1"]


@pytest.mark.asyncio
async def test_add_note_on_client_error_without_add_notes(bedrock_client, model, alist, messages):
    """Test that when add_note is not used, the region & model are still included in the error output."""
    with unittest.mock.patch.object(_exception_notes, "supports_add_note", False):
        # Mock the client error response
        error_response = {"Error": {"Code": "ValidationException", "Message": "Some error message"}}
        bedrock_client.converse_stream.side_effect = ClientError(error_response, "ConversationStream")

        # Call the stream method which should catch and add notes to the exception
        with pytest.raises(ClientError) as err:
            await alist(model.stream(messages))

    error_str = "".join(traceback.format_exception(err.value))
    assert "└ Bedrock region: us-west-2" in error_str
    assert "└ Model id: m1" in error_str


@pytest.mark.asyncio
async def test_no_add_note_when_not_available(bedrock_client, model, alist, messages):
    """Verify that on any python version (even < 3.11 where add_note is not available, we get the right exception)."""
    # Mock the client error response
    error_response = {"Error": {"Code": "ValidationException", "Message": "Some error message"}}
    bedrock_client.converse_stream.side_effect = ClientError(error_response, "ConversationStream")

    # Call the stream method which should catch and add notes to the exception
    with pytest.raises(ClientError):
        await alist(model.stream(messages))


@pytest.mark.skipif(sys.version_info < (3, 11), reason="This test requires Python 3.11 or higher (need add_note)")
@pytest.mark.asyncio
async def test_add_note_on_access_denied_exception(bedrock_client, model, alist, messages):
    """Test that add_note adds documentation link for AccessDeniedException."""
    # Mock the client error response for access denied
    error_response = {
        "Error": {
            "Code": "AccessDeniedException",
            "Message": "An error occurred (AccessDeniedException) when calling the ConverseStream operation: "
            "You don't have access to the model with the specified model ID.",
        }
    }
    bedrock_client.converse_stream.side_effect = ClientError(error_response, "ConversationStream")

    # Call the stream method which should catch and add notes to the exception
    with pytest.raises(ClientError) as err:
        await alist(model.stream(messages))

    assert err.value.__notes__ == [
        "└ Bedrock region: us-west-2",
        "└ Model id: m1",
        "└ For more information see "
        "https://strandsagents.com/latest/user-guide/concepts/model-providers/amazon-bedrock/#model-access-issue",
    ]


@pytest.mark.skipif(sys.version_info < (3, 11), reason="This test requires Python 3.11 or higher (need add_note)")
@pytest.mark.asyncio
async def test_add_note_on_validation_exception_throughput(bedrock_client, model, alist, messages):
    """Test that add_note adds documentation link for ValidationException about on-demand throughput."""
    # Mock the client error response for validation exception
    error_response = {
        "Error": {
            "Code": "ValidationException",
            "Message": "An error occurred (ValidationException) when calling the ConverseStream operation: "
            "Invocation of model ID anthropic.claude-3-7-sonnet-20250219-v1:0 with on-demand throughput "
            "isn’t supported. Retry your request with the ID or ARN of an inference profile that contains "
            "this model.",
        }
    }
    bedrock_client.converse_stream.side_effect = ClientError(error_response, "ConversationStream")

    # Call the stream method which should catch and add notes to the exception
    with pytest.raises(ClientError) as err:
        await alist(model.stream(messages))

    assert err.value.__notes__ == [
        "└ Bedrock region: us-west-2",
        "└ Model id: m1",
        "└ For more information see "
        "https://strandsagents.com/latest/user-guide/concepts/model-providers/amazon-bedrock/#on-demand-throughput-isnt-supported",
    ]


@pytest.mark.parametrize(
    "overflow_message",
    [
        "Input is too long for requested model",
        "input length and `max_tokens` exceed context limit",
        "too many total text bytes",
        "prompt is too long: 903884 tokens > 200000 maximum",
    ],
)
@pytest.mark.asyncio
async def test_stream_context_window_overflow(overflow_message, bedrock_client, model, alist, messages):
    """Test that ClientError with overflow messages raises ContextWindowOverflowException."""
    error_response = {
        "Error": {
            "Code": "ValidationException",
            "Message": f"An error occurred (ValidationException) when calling the ConverseStream operation: "
            f"The model returned the following errors: {overflow_message}",
        }
    }
    bedrock_client.converse_stream.side_effect = ClientError(error_response, "ConverseStream")

    with pytest.raises(ContextWindowOverflowException):
        await alist(model.stream(messages))


@pytest.mark.asyncio
async def test_stream_logging(bedrock_client, model, messages, caplog, alist):
    """Test that stream method logs debug messages at the expected stages."""

    # Set the logger to debug level to capture debug messages
    caplog.set_level(logging.DEBUG, logger="strands.models.bedrock")

    # Mock the response
    bedrock_client.converse_stream.return_value = {"stream": ["e1", "e2"]}

    # Execute the stream method
    response = model.stream(messages)
    await alist(response)

    # Check that the expected log messages are present
    log_text = caplog.text
    assert "formatting request" in log_text
    assert "request=<" in log_text
    assert "invoking model" in log_text
    assert "got response from model" in log_text
    assert "finished streaming response from model" in log_text


@pytest.mark.asyncio
async def test_stream_stop_reason_override_streaming(bedrock_client, model, messages, alist):
    """Test that stopReason is overridden from end_turn to tool_use in streaming mode when tool use is detected."""
    bedrock_client.converse_stream.return_value = {
        "stream": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "123", "name": "test_tool"}}}},
            {"contentBlockDelta": {"delta": {"test": {"input": '{"param": "value"}'}}}},
            {"contentBlockStop": {}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
    }

    response = model.stream(messages)
    events = await alist(response)

    # Find the messageStop event
    message_stop_event = next(event for event in events if "messageStop" in event)

    # Verify stopReason was overridden to tool_use
    assert message_stop_event["messageStop"]["stopReason"] == "tool_use"


@pytest.mark.asyncio
async def test_stream_stop_reason_override_non_streaming(bedrock_client, alist, messages):
    """Test that stopReason is overridden from end_turn to tool_use in non-streaming mode when tool use is detected."""
    bedrock_client.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "123", "name": "test_tool", "input": {"param": "value"}}}],
            }
        },
        "stopReason": "end_turn",
    }

    model = BedrockModel(model_id="test-model", streaming=False)
    response = model.stream(messages)
    events = await alist(response)

    # Find the messageStop event
    message_stop_event = next(event for event in events if "messageStop" in event)

    # Verify stopReason was overridden to tool_use
    assert message_stop_event["messageStop"]["stopReason"] == "tool_use"


def test_format_request_cleans_tool_result_content_blocks(model, model_id):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "content": [{"text": "Tool output"}],
                        "toolUseId": "tool123",
                        "status": "success",
                        "extraField": "should be removed",
                        "mcpMetadata": {"server": "test"},
                    }
                },
            ],
        }
    ]

    formatted_request = model._format_request(messages)

    tool_result = formatted_request["messages"][0]["content"][0]["toolResult"]
    expected = {"toolUseId": "tool123", "content": [{"text": "Tool output"}]}
    assert tool_result == expected
    assert "extraField" not in tool_result
    assert "mcpMetadata" not in tool_result
    assert "status" not in tool_result


def test_format_request_removes_status_field_when_configured(model, model_id):
    model.update_config(include_tool_result_status=False)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "content": [{"text": "Tool output"}],
                        "toolUseId": "tool123",
                        "status": "success",
                    }
                },
            ],
        }
    ]

    formatted_request = model._format_request(messages)

    tool_result = formatted_request["messages"][0]["content"][0]["toolResult"]
    expected = {"toolUseId": "tool123", "content": [{"text": "Tool output"}]}
    assert tool_result == expected
    assert "status" not in tool_result


def test_auto_behavior_anthropic_vs_non_anthropic(bedrock_client):
    model_anthropic = BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0")
    assert model_anthropic.get_config()["include_tool_result_status"] == "auto"

    model_non_anthropic = BedrockModel(model_id="amazon.titan-text-v1")
    assert model_non_anthropic.get_config()["include_tool_result_status"] == "auto"


def test_explicit_boolean_values_preserved(bedrock_client):
    model = BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0", include_tool_result_status=True)
    assert model.get_config()["include_tool_result_status"] is True

    model2 = BedrockModel(model_id="amazon.titan-text-v1", include_tool_result_status=False)
    assert model2.get_config()["include_tool_result_status"] is False
    """Test that format_request keeps status field by default for anthropic.claude models."""
    # Default model is anthropic.claude, so should keep status
    model = BedrockModel()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "content": [{"text": "Tool output"}],
                        "toolUseId": "tool123",
                        "status": "success",
                    }
                },
            ],
        }
    ]

    formatted_request = model._format_request(messages)

    # Verify toolResult contains status field by default
    tool_result = formatted_request["messages"][0]["content"][0]["toolResult"]
    expected = {"content": [{"text": "Tool output"}], "toolUseId": "tool123", "status": "success"}
    assert tool_result == expected
    assert "status" in tool_result


def test_format_request_filters_sdk_unknown_member_content_blocks(model, model_id, caplog):
    """Test that format_request filters out SDK_UNKNOWN_MEMBER content blocks."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"text": "Hello"},
                {"SDK_UNKNOWN_MEMBER": {"name": "reasoningContent"}},
                {"text": "World"},
            ],
        }
    ]

    formatted_request = model._format_request(messages)

    content = formatted_request["messages"][0]["content"]
    assert len(content) == 2
    assert content[0] == {"text": "Hello"}
    assert content[1] == {"text": "World"}

    for block in content:
        assert "SDK_UNKNOWN_MEMBER" not in block


@pytest.mark.asyncio
async def test_stream_deepseek_filters_reasoning_content(bedrock_client, alist):
    """Test that DeepSeek models filter reasoningContent from messages during streaming."""
    model = BedrockModel(model_id="us.deepseek.r1-v1:0")

    messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {
            "role": "assistant",
            "content": [
                {"text": "Response"},
                {"reasoningContent": {"reasoningText": {"text": "Thinking..."}}},
            ],
        },
    ]

    bedrock_client.converse_stream.return_value = {"stream": []}

    await alist(model.stream(messages))

    # Verify the request was made with filtered messages (no reasoningContent)
    call_args = bedrock_client.converse_stream.call_args[1]
    sent_messages = call_args["messages"]

    assert len(sent_messages) == 2
    assert sent_messages[0]["content"] == [{"text": "Hello"}]
    assert sent_messages[1]["content"] == [{"text": "Response"}]


@pytest.mark.asyncio
async def test_stream_deepseek_skips_empty_messages(bedrock_client, alist):
    """Test that DeepSeek models skip messages that would be empty after filtering reasoningContent."""
    model = BedrockModel(model_id="us.deepseek.r1-v1:0")

    messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"reasoningContent": {"reasoningText": {"text": "Only reasoning..."}}}]},
        {"role": "user", "content": [{"text": "Follow up"}]},
    ]

    bedrock_client.converse_stream.return_value = {"stream": []}

    await alist(model.stream(messages))

    # Verify the request was made with only non-empty messages
    call_args = bedrock_client.converse_stream.call_args[1]
    sent_messages = call_args["messages"]

    assert len(sent_messages) == 2
    assert sent_messages[0]["content"] == [{"text": "Hello"}]
    assert sent_messages[1]["content"] == [{"text": "Follow up"}]


def test_format_request_filters_image_content_blocks(model, model_id):
    """Test that format_request filters extra fields from image content blocks."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": "png",
                        "source": {"bytes": b"image_data"},
                        "filename": "test.png",  # Extra field that should be filtered
                        "metadata": {"size": 1024},  # Extra field that should be filtered
                    }
                },
            ],
        }
    ]

    formatted_request = model._format_request(messages)

    image_block = formatted_request["messages"][0]["content"][0]["image"]
    expected = {"format": "png", "source": {"bytes": b"image_data"}}
    assert image_block == expected
    assert "filename" not in image_block
    assert "metadata" not in image_block


def test_format_request_image_s3_location_only(model, model_id):
    """Test that image with only s3Location is properly formatted."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": "png",
                        "source": {
                            "location": {"type": "s3", "uri": "s3://my-bucket/image.png"},
                        },
                    }
                }
            ],
        }
    ]

    formatted_request = model._format_request(messages)
    image_source = formatted_request["messages"][0]["content"][0]["image"]["source"]

    assert image_source == {"s3Location": {"uri": "s3://my-bucket/image.png"}}


def test_format_request_image_bytes_only(model, model_id):
    """Test that image with only bytes source is properly formatted."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": "png",
                        "source": {"bytes": b"image_data"},
                    }
                }
            ],
        }
    ]

    formatted_request = model._format_request(messages)
    image_source = formatted_request["messages"][0]["content"][0]["image"]["source"]

    assert image_source == {"bytes": b"image_data"}


def test_format_request_document_s3_location(model, model_id):
    """Test that document with s3Location is properly formatted."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "document": {
                        "name": "report.pdf",
                        "format": "pdf",
                        "source": {
                            "location": {"type": "s3", "uri": "s3://my-bucket/report.pdf"},
                        },
                    }
                },
                {
                    "document": {
                        "name": "report.pdf",
                        "format": "pdf",
                        "source": {
                            "location": {
                                "type": "s3",
                                "uri": "s3://my-bucket/report.pdf",
                                "bucketOwner": "123456789012",
                            },
                        },
                    }
                },
            ],
        }
    ]

    formatted_request = model._format_request(messages)
    document = formatted_request["messages"][0]["content"][0]["document"]
    document_with_bucket_owner = formatted_request["messages"][0]["content"][1]["document"]

    assert document["source"] == {"s3Location": {"uri": "s3://my-bucket/report.pdf"}}

    assert document_with_bucket_owner["source"] == {
        "s3Location": {"uri": "s3://my-bucket/report.pdf", "bucketOwner": "123456789012"}
    }


def test_format_request_unsupported_location(model, caplog):
    """Test that document with s3Location is properly formatted."""

    caplog.set_level(logging.WARNING, logger="strands.models.bedrock")

    messages = [
        {
            "role": "user",
            "content": [
                {"text": "Hello!"},
                {
                    "document": {
                        "name": "report.pdf",
                        "format": "pdf",
                        "source": {
                            "location": {
                                "type": "other",
                            },
                        },
                    }
                },
                {
                    "video": {
                        "format": "mp4",
                        "source": {
                            "location": {
                                "type": "other",
                            },
                        },
                    }
                },
                {
                    "image": {
                        "format": "png",
                        "source": {
                            "location": {
                                "type": "other",
                            },
                        },
                    }
                },
            ],
        }
    ]

    formatted_request = model._format_request(messages)
    assert len(formatted_request["messages"][0]["content"]) == 1
    assert "Non s3 location sources are not supported by Bedrock | skipping content block" in caplog.text


def test_format_request_video_s3_location(model, model_id):
    """Test that video with s3Location is properly formatted."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "video": {
                        "format": "mp4",
                        "source": {
                            "location": {"type": "s3", "uri": "s3://my-bucket/video.mp4"},
                        },
                    }
                },
            ],
        }
    ]

    formatted_request = model._format_request(messages)
    video_source = formatted_request["messages"][0]["content"][0]["video"]["source"]

    assert video_source == {"s3Location": {"uri": "s3://my-bucket/video.mp4"}}


def test_format_request_filters_document_content_blocks(model, model_id):
    """Test that format_request filters extra fields from document content blocks."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "document": {
                        "name": "test.pdf",
                        "source": {"bytes": b"pdf_data"},
                        "format": "pdf",
                        "extraField": "should be removed",
                        "metadata": {"pages": 10},
                    }
                },
            ],
        }
    ]

    formatted_request = model._format_request(messages)

    document_block = formatted_request["messages"][0]["content"][0]["document"]
    expected = {"name": "test.pdf", "source": {"bytes": b"pdf_data"}, "format": "pdf"}
    assert document_block == expected
    assert "extraField" not in document_block
    assert "metadata" not in document_block


def test_format_request_filters_nested_reasoning_content(model, model_id):
    """Test deep filtering of nested reasoningText fields."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "reasoningContent": {
                        "reasoningText": {"text": "thinking...", "signature": "abc123", "extraField": "filtered"}
                    }
                }
            ],
        }
    ]

    formatted_request = model._format_request(messages)
    reasoning_text = formatted_request["messages"][0]["content"][0]["reasoningContent"]["reasoningText"]

    assert reasoning_text == {"text": "thinking...", "signature": "abc123"}


def test_format_request_filters_video_content_blocks(model, model_id):
    """Test that format_request filters extra fields from video content blocks."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "video": {
                        "format": "mp4",
                        "source": {"bytes": b"video_data"},
                        "duration": 120,  # Extra field that should be filtered
                        "resolution": "1080p",  # Extra field that should be filtered
                    }
                },
            ],
        }
    ]

    formatted_request = model._format_request(messages)

    video_block = formatted_request["messages"][0]["content"][0]["video"]
    expected = {"format": "mp4", "source": {"bytes": b"video_data"}}
    assert video_block == expected
    assert "duration" not in video_block
    assert "resolution" not in video_block


def test_format_request_filters_cache_point_content_blocks(model, model_id):
    """Test that format_request filters extra fields from cachePoint content blocks."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "cachePoint": {
                        "type": "default",
                        "extraField": "should be removed",
                    }
                },
            ],
        }
    ]

    formatted_request = model._format_request(messages)

    cache_point_block = formatted_request["messages"][0]["content"][0]["cachePoint"]
    expected = {"type": "default"}
    assert cache_point_block == expected
    assert "extraField" not in cache_point_block


def test_config_validation_warns_on_unknown_keys(bedrock_client, captured_warnings):
    """Test that unknown config keys emit a warning."""
    BedrockModel(model_id="test-model", invalid_param="test")

    assert len(captured_warnings) == 1
    assert "Invalid configuration parameters" in str(captured_warnings[0].message)
    assert "invalid_param" in str(captured_warnings[0].message)


def test_update_config_validation_warns_on_unknown_keys(model, captured_warnings):
    """Test that update_config warns on unknown keys."""
    model.update_config(wrong_param="test")

    assert len(captured_warnings) == 1
    assert "Invalid configuration parameters" in str(captured_warnings[0].message)
    assert "wrong_param" in str(captured_warnings[0].message)


def test_tool_choice_supported_no_warning(model, messages, tool_spec, captured_warnings):
    """Test that toolChoice doesn't emit warning for supported providers."""
    tool_choice = {"auto": {}}
    model._format_request(messages, [tool_spec], tool_choice=tool_choice)

    assert len(captured_warnings) == 0


def test_tool_choice_none_no_warning(model, messages, captured_warnings):
    """Test that None toolChoice doesn't emit warning."""
    model._format_request(messages, tool_choice=None)

    assert len(captured_warnings) == 0


def test_get_default_model_with_warning_supported_regions_shows_no_warning(captured_warnings):
    """Test get_model_prefix_with_warning doesn't warn for supported region prefixes."""
    BedrockModel._get_default_model_with_warning("us-west-2")
    BedrockModel._get_default_model_with_warning("eu-west-2")
    assert len(captured_warnings) == 0


def test_get_default_model_for_supported_eu_region_returns_correct_model_id(captured_warnings):
    model_id = BedrockModel._get_default_model_with_warning("eu-west-1")
    assert model_id == "eu.anthropic.claude-sonnet-4-20250514-v1:0"
    assert len(captured_warnings) == 0


def test_get_default_model_for_supported_us_region_returns_correct_model_id(captured_warnings):
    model_id = BedrockModel._get_default_model_with_warning("us-east-1")
    assert model_id == "us.anthropic.claude-sonnet-4-20250514-v1:0"
    assert len(captured_warnings) == 0


def test_get_default_model_for_supported_gov_region_returns_correct_model_id(captured_warnings):
    model_id = BedrockModel._get_default_model_with_warning("us-gov-west-1")
    assert model_id == "us-gov.anthropic.claude-sonnet-4-20250514-v1:0"
    assert len(captured_warnings) == 0


def test_get_model_prefix_for_ap_region_converts_to_apac_endpoint(captured_warnings):
    """Test _get_default_model_with_warning warns for APAC regions since 'ap' is not in supported prefixes."""
    model_id = BedrockModel._get_default_model_with_warning("ap-southeast-1")
    assert model_id == "apac.anthropic.claude-sonnet-4-20250514-v1:0"


def test_get_default_model_with_warning_unsupported_region_warns(captured_warnings):
    """Test _get_default_model_with_warning warns for unsupported regions."""
    BedrockModel._get_default_model_with_warning("ca-central-1")
    assert len(captured_warnings) == 1
    assert "This region ca-central-1 does not support" in str(captured_warnings[0].message)
    assert "our default inference endpoint" in str(captured_warnings[0].message)


def test_get_default_model_with_warning_no_warning_with_custom_model_id(captured_warnings):
    """Test _get_default_model_with_warning doesn't warn when custom model_id provided."""
    model_config = {"model_id": "custom-model"}
    model_id = BedrockModel._get_default_model_with_warning("ca-central-1", model_config)

    assert model_id == "custom-model"
    assert len(captured_warnings) == 0


def test_init_with_unsupported_region_warns(session_cls, captured_warnings):
    """Test BedrockModel initialization warns for unsupported regions."""
    BedrockModel(region_name="ca-central-1")

    assert len(captured_warnings) == 1
    assert "This region ca-central-1 does not support" in str(captured_warnings[0].message)


def test_init_with_unsupported_region_custom_model_no_warning(session_cls, captured_warnings):
    """Test BedrockModel initialization doesn't warn when custom model_id provided."""
    BedrockModel(region_name="ca-central-1", model_id="custom-model")
    assert len(captured_warnings) == 0


def test_override_default_model_id_uses_the_overriden_value(captured_warnings):
    with unittest.mock.patch("strands.models.bedrock.DEFAULT_BEDROCK_MODEL_ID", "custom-overridden-model"):
        model_id = BedrockModel._get_default_model_with_warning("us-east-1")
        assert model_id == "custom-overridden-model"


def test_no_override_uses_formatted_default_model_id(captured_warnings):
    model_id = BedrockModel._get_default_model_with_warning("us-east-1")
    assert model_id == "us.anthropic.claude-sonnet-4-20250514-v1:0"
    assert model_id != _DEFAULT_BEDROCK_MODEL_ID
    assert len(captured_warnings) == 0


def test_custom_model_id_not_overridden_by_region_formatting(session_cls):
    """Test that custom model_id is not overridden by region formatting."""
    custom_model_id = "custom.model.id"

    model = BedrockModel(model_id=custom_model_id)
    model_id = model.get_config().get("model_id")

    assert model_id == custom_model_id


def test_format_request_filters_output_schema(model, messages, model_id):
    """Test that outputSchema is filtered out from tool specs in Bedrock requests."""
    tool_spec_with_output_schema = {
        "description": "Test tool with output schema",
        "name": "test_tool",
        "inputSchema": {"type": "object", "properties": {}},
        "outputSchema": {"type": "object", "properties": {"result": {"type": "string"}}},
    }

    request = model._format_request(messages, [tool_spec_with_output_schema])

    tool_spec = request["toolConfig"]["tools"][0]["toolSpec"]

    # Verify outputSchema is not included
    assert "outputSchema" not in tool_spec

    # Verify other fields are preserved
    assert tool_spec["name"] == "test_tool"
    assert tool_spec["description"] == "Test tool with output schema"
    assert tool_spec["inputSchema"] == {"type": "object", "properties": {}}


@pytest.mark.asyncio
async def test_stream_backward_compatibility_system_prompt(bedrock_client, model, messages, alist):
    """Test that system_prompt is converted to system_prompt_content when system_prompt_content is None."""
    bedrock_client.converse_stream.return_value = {"stream": ["e1", "e2"]}

    system_prompt = "You are a helpful assistant."

    response = model.stream(messages, system_prompt=system_prompt)
    await alist(response)

    # Verify the request was formatted with system_prompt converted to system_prompt_content
    expected_request = {
        "inferenceConfig": {},
        "modelId": "m1",
        "messages": messages,
        "system": [{"text": system_prompt}],
    }
    bedrock_client.converse_stream.assert_called_once_with(**expected_request)


@pytest.mark.asyncio
async def test_citations_content_preserves_tagged_union_structure(bedrock_client, model, alist):
    """Test that citationsContent preserves AWS Bedrock's required tagged union structure for citation locations.

    This test verifies that when messages contain citationsContent with tagged union CitationLocation objects,
    the structure is preserved when sent to AWS Bedrock API. AWS Bedrock expects CitationLocation to be a
    tagged union with exactly one wrapper key (documentChar, documentPage, documentChunk, searchResultLocation, web)
    containing the location fields.
    """
    # Mock the Bedrock response
    bedrock_client.converse_stream.return_value = {"stream": []}

    # Messages with citationsContent using all tagged union CitationLocation types
    messages = [
        {"role": "user", "content": [{"text": "Analyze multiple sources"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "citationsContent": {
                        "citations": [
                            {
                                "location": {"documentChar": {"documentIndex": 0, "start": 150, "end": 300}},
                                "sourceContent": [
                                    {"text": "Employee benefits include health insurance and retirement plans"}
                                ],
                                "title": "Benefits Section",
                            },
                            {
                                "location": {"documentPage": {"documentIndex": 0, "start": 2, "end": 3}},
                                "sourceContent": [{"text": "Vacation policy allows 15 days per year"}],
                                "title": "Vacation Policy",
                            },
                            {
                                "location": {"documentChunk": {"documentIndex": 1, "start": 5, "end": 8}},
                                "sourceContent": [{"text": "Company culture emphasizes work-life balance"}],
                                "title": "Culture Section",
                            },
                            {
                                "location": {
                                    "searchResultLocation": {
                                        "searchResultIndex": 0,
                                        "start": 25,
                                        "end": 150,
                                    }
                                },
                                "sourceContent": [{"text": "Search results show industry best practices"}],
                                "title": "Search Results",
                            },
                            {
                                "location": {
                                    "web": {
                                        "url": "https://example.com/hr-policies",
                                        "domain": "example.com",
                                    }
                                },
                                "sourceContent": [{"text": "External HR policy guidelines"}],
                                "title": "External Reference",
                            },
                        ],
                        "content": [{"text": "Based on multiple sources, the company offers comprehensive benefits."}],
                    }
                }
            ],
        },
    ]

    # Call the public stream method
    await alist(model.stream(messages))

    # Verify the request sent to Bedrock preserves the tagged union structure
    bedrock_client.converse_stream.assert_called_once()
    call_args = bedrock_client.converse_stream.call_args[1]

    # Extract the citationsContent from the formatted messages
    formatted_messages = call_args["messages"]
    citations_content = formatted_messages[1]["content"][0]["citationsContent"]

    # Verify the tagged union structure is preserved for all location types
    expected_citations = [
        {
            "location": {"documentChar": {"documentIndex": 0, "start": 150, "end": 300}},
            "sourceContent": [{"text": "Employee benefits include health insurance and retirement plans"}],
            "title": "Benefits Section",
        },
        {
            "location": {"documentPage": {"documentIndex": 0, "start": 2, "end": 3}},
            "sourceContent": [{"text": "Vacation policy allows 15 days per year"}],
            "title": "Vacation Policy",
        },
        {
            "location": {"documentChunk": {"documentIndex": 1, "start": 5, "end": 8}},
            "sourceContent": [{"text": "Company culture emphasizes work-life balance"}],
            "title": "Culture Section",
        },
        {
            "location": {
                "searchResultLocation": {
                    "searchResultIndex": 0,
                    "start": 25,
                    "end": 150,
                }
            },
            "sourceContent": [{"text": "Search results show industry best practices"}],
            "title": "Search Results",
        },
        {
            "location": {
                "web": {
                    "url": "https://example.com/hr-policies",
                    "domain": "example.com",
                }
            },
            "sourceContent": [{"text": "External HR policy guidelines"}],
            "title": "External Reference",
        },
    ]

    assert citations_content["citations"] == expected_citations, (
        "Citation location tagged union structure was not preserved. "
        "AWS Bedrock requires CitationLocation to have exactly one wrapper key "
        "(documentChar, documentPage, documentChunk, searchResultLocation, or web) "
        "with the location fields nested inside."
    )


@pytest.mark.asyncio
async def test_format_request_with_guardrail_latest_message(model):
    """Test that guardrail_latest_message wraps the latest user message with text and image."""
    model.update_config(
        guardrail_id="test-guardrail",
        guardrail_version="DRAFT",
        guardrail_latest_message=True,
    )

    messages = [
        {"role": "user", "content": [{"text": "First message"}]},
        {"role": "assistant", "content": [{"text": "First response"}]},
        {
            "role": "user",
            "content": [
                {"text": "Look at this image"},
                {"image": {"format": "png", "source": {"bytes": b"fake_image_data"}}},
            ],
        },
    ]

    request = model._format_request(messages)
    formatted_messages = request["messages"]

    # All messages should be in the request
    assert len(formatted_messages) == 3

    # First user message should NOT be wrapped
    assert "text" in formatted_messages[0]["content"][0]
    assert formatted_messages[0]["content"][0]["text"] == "First message"

    # Assistant message should NOT be wrapped
    assert "text" in formatted_messages[1]["content"][0]
    assert formatted_messages[1]["content"][0]["text"] == "First response"

    # Latest user message text should be wrapped
    assert "guardContent" in formatted_messages[2]["content"][0]
    assert formatted_messages[2]["content"][0]["guardContent"]["text"]["text"] == "Look at this image"

    # Latest user message image should also be wrapped
    assert "guardContent" in formatted_messages[2]["content"][1]
    assert formatted_messages[2]["content"][1]["guardContent"]["image"]["format"] == "png"


@pytest.mark.asyncio
async def test_format_request_with_guardrail_latest_message_after_tool_use(model):
    """Test that guardContent wraps the last user text message even when a toolResult follows it."""
    model.update_config(
        guardrail_id="test-guardrail",
        guardrail_version="DRAFT",
        guardrail_latest_message=True,
    )

    messages = [
        {"role": "user", "content": [{"text": "First message"}]},
        {"role": "assistant", "content": [{"text": "First response"}]},
        {"role": "user", "content": [{"text": "what is the standard deduction?"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "tool-1",
                        "name": "knowledge_base",
                        "input": {"query": "standard deduction"},
                    }
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "tool-1",
                        "content": [{"text": "The standard deduction for 2024 is $14,600."}],
                        "status": "success",
                    }
                }
            ],
        },
    ]

    request = model._format_request(messages)
    formatted_messages = request["messages"]

    assert len(formatted_messages) == 5

    # Earlier user message should NOT be wrapped
    assert "text" in formatted_messages[0]["content"][0]
    assert formatted_messages[0]["content"][0]["text"] == "First message"

    # Last user message with text content should be wrapped, even though a toolResult comes after
    assert "guardContent" in formatted_messages[2]["content"][0]
    assert formatted_messages[2]["content"][0]["guardContent"]["text"]["text"] == "what is the standard deduction?"

    # toolResult-only user message should NOT be wrapped
    assert "toolResult" in formatted_messages[4]["content"][0]
    assert "guardContent" not in formatted_messages[4]["content"][0]


@pytest.mark.asyncio
async def test_format_request_with_guardrail_latest_message_wraps_final_user_text(model):
    """Test that guardContent wraps the last user message when it contains text content."""
    model.update_config(
        guardrail_id="test-guardrail",
        guardrail_version="DRAFT",
        guardrail_latest_message=True,
    )

    messages = [
        {"role": "user", "content": [{"text": "First message"}]},
        {"role": "assistant", "content": [{"text": "First response"}]},
        {"role": "user", "content": [{"text": "Tell me about taxes"}]},
    ]

    request = model._format_request(messages)
    formatted_messages = request["messages"]

    assert "guardContent" in formatted_messages[2]["content"][0]
    assert formatted_messages[2]["content"][0]["guardContent"]["text"]["text"] == "Tell me about taxes"


@pytest.mark.asyncio
async def test_format_request_with_guardrail_multiple_sequential_tool_calls(model):
    """Test guardContent with multiple tool calls in sequence (no new user input between)."""
    model.update_config(
        guardrail_id="test-guardrail",
        guardrail_version="DRAFT",
        guardrail_latest_message=True,
    )

    messages = [
        {"role": "user", "content": [{"text": "First question"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "tool1", "input": {}}}]},
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "Result 1"}], "status": "success"}}],
        },
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t2", "name": "tool2", "input": {}}}]},
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "t2", "content": [{"text": "Result 2"}], "status": "success"}}],
        },
    ]

    request = model._format_request(messages)
    formatted_messages = request["messages"]

    # Should wrap the first user text message, not the toolResults
    assert "guardContent" in formatted_messages[0]["content"][0]
    assert formatted_messages[0]["content"][0]["guardContent"]["text"]["text"] == "First question"

    # toolResults should not be wrapped
    assert "toolResult" in formatted_messages[2]["content"][0]
    assert "guardContent" not in formatted_messages[2]["content"][0]
    assert "toolResult" in formatted_messages[4]["content"][0]
    assert "guardContent" not in formatted_messages[4]["content"][0]


@pytest.mark.asyncio
async def test_format_request_with_guardrail_image_before_tool_result(model):
    """Test guardContent wraps image content even when toolResult follows."""
    model.update_config(
        guardrail_id="test-guardrail",
        guardrail_version="DRAFT",
        guardrail_latest_message=True,
    )

    messages = [
        {"role": "user", "content": [{"image": {"format": "png", "source": {"bytes": b"fake"}}}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "vision", "input": {}}}]},
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "I see a cat"}], "status": "success"}}],
        },
    ]

    request = model._format_request(messages)
    formatted_messages = request["messages"]

    # Image should be wrapped even though toolResult comes after
    assert "guardContent" in formatted_messages[0]["content"][0]
    assert "image" in formatted_messages[0]["content"][0]["guardContent"]


@pytest.mark.asyncio
async def test_format_request_with_guardrail_multiple_tool_results_same_message(model):
    """Test guardContent with multiple parallel tool calls (multiple toolResults in one message)."""
    model.update_config(
        guardrail_id="test-guardrail",
        guardrail_version="DRAFT",
        guardrail_latest_message=True,
    )

    messages = [
        {"role": "user", "content": [{"text": "Question requiring multiple tools"}]},
        {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "t1", "name": "tool1", "input": {}}},
                {"toolUse": {"toolUseId": "t2", "name": "tool2", "input": {}}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "t1", "content": [{"text": "Result 1"}], "status": "success"}},
                {"toolResult": {"toolUseId": "t2", "content": [{"text": "Result 2"}], "status": "success"}},
            ],
        },
    ]

    request = model._format_request(messages)
    formatted_messages = request["messages"]

    # Should wrap the question
    assert "guardContent" in formatted_messages[0]["content"][0]
    assert formatted_messages[0]["content"][0]["guardContent"]["text"]["text"] == "Question requiring multiple tools"


def test_supports_caching_true_for_claude(bedrock_client):
    """Test that supports_caching returns True for Claude models."""
    model = BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0")
    assert model._supports_caching is True

    model2 = BedrockModel(model_id="anthropic.claude-3-haiku-20240307-v1:0")
    assert model2._supports_caching is True


def test_supports_caching_false_for_non_claude(bedrock_client):
    """Test that supports_caching returns False for non-Claude models."""
    model = BedrockModel(model_id="amazon.nova-pro-v1:0")
    assert model._supports_caching is False


def test_supports_caching_true_for_application_inference_profile_arn(session_cls):
    """Test that _supports_caching resolves application inference profile ARNs to the underlying model."""
    mock_runtime_client = session_cls.return_value.client.return_value
    mock_runtime_client.meta = unittest.mock.MagicMock()
    mock_runtime_client.meta.region_name = "us-east-1"

    # The bedrock control-plane client (same mock due to session_cls fixture)
    mock_bedrock_client = mock_runtime_client
    mock_bedrock_client.get_inference_profile.return_value = {
        "models": [
            {"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0"}
        ]
    }

    arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123"
    model = BedrockModel(model_id=arn)
    assert model._supports_caching is True


def test_supports_caching_false_for_non_claude_application_inference_profile_arn(session_cls):
    """Test that _supports_caching returns False for ARNs that resolve to non-Claude models."""
    mock_runtime_client = session_cls.return_value.client.return_value
    mock_runtime_client.meta = unittest.mock.MagicMock()
    mock_runtime_client.meta.region_name = "us-east-1"

    mock_bedrock_client = mock_runtime_client
    mock_bedrock_client.get_inference_profile.return_value = {
        "models": [{"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0"}]
    }

    arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/def456"
    model = BedrockModel(model_id=arn)
    assert model._supports_caching is False


def test_supports_caching_false_when_arn_resolution_fails(session_cls):
    """Test that _supports_caching returns False when the inference profile API call fails."""
    mock_runtime_client = session_cls.return_value.client.return_value
    mock_runtime_client.meta = unittest.mock.MagicMock()
    mock_runtime_client.meta.region_name = "us-east-1"

    mock_bedrock_client = mock_runtime_client
    mock_bedrock_client.get_inference_profile.side_effect = ClientError(
        {"Error": {"Code": "ResourceNotFoundException", "Message": "Not found"}},
        "GetInferenceProfile",
    )

    arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/missing"
    model = BedrockModel(model_id=arn)
    assert model._supports_caching is False


def test_supports_caching_result_is_cached(session_cls):
    """Test that the caching support result is cached after first resolution."""
    mock_runtime_client = session_cls.return_value.client.return_value
    mock_runtime_client.meta = unittest.mock.MagicMock()
    mock_runtime_client.meta.region_name = "us-east-1"

    mock_bedrock_client = mock_runtime_client
    mock_bedrock_client.get_inference_profile.return_value = {
        "models": [
            {"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0"}
        ]
    }

    arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/cached"
    model = BedrockModel(model_id=arn)

    # First call resolves via API
    assert model._supports_caching is True
    # Second call should use cached value — reset mock to verify no extra calls
    mock_bedrock_client.get_inference_profile.reset_mock()
    assert model._supports_caching is True
    mock_bedrock_client.get_inference_profile.assert_not_called()


def test_supports_caching_cache_invalidated_on_model_id_change(session_cls):
    """Test that _caching_supported cache is invalidated when model_id changes via update_config."""
    mock_runtime_client = session_cls.return_value.client.return_value
    mock_runtime_client.meta = unittest.mock.MagicMock()
    mock_runtime_client.meta.region_name = "us-east-1"

    model = BedrockModel(model_id="amazon.nova-pro-v1:0")
    assert model._supports_caching is False

    model.update_config(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0")
    assert model._supports_caching is True


def test_supports_caching_arn_uses_original_case(session_cls):
    """Test that the original ARN casing is preserved when calling the API."""
    mock_runtime_client = session_cls.return_value.client.return_value
    mock_runtime_client.meta = unittest.mock.MagicMock()
    mock_runtime_client.meta.region_name = "us-east-1"

    mock_bedrock_client = mock_runtime_client
    mock_bedrock_client.get_inference_profile.return_value = {
        "models": [
            {"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0"}
        ]
    }

    # ARN with mixed case in the profile ID
    arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/AbCdEf123"
    model = BedrockModel(model_id=arn)
    assert model._supports_caching is True
    # Verify the original ARN (not lowercased) was passed to the API
    mock_bedrock_client.get_inference_profile.assert_called_once_with(inferenceProfileIdentifier=arn)


def test_supports_caching_empty_models_list(session_cls):
    """Test that _supports_caching returns False when the API returns an empty models list."""
    mock_runtime_client = session_cls.return_value.client.return_value
    mock_runtime_client.meta = unittest.mock.MagicMock()
    mock_runtime_client.meta.region_name = "us-east-1"

    mock_bedrock_client = mock_runtime_client
    mock_bedrock_client.get_inference_profile.return_value = {"models": []}

    arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/empty"
    model = BedrockModel(model_id=arn)
    assert model._supports_caching is False


def test_inject_cache_point_adds_to_last_assistant(bedrock_client):
    """Test that _inject_cache_point adds cache point to last assistant message."""
    model = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-20250514-v1:0", cache_config=CacheConfig(strategy="auto")
    )

    cleaned_messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi there!"}]},
        {"role": "user", "content": [{"text": "How are you?"}]},
    ]

    model._inject_cache_point(cleaned_messages)

    assert len(cleaned_messages[1]["content"]) == 2
    assert "cachePoint" in cleaned_messages[1]["content"][-1]
    assert cleaned_messages[1]["content"][-1]["cachePoint"]["type"] == "default"


def test_inject_cache_point_no_assistant_message(bedrock_client):
    """Test that _inject_cache_point does nothing when no assistant message exists."""
    model = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-20250514-v1:0", cache_config=CacheConfig(strategy="auto")
    )

    cleaned_messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
    ]

    model._inject_cache_point(cleaned_messages)

    assert len(cleaned_messages) == 1
    assert len(cleaned_messages[0]["content"]) == 1


def test_inject_cache_point_skipped_for_non_claude(bedrock_client):
    """Test that cache point injection is skipped for non-Claude models."""
    model = BedrockModel(model_id="amazon.nova-pro-v1:0", cache_config=CacheConfig(strategy="auto"))

    messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Response"}]},
    ]

    formatted = model._format_bedrock_messages(messages)

    assert len(formatted[1]["content"]) == 1
    assert "cachePoint" not in formatted[1]["content"][0]


def test_format_bedrock_messages_does_not_mutate_original(bedrock_client):
    """Test that _format_bedrock_messages does not mutate original messages."""

    model = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-20250514-v1:0", cache_config=CacheConfig(strategy="auto")
    )

    original_messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi there!"}]},
        {"role": "user", "content": [{"text": "How are you?"}]},
    ]

    messages_before = copy.deepcopy(original_messages)
    formatted = model._format_bedrock_messages(original_messages)

    assert original_messages == messages_before
    assert "cachePoint" not in original_messages[1]["content"][-1]
    assert "cachePoint" in formatted[1]["content"][-1]


def test_inject_cache_point_strips_existing_cache_points(bedrock_client):
    """Test that _inject_cache_point strips existing cache points and adds new one at correct position."""
    model = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-20250514-v1:0", cache_config=CacheConfig(strategy="auto")
    )

    # Messages with existing cache points in various positions
    cleaned_messages = [
        {"role": "user", "content": [{"text": "Hello"}, {"cachePoint": {"type": "default"}}]},
        {"role": "assistant", "content": [{"text": "First response"}, {"cachePoint": {"type": "default"}}]},
        {"role": "user", "content": [{"text": "Follow up"}]},
        {"role": "assistant", "content": [{"text": "Second response"}]},
    ]

    model._inject_cache_point(cleaned_messages)

    # All old cache points should be stripped
    assert len(cleaned_messages[0]["content"]) == 1  # user: only text
    assert len(cleaned_messages[1]["content"]) == 1  # first assistant: only text

    # New cache point should be at end of last assistant message
    assert len(cleaned_messages[3]["content"]) == 2
    assert "cachePoint" in cleaned_messages[3]["content"][-1]


def test_find_last_user_text_message_index_no_user_messages(bedrock_client):
    """Test _find_last_user_text_message_index returns None when no user text messages exist."""
    model = BedrockModel(model_id="test-model")

    messages = [
        {"role": "assistant", "content": [{"text": "hello"}]},
    ]

    assert model._find_last_user_text_message_index(messages) is None


def test_find_last_user_text_message_index_only_tool_results(bedrock_client):
    """Test _find_last_user_text_message_index returns None when user messages only have toolResult."""
    model = BedrockModel(model_id="test-model")

    messages = [
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "result"}]}}],
        },
    ]

    assert model._find_last_user_text_message_index(messages) is None


def test_find_last_user_text_message_index_returns_last_text_message(bedrock_client):
    """Test _find_last_user_text_message_index returns the index of the last user message with text."""
    model = BedrockModel(model_id="test-model")

    messages = [
        {"role": "user", "content": [{"text": "First question"}]},
        {"role": "assistant", "content": [{"text": "Response"}]},
        {"role": "user", "content": [{"text": "Second question"}]},
    ]

    assert model._find_last_user_text_message_index(messages) == 2


def test_find_last_user_text_message_index_skips_tool_result_messages(bedrock_client):
    """Test _find_last_user_text_message_index skips toolResult-only user messages."""
    model = BedrockModel(model_id="test-model")

    messages = [
        {"role": "user", "content": [{"text": "Question"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "tool", "input": {}}}]},
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "Result"}]}}],
        },
    ]

    assert model._find_last_user_text_message_index(messages) == 0


def test_find_last_user_text_message_index_finds_image_message(bedrock_client):
    """Test _find_last_user_text_message_index finds user messages with image content."""
    model = BedrockModel(model_id="test-model")

    messages = [
        {"role": "user", "content": [{"image": {"format": "png", "source": {"bytes": b"fake"}}}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "vision", "input": {}}}]},
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "Result"}]}}],
        },
    ]

    assert model._find_last_user_text_message_index(messages) == 0


def test_find_last_user_text_message_index_empty_messages(bedrock_client):
    """Test _find_last_user_text_message_index returns None for empty message list."""
    model = BedrockModel(model_id="test-model")

    assert model._find_last_user_text_message_index([]) is None


def test_guardrail_latest_message_disabled_does_not_wrap(model):
    """Test that guardContent wrapping is skipped when guardrail_latest_message is not set."""
    messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
    ]

    request = model._format_request(messages)
    formatted = request["messages"][0]["content"][0]

    assert "text" in formatted
    assert "guardContent" not in formatted
