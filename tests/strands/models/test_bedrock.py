import os
import sys
import unittest.mock
from unittest.mock import ANY

import boto3
import pydantic
import pytest
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError, EventStreamError

import strands
from strands.models import BedrockModel
from strands.models.bedrock import (
    _DEFAULT_BEDROCK_MODEL_ID,
    DEFAULT_BEDROCK_MODEL_ID,
    DEFAULT_BEDROCK_REGION,
    DEFAULT_READ_TIMEOUT,
)
from strands.types.exceptions import ModelThrottledException
from strands.types.tools import ToolSpec

FORMATTED_DEFAULT_MODEL_ID = DEFAULT_BEDROCK_MODEL_ID.format("us")


@pytest.fixture
def session_cls():
    # Mock the creation of a Session so that we don't depend on environment variables or profiles
    with unittest.mock.patch.object(strands.models.bedrock.boto3, "Session") as mock_session_cls:
        mock_session_cls.return_value.region_name = None
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
    BedrockModel(endpoint_url=custom_endpoint)
    mock_client_method.assert_called_with(
        region_name=DEFAULT_BEDROCK_REGION, config=ANY, service_name=ANY, endpoint_url=custom_endpoint
    )


def test__init__with_region_and_session_raises_value_error():
    """Test that BedrockModel raises ValueError when both region and session are provided."""
    with pytest.raises(ValueError):
        _ = BedrockModel(region_name="us-east-1", boto_session=boto3.Session(region_name="us-east-1"))


def test__init__default_user_agent(bedrock_client):
    """Set user agent when no boto_client_config is provided."""
    with unittest.mock.patch("strands.models.bedrock.boto3.Session") as mock_session_cls:
        mock_session = mock_session_cls.return_value
        _ = BedrockModel()

        # Verify the client was created with the correct config
        mock_session.client.assert_called_once()
        args, kwargs = mock_session.client.call_args
        assert kwargs["service_name"] == "bedrock-runtime"
        assert isinstance(kwargs["config"], BotocoreConfig)
        assert kwargs["config"].user_agent_extra == "strands-agents"
        assert kwargs["config"].read_timeout == DEFAULT_READ_TIMEOUT


def test__init__default_read_timeout(bedrock_client):
    """Set default read timeout when no boto_client_config is provided."""
    with unittest.mock.patch("strands.models.bedrock.boto3.Session") as mock_session_cls:
        mock_session = mock_session_cls.return_value
        _ = BedrockModel()

        # Verify the client was created with the correct read timeout
        mock_session.client.assert_called_once()
        args, kwargs = mock_session.client.call_args
        assert isinstance(kwargs["config"], BotocoreConfig)
        assert kwargs["config"].read_timeout == DEFAULT_READ_TIMEOUT


def test__init__with_custom_boto_client_config_no_user_agent(bedrock_client):
    """Set user agent when boto_client_config is provided without user_agent_extra."""
    custom_config = BotocoreConfig(read_timeout=900)

    with unittest.mock.patch("strands.models.bedrock.boto3.Session") as mock_session_cls:
        mock_session = mock_session_cls.return_value
        _ = BedrockModel(boto_client_config=custom_config)

        # Verify the client was created with the correct config
        mock_session.client.assert_called_once()
        args, kwargs = mock_session.client.call_args
        assert kwargs["service_name"] == "bedrock-runtime"
        assert isinstance(kwargs["config"], BotocoreConfig)
        assert kwargs["config"].user_agent_extra == "strands-agents"
        assert kwargs["config"].read_timeout == 900


def test__init__with_custom_boto_client_config_with_user_agent(bedrock_client):
    """Append to existing user agent when boto_client_config is provided with user_agent_extra."""
    custom_config = BotocoreConfig(user_agent_extra="existing-agent", read_timeout=900)

    with unittest.mock.patch("strands.models.bedrock.boto3.Session") as mock_session_cls:
        mock_session = mock_session_cls.return_value
        _ = BedrockModel(boto_client_config=custom_config)

        # Verify the client was created with the correct config
        mock_session.client.assert_called_once()
        args, kwargs = mock_session.client.call_args
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
    tru_request = model.format_request(messages)
    exp_request = {
        "inferenceConfig": {},
        "modelId": model_id,
        "messages": messages,
        "system": [],
    }

    assert tru_request == exp_request


def test_format_request_additional_request_fields(model, messages, model_id, additional_request_fields):
    model.update_config(additional_request_fields=additional_request_fields)
    tru_request = model.format_request(messages)
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
    tru_request = model.format_request(messages)
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
    tru_request = model.format_request(messages)
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
    tru_request = model.format_request(messages)
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
    tru_request = model.format_request(messages)
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
    tru_request = model.format_request(messages, system_prompt=system_prompt)
    exp_request = {
        "inferenceConfig": {},
        "modelId": model_id,
        "messages": messages,
        "system": [{"text": system_prompt}],
    }

    assert tru_request == exp_request


def test_format_request_tool_specs(model, messages, model_id, tool_spec):
    tru_request = model.format_request(messages, [tool_spec])
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
    tru_request = model.format_request(messages, [tool_spec], tool_choice=tool_choice)
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
    tru_request = model.format_request(messages, [tool_spec], tool_choice=tool_choice)
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
    tru_request = model.format_request(messages, [tool_spec], tool_choice=tool_choice)
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
    tru_request = model.format_request(messages, [tool_spec])
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


@pytest.mark.asyncio
async def test_stream_logging(bedrock_client, model, messages, caplog, alist):
    """Test that stream method logs debug messages at the expected stages."""
    import logging

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

    formatted_request = model.format_request(messages)

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

    formatted_request = model.format_request(messages)

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

    formatted_request = model.format_request(messages)

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

    formatted_request = model.format_request(messages)

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
    model.format_request(messages, [tool_spec], tool_choice=tool_choice)

    assert len(captured_warnings) == 0


def test_tool_choice_none_no_warning(model, messages, captured_warnings):
    """Test that None toolChoice doesn't emit warning."""
    model.format_request(messages, tool_choice=None)

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
