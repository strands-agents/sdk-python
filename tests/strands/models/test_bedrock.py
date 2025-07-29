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
from strands.models.bedrock import DEFAULT_BEDROCK_MODEL_ID, DEFAULT_BEDROCK_REGION
from strands.types.exceptions import ModelThrottledException
from strands.types.tools import ToolSpec


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
    exp_model_id = DEFAULT_BEDROCK_MODEL_ID

    assert tru_model_id == exp_model_id


def test__init__with_default_region(session_cls, mock_client_method):
    """Test that BedrockModel uses the provided region."""
    with unittest.mock.patch.object(os, "environ", {}):
        BedrockModel()
        session_cls.return_value.client.assert_called_with(
            region_name=DEFAULT_BEDROCK_REGION, config=ANY, service_name=ANY
        )


def test__init__with_session_region(session_cls, mock_client_method):
    """Test that BedrockModel uses the provided region."""
    session_cls.return_value.region_name = "eu-blah-1"

    BedrockModel()

    mock_client_method.assert_called_with(region_name="eu-blah-1", config=ANY, service_name=ANY)


def test__init__with_custom_region(mock_client_method):
    """Test that BedrockModel uses the provided region."""
    custom_region = "us-east-1"
    BedrockModel(region_name=custom_region)
    mock_client_method.assert_called_with(region_name=custom_region, config=ANY, service_name=ANY)


def test__init__with_default_environment_variable_region(mock_client_method):
    """Test that BedrockModel uses the AWS_REGION since we code that in."""
    with unittest.mock.patch.object(os, "environ", {"AWS_REGION": "eu-west-2"}):
        BedrockModel()

    mock_client_method.assert_called_with(region_name="eu-west-2", config=ANY, service_name=ANY)


def test__init__region_precedence(mock_client_method, session_cls):
    """Test that BedrockModel uses the correct ordering of precedence when determining region."""
    with unittest.mock.patch.object(os, "environ", {"AWS_REGION": "us-environment-1"}) as mock_os_environ:
        session_cls.return_value.region_name = "us-session-1"

        # specifying a region always wins out
        BedrockModel(region_name="us-specified-1")
        mock_client_method.assert_called_with(region_name="us-specified-1", config=ANY, service_name=ANY)

        # other-wise uses the session's
        BedrockModel()
        mock_client_method.assert_called_with(region_name="us-session-1", config=ANY, service_name=ANY)

        # environment variable next
        session_cls.return_value.region_name = None
        BedrockModel()
        mock_client_method.assert_called_with(region_name="us-environment-1", config=ANY, service_name=ANY)

        mock_os_environ.pop("AWS_REGION")
        session_cls.return_value.region_name = None  # No session region
        BedrockModel()
        mock_client_method.assert_called_with(region_name=DEFAULT_BEDROCK_REGION, config=ANY, service_name=ANY)


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


def test_format_request_cleans_tool_result_content_blocks(model, model_id):
    """Test that format_request cleans toolResult blocks by removing extra fields."""
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

    # Verify toolResult only contains allowed fields in the formatted request
    tool_result = formatted_request["messages"][0]["content"][0]["toolResult"]
    expected = {"content": [{"text": "Tool output"}], "toolUseId": "tool123", "status": "success"}
    assert tool_result == expected
    assert "extraField" not in tool_result
    assert "mcpMetadata" not in tool_result
