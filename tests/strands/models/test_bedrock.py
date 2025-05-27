import os
import unittest.mock

import boto3
import pytest
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError, EventStreamError

import strands
from strands.models import BedrockModel
from strands.models.bedrock import DEFAULT_BEDROCK_MODEL_ID
from strands.types.exceptions import ModelThrottledException


@pytest.fixture
def bedrock_client():
    with unittest.mock.patch.object(strands.models.bedrock.boto3, "Session") as mock_session_cls:
        yield mock_session_cls.return_value.client.return_value


@pytest.fixture
def model_id():
    return "m1"


@pytest.fixture
def model(bedrock_client, model_id):
    _ = bedrock_client

    return BedrockModel(model_id=model_id)


@pytest.fixture
def messages():
    return [{"role": "user", "content": {"text": "test"}}]


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
def tool_spec():
    return {"t1": 1}


@pytest.fixture
def cache_type():
    return "default"


def test__init__default_model_id(bedrock_client):
    """Test that BedrockModel uses DEFAULT_MODEL_ID when no model_id is provided."""
    _ = bedrock_client
    model = BedrockModel()

    tru_model_id = model.get_config().get("model_id")
    exp_model_id = DEFAULT_BEDROCK_MODEL_ID

    assert tru_model_id == exp_model_id


def test__init__with_custom_region(bedrock_client):
    """Test that BedrockModel uses the provided region."""
    _ = bedrock_client
    custom_region = "us-east-1"

    with unittest.mock.patch("strands.models.bedrock.boto3.Session") as mock_session_cls:
        _ = BedrockModel(region_name=custom_region)
        mock_session_cls.assert_called_once_with(region_name=custom_region)


def test__init__with_environment_variable_region(bedrock_client):
    """Test that BedrockModel uses the provided region."""
    _ = bedrock_client
    os.environ["AWS_REGION"] = "eu-west-1"

    with unittest.mock.patch("strands.models.bedrock.boto3.Session") as mock_session_cls:
        _ = BedrockModel()
        mock_session_cls.assert_called_once_with(region_name="eu-west-1")


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


def test_format_chunk(model):
    tru_chunk = model.format_chunk("event")
    exp_chunk = "event"

    assert tru_chunk == exp_chunk


def test_stream(bedrock_client, model):
    bedrock_client.converse_stream.return_value = {"stream": ["e1", "e2"]}

    request = {"a": 1}
    response = model.stream(request)

    tru_events = list(response)
    exp_events = ["e1", "e2"]

    assert tru_events == exp_events
    bedrock_client.converse_stream.assert_called_once_with(a=1)


def test_stream_throttling_exception_from_event_stream_error(bedrock_client, model):
    error_message = "Rate exceeded"
    bedrock_client.converse_stream.side_effect = EventStreamError(
        {"Error": {"Message": error_message, "Code": "ThrottlingException"}}, "ConverseStream"
    )

    request = {"a": 1}

    with pytest.raises(ModelThrottledException) as excinfo:
        list(model.stream(request))

    assert error_message in str(excinfo.value)
    bedrock_client.converse_stream.assert_called_once_with(a=1)


def test_stream_throttling_exception_from_general_exception(bedrock_client, model):
    error_message = "ThrottlingException: Rate exceeded for ConverseStream"
    bedrock_client.converse_stream.side_effect = ClientError(
        {"Error": {"Message": error_message, "Code": "ThrottlingException"}}, "Any"
    )

    request = {"a": 1}

    with pytest.raises(ModelThrottledException) as excinfo:
        list(model.stream(request))

    assert error_message in str(excinfo.value)
    bedrock_client.converse_stream.assert_called_once_with(a=1)


def test_general_exception_is_raised(bedrock_client, model):
    error_message = "Should be raised up"
    bedrock_client.converse_stream.side_effect = ValueError(error_message)

    request = {"a": 1}

    with pytest.raises(ValueError) as excinfo:
        list(model.stream(request))

    assert error_message in str(excinfo.value)
    bedrock_client.converse_stream.assert_called_once_with(a=1)


def test_converse(bedrock_client, model, messages, tool_spec, model_id, additional_request_fields):
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
    chunks = model.converse(messages, [tool_spec])

    tru_chunks = list(chunks)
    exp_chunks = ["e1", "e2"]

    assert tru_chunks == exp_chunks
    bedrock_client.converse_stream.assert_called_once_with(**request)


def test_converse_stream_input_guardrails(
    bedrock_client, model, messages, tool_spec, model_id, additional_request_fields
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
    chunks = model.converse(messages, [tool_spec])

    tru_chunks = list(chunks)
    exp_chunks = [
        {"redactContent": {"redactUserContentMessage": "[User input redacted.]"}},
        metadata_event,
    ]

    assert tru_chunks == exp_chunks
    bedrock_client.converse_stream.assert_called_once_with(**request)


def test_converse_stream_output_guardrails(
    bedrock_client, model, messages, tool_spec, model_id, additional_request_fields
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
    chunks = model.converse(messages, [tool_spec])

    tru_chunks = list(chunks)
    exp_chunks = [
        {"redactContent": {"redactAssistantContentMessage": "[Assistant output redacted.]"}},
        metadata_event,
    ]

    assert tru_chunks == exp_chunks
    bedrock_client.converse_stream.assert_called_once_with(**request)


def test_converse_output_guardrails_redacts_input_and_output(
    bedrock_client, model, messages, tool_spec, model_id, additional_request_fields
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
    chunks = model.converse(messages, [tool_spec])

    tru_chunks = list(chunks)
    exp_chunks = [
        {"redactContent": {"redactUserContentMessage": "[User input redacted.]"}},
        {"redactContent": {"redactAssistantContentMessage": "[Assistant output redacted.]"}},
        metadata_event,
    ]

    assert tru_chunks == exp_chunks
    bedrock_client.converse_stream.assert_called_once_with(**request)


def test_converse_output_no_blocked_guardrails_doesnt_redact(
    bedrock_client, model, messages, tool_spec, model_id, additional_request_fields
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
    chunks = model.converse(messages, [tool_spec])

    tru_chunks = list(chunks)
    exp_chunks = [metadata_event]

    assert tru_chunks == exp_chunks
    bedrock_client.converse_stream.assert_called_once_with(**request)


def test_converse_output_no_guardrail_redact(
    bedrock_client, model, messages, tool_spec, model_id, additional_request_fields
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
    chunks = model.converse(messages, [tool_spec])

    tru_chunks = list(chunks)
    exp_chunks = [metadata_event]

    assert tru_chunks == exp_chunks
    bedrock_client.converse_stream.assert_called_once_with(**request)


def test_stream_with_streaming_false(bedrock_client):
    """Test stream method with streaming=False."""
    bedrock_client.converse.return_value = {
        "output": {"message": {"role": "assistant", "content": [{"text": "test"}]}},
        "stopReason": "end_turn",
    }
    expected_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "test"}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn", "additionalModelResponseFields": None}},
    ]

    # Create model and call stream
    model = BedrockModel(model_id="test-model", streaming=False)
    request = {"modelId": "test-model"}
    events = list(model.stream(request))

    assert expected_events == events

    bedrock_client.converse.assert_called_once()
    bedrock_client.converse_stream.assert_not_called()


def test_stream_with_streaming_false_and_tool_use(bedrock_client):
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

    expected_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "123", "name": "dummyTool"}}}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"hello": "world!"}'}}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "tool_use", "additionalModelResponseFields": None}},
    ]

    # Create model and call stream
    model = BedrockModel(model_id="test-model", streaming=False)
    request = {"modelId": "test-model"}
    events = list(model.stream(request))

    assert expected_events == events

    bedrock_client.converse.assert_called_once()
    bedrock_client.converse_stream.assert_not_called()


def test_stream_with_streaming_false_and_reasoning(bedrock_client):
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

    expected_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "Thinking really hard...."}}}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"signature": "123"}}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "tool_use", "additionalModelResponseFields": None}},
    ]

    # Create model and call stream
    model = BedrockModel(model_id="test-model", streaming=False)
    request = {"modelId": "test-model"}
    events = list(model.stream(request))

    assert expected_events == events

    # Verify converse was called
    bedrock_client.converse.assert_called_once()
    bedrock_client.converse_stream.assert_not_called()


def test_converse_and_reasoning_no_signature(bedrock_client):
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

    expected_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "Thinking really hard...."}}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "tool_use", "additionalModelResponseFields": None}},
    ]

    # Create model and call stream
    model = BedrockModel(model_id="test-model", streaming=False)
    request = {"modelId": "test-model"}
    events = list(model.stream(request))

    assert expected_events == events

    bedrock_client.converse.assert_called_once()
    bedrock_client.converse_stream.assert_not_called()


def test_stream_with_streaming_false_with_metrics_and_usage(bedrock_client):
    """Test stream method with streaming=False."""
    bedrock_client.converse.return_value = {
        "output": {"message": {"role": "assistant", "content": [{"text": "test"}]}},
        "usage": {"inputTokens": 1234, "outputTokens": 1234, "totalTokens": 2468},
        "metrics": {"latencyMs": 1234},
        "stopReason": "tool_use",
    }

    expected_events = [
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

    # Create model and call stream
    model = BedrockModel(model_id="test-model", streaming=False)
    request = {"modelId": "test-model"}
    events = list(model.stream(request))

    assert expected_events == events

    # Verify converse was called
    bedrock_client.converse.assert_called_once()
    bedrock_client.converse_stream.assert_not_called()


def test_converse_input_guardrails(bedrock_client):
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

    expected_events = [
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

    # Create model and call stream
    model = BedrockModel(model_id="test-model", streaming=False)
    request = {"modelId": "test-model"}
    events = list(model.stream(request))

    assert expected_events == events

    bedrock_client.converse.assert_called_once()
    bedrock_client.converse_stream.assert_not_called()


def test_converse_output_guardrails(bedrock_client):
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

    expected_events = [
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

    model = BedrockModel(model_id="test-model", streaming=False)
    request = {"modelId": "test-model"}
    events = list(model.stream(request))

    assert expected_events == events

    bedrock_client.converse.assert_called_once()
    bedrock_client.converse_stream.assert_not_called()


def test_converse_output_guardrails_redacts_output(bedrock_client):
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

    expected_events = [
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

    model = BedrockModel(model_id="test-model", streaming=False)
    request = {"modelId": "test-model"}
    events = list(model.stream(request))

    assert expected_events == events

    bedrock_client.converse.assert_called_once()
    bedrock_client.converse_stream.assert_not_called()
