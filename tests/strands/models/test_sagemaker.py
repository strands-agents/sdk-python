import json
import unittest.mock

import boto3
import pytest

from strands.models.sagemaker import SageMakerAIModel


@pytest.fixture
def boto_session():
    with unittest.mock.patch.object(boto3, "Session") as mock_session:
        yield mock_session.return_value


@pytest.fixture
def sagemaker_client(boto_session):
    return boto_session.client.return_value


@pytest.fixture
def model_config():
    return {
        "endpoint_name": "test-endpoint",
        "inference_component_name": "test-component",
        "max_tokens": 1024,
        "temperature": 0.7,
    }


@pytest.fixture
def model(boto_session, model_config):
    return SageMakerAIModel(boto_session=boto_session, **model_config)


@pytest.fixture
def messages():
    return [{"role": "user", "content": [{"text": "What is the capital of France?"}]}]


@pytest.fixture
def system_prompt():
    return "You are a helpful assistant."


def test_init(boto_session, model_config):
    model = SageMakerAIModel(boto_session=boto_session, **model_config)

    assert model.config == model_config
    boto_session.client.assert_called_once_with(
        service_name="sagemaker-runtime",
        config=unittest.mock.ANY,
    )


def test_update_config(model, model_config):
    new_config = {"temperature": 0.5, "top_p": 0.9}
    model.update_config(**new_config)

    expected_config = {**model_config, **new_config}
    assert model.config == expected_config


def test_format_request(model, messages, system_prompt):
    tool_specs = [
        {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "inputSchema": {"json": {"type": "object", "properties": {"location": {"type": "string"}}}},
        }
    ]

    request = model.format_request(messages, tool_specs, system_prompt)

    assert request["EndpointName"] == "test-endpoint"
    assert request["InferenceComponentName"] == "test-component"
    assert request["ContentType"] == "application/json"
    assert request["Accept"] == "application/json"

    payload = json.loads(request["Body"])
    assert "messages" in payload
    assert "tools" in payload
    assert payload["max_tokens"] == 1024
    assert payload["temperature"] == 0.7


def test_stream(sagemaker_client, model):
    # Mock the response from SageMaker
    mock_response = {
        "Body": [
            {
                "PayloadPart": {
                    "Bytes": json.dumps(
                        {
                            "choices": [
                                {
                                    "message": {"content": "Paris is the capital of France.", "tool_calls": None},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 20,
                                "total_tokens": 30,
                                "prompt_tokens_details": 10,
                            },
                        }
                    ).encode("utf-8")
                }
            }
        ]
    }
    sagemaker_client.invoke_endpoint_with_response_stream.return_value = mock_response

    request = {
        "EndpointName": "test-endpoint",
        "Body": "{}",
        "ContentType": "application/json",
        "Accept": "application/json",
    }
    response = model.stream(request)

    events = list(response)
    print(events)

    assert len(events) == 6
    assert events[0] == {"chunk_type": "message_start"}
    assert events[1] == {"chunk_type": "content_start", "data_type": "text"}
    assert events[2] == {"chunk_type": "content_delta", "data_type": "text", "data": "Paris is the capital of France."}
    assert events[3] == {"chunk_type": "content_stop", "data_type": "text"}
    assert events[4]["chunk_type"] == "message_stop"

    sagemaker_client.invoke_endpoint_with_response_stream.assert_called_once_with(**request)


def test_stream_with_tool_calls(sagemaker_client, model):
    # Mock the response from SageMaker with tool calls
    tool_call = {
        "id": "tool123",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
    }

    mock_response = {
        "Body": [
            {
                "PayloadPart": {
                    "Bytes": json.dumps(
                        {
                            "choices": [
                                {"message": {"content": "", "tool_calls": [tool_call]}, "finish_reason": "tool_calls"}
                            ],
                            "usage": {
                                "prompt_tokens": 15,
                                "completion_tokens": 25,
                                "total_tokens": 40,
                                "prompt_tokens_details": 15,
                            },
                        }
                    ).encode("utf-8")
                }
            }
        ]
    }
    sagemaker_client.invoke_endpoint_with_response_stream.return_value = mock_response

    request = {
        "EndpointName": "test-endpoint",
        "Body": "{}",
        "ContentType": "application/json",
        "Accept": "application/json",
    }
    response = model.stream(request)

    events = list(response)
    print(events)

    assert len(events) == 9
    assert events[0] == {"chunk_type": "message_start"}
    assert events[1] == {"chunk_type": "content_start", "data_type": "text"}
    assert events[2] == {"chunk_type": "content_delta", "data_type": "text", "data": ""}
    assert events[3] == {"chunk_type": "content_stop", "data_type": "text"}
    assert events[4]["chunk_type"] == "content_start"
    assert events[4]["data_type"] == "tool"
    assert events[5]["chunk_type"] == "content_delta"
    assert events[6]["chunk_type"] == "content_stop"
    assert events[7]["chunk_type"] == "message_stop"
    assert events[7]["data"] == "tool_calls"
