import json
import unittest.mock
import uuid

import boto3
import pytest
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError

import strands
from strands.models.sagemaker import SageMakerAIModel
from strands.types.exceptions import ModelThrottledException


@pytest.fixture
def sagemaker_client():
    with unittest.mock.patch.object(strands.models.sagemaker.boto3, "Session") as mock_session_cls:
        yield mock_session_cls.return_value.client.return_value


@pytest.fixture
def endpoint_name():
    return "test-endpoint"


@pytest.fixture
def inference_component_name():
    return "test-inference-component"


@pytest.fixture
def model(sagemaker_client, endpoint_name, inference_component_name):
    _ = sagemaker_client
    return SageMakerAIModel(endpoint_name=endpoint_name, inference_component_name=inference_component_name)


@pytest.fixture
def messages():
    return [{"role": "user", "content": [{"text": "test"}]}]


@pytest.fixture
def system_prompt():
    return "system prompt"


@pytest.fixture
def tool_specs():
    return [
        {
            "name": "calculator",
            "description": "Calculate mathematical expressions",
            "inputSchema": {
                "json": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
            },
        }
    ]


def test__init__default_config(sagemaker_client, endpoint_name):
    _ = sagemaker_client
    model = SageMakerAIModel(endpoint_name=endpoint_name)
    
    config = model.get_config()
    assert config["endpoint_name"] == endpoint_name
    assert config.get("inference_component_name") is None


def test__init__with_inference_component(sagemaker_client, endpoint_name, inference_component_name):
    _ = sagemaker_client
    model = SageMakerAIModel(endpoint_name=endpoint_name, inference_component_name=inference_component_name)
    
    config = model.get_config()
    assert config["endpoint_name"] == endpoint_name
    assert config["inference_component_name"] == inference_component_name


def test__init__with_boto_session(endpoint_name):
    mock_session = unittest.mock.Mock(spec=boto3.Session)
    model = SageMakerAIModel(endpoint_name=endpoint_name, boto_session=mock_session)
    
    mock_session.client.assert_called_once_with(
        service_name="sagemaker-runtime",
        config=None,
    )


def test__init__with_boto_client_config(endpoint_name):
    mock_config = unittest.mock.Mock(spec=BotocoreConfig)
    
    with unittest.mock.patch.object(strands.models.sagemaker.boto3, "Session") as mock_session_cls:
        model = SageMakerAIModel(endpoint_name=endpoint_name, boto_client_config=mock_config)
        mock_session_cls.return_value.client.assert_called_once_with(
            service_name="sagemaker-runtime",
            config=mock_config,
        )


def test__init__with_retry_config(sagemaker_client, endpoint_name):
    _ = sagemaker_client
    retry_attempts = 5
    retry_delay = 60
    
    model = SageMakerAIModel(endpoint_name=endpoint_name, retry_attempts=retry_attempts, retry_delay=retry_delay)
    
    assert model.retry_attempts == retry_attempts
    assert model.retry_delay == retry_delay


def test_update_config(model, endpoint_name):
    new_endpoint = "new-endpoint"
    model.update_config(endpoint_name=new_endpoint)
    
    config = model.get_config()
    assert config["endpoint_name"] == new_endpoint


def test_format_request_default(model, messages, endpoint_name):
    with unittest.mock.patch("uuid.uuid4", return_value=unittest.mock.Mock(hex="123456789")):
        request = model.format_request(messages)
    
    assert request["EndpointName"] == endpoint_name
    assert request["InferenceComponentName"] == model.config["inference_component_name"]
    assert request["ContentType"] == "application/json"
    assert request["Accept"] == "application/json"
    
    body = json.loads(request["Body"])
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == "test"


def test_format_request_with_system_prompt(model, messages, system_prompt):
    request = model.format_request(messages, system_prompt=system_prompt)
    
    body = json.loads(request["Body"])
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == system_prompt
    assert body["messages"][1]["role"] == "user"


def test_format_request_with_tool_specs(model, messages, tool_specs):
    request = model.format_request(messages, tool_specs=tool_specs)
    
    body = json.loads(request["Body"])
    assert len(body["tools"]) == 1
    assert body["tools"][0]["type"] == "function"
    assert body["tools"][0]["function"]["name"] == "calculator"
    assert body["tools"][0]["function"]["description"] == "Calculate mathematical expressions"


def test_format_request_with_config_params(model, messages):
    model.update_config(max_tokens=100, temperature=0.7, top_p=0.9, stop_sequences=["stop"])
    
    request = model.format_request(messages)
    body = json.loads(request["Body"])
    
    assert body["max_tokens"] == 100
    assert body["temperature"] == 0.7
    assert body["top_p"] == 0.9
    assert body["stop"] == ["stop"]


def test_format_request_with_additional_args(model, messages):
    model.update_config(additional_args={"frequency_penalty": 0.5})
    
    request = model.format_request(messages)
    body = json.loads(request["Body"])
    
    assert body["frequency_penalty"] == 0.5


def test_format_request_with_image(model):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": "jpg",
                        "source": {"bytes": b"base64encodedimage"},
                    },
                },
            ],
        },
    ]
    
    request = model.format_request(messages)
    body = json.loads(request["Body"])
    
    assert body["messages"][0]["role"] == "user"
    assert "images" in body["messages"][0]
    assert body["messages"][0]["images"][0] == "base64encodedimage"


def test_format_request_with_tool_use(model):
    tool_id = "tool123"
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": tool_id,
                        "name": "calculator",
                        "input": {"expression": "2+2"},
                    },
                },
            ],
        },
    ]
    
    request = model.format_request(messages)
    body = json.loads(request["Body"])
    
    assert body["messages"][0]["role"] == "assistant"
    assert "tool_calls" in body["messages"][0]
    assert body["messages"][0]["tool_calls"][0]["id"] == tool_id
    assert body["messages"][0]["tool_calls"][0]["type"] == "function"
    assert body["messages"][0]["tool_calls"][0]["function"]["name"] == "calculator"
    assert json.loads(body["messages"][0]["tool_calls"][0]["function"]["arguments"]) == {"expression": "2+2"}


def test_format_request_with_tool_result_text(model):
    tool_id = "tool123"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": tool_id,
                        "status": "success",
                        "content": [{"text": "4"}],
                    },
                },
            ],
        },
    ]
    
    request = model.format_request(messages)
    body = json.loads(request["Body"])
    
    assert body["messages"][0]["role"] == "tool"
    assert body["messages"][0]["name"] == tool_id
    assert body["messages"][0]["tool_call_id"] == tool_id
    assert "content" in body["messages"][0]
    content_obj = json.loads(body["messages"][0]["content"])
    assert content_obj["result"] == "4"
    assert content_obj["status"] == "success"


def test_format_request_with_tool_result_image(model):
    tool_id = "image_gen"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": tool_id,
                        "status": "success",
                        "content": [
                            {"text": "see images"},
                            {"image": {"source": {"bytes": b"base64encodedimage"}}},
                        ],
                    },
                },
            ],
        },
    ]
    
    request = model.format_request(messages)
    body = json.loads(request["Body"])
    
    assert body["messages"][0]["role"] == "tool"
    assert body["messages"][0]["name"] == tool_id
    assert body["messages"][0]["tool_call_id"] == tool_id
    assert "images" in body["messages"][0]
    assert body["messages"][0]["images"][0] == "base64encodedimage"
    content_obj = json.loads(body["messages"][0]["content"])
    assert content_obj["result"] == "see images"


def test_format_request_empty_content_handling(model):
    messages = [
        {"role": "assistant", "content": [{"text": None}]},
        {"role": "assistant", "content": [{"text": ""}]},
    ]
    
    request = model.format_request(messages)
    body = json.loads(request["Body"])
    
    # Check that empty content is replaced with "Thinking ..."
    assert body["messages"][0]["content"] == "Thinking ..."
    assert body["messages"][1]["content"] == "Thinking ..."


def test_format_chunk_message_start(model):
    event = {"chunk_type": "message_start"}
    
    chunk = model.format_chunk(event)
    
    assert chunk == {"messageStart": {"role": "assistant"}}


def test_format_chunk_content_start_text(model):
    event = {"chunk_type": "content_start", "data_type": "text"}
    
    chunk = model.format_chunk(event)
    
    assert chunk == {"contentBlockStart": {"start": {}}}


def test_format_chunk_content_start_tool_use(model):
    with unittest.mock.patch("uuid.uuid4", return_value=unittest.mock.Mock(hex="123456789")):
        event = {
            "chunk_type": "content_start", 
            "data_type": "tool", 
            "data": {"function": {"name": "calculator"}}
        }
        
        chunk = model.format_chunk(event)
        
        assert chunk["contentBlockStart"]["start"]["toolUse"]["name"] == "calculator"
        assert chunk["contentBlockStart"]["start"]["toolUse"]["toolUseId"] is not None


def test_format_chunk_content_delta_text(model):
    event = {"chunk_type": "content_delta", "data_type": "text", "data": "Hello world"}
    
    chunk = model.format_chunk(event)
    
    assert chunk == {"contentBlockDelta": {"delta": {"text": "Hello world"}}}


def test_format_chunk_content_delta_tool_use(model):
    event = {
        "chunk_type": "content_delta", 
        "data_type": "tool", 
        "data": {"function": {"arguments": '{"expression": "2+2"}'}}
    }
    
    chunk = model.format_chunk(event)
    
    assert chunk == {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"expression": "2+2"}'}}}}


def test_format_chunk_content_stop(model):
    event = {"chunk_type": "content_stop"}
    
    chunk = model.format_chunk(event)
    
    assert chunk == {"contentBlockStop": {}}


def test_format_chunk_message_stop_end_turn(model):
    event = {"chunk_type": "message_stop", "data": "stop"}
    
    chunk = model.format_chunk(event)
    
    assert chunk == {"messageStop": {"stopReason": "end_turn"}}


def test_format_chunk_message_stop_tool_use(model):
    event = {"chunk_type": "message_stop", "data": "tool_use"}
    
    chunk = model.format_chunk(event)
    
    assert chunk == {"messageStop": {"stopReason": "tool_use"}}


def test_format_chunk_message_stop_max_tokens(model):
    event = {"chunk_type": "message_stop", "data": "length"}
    
    chunk = model.format_chunk(event)
    
    assert chunk == {"messageStop": {"stopReason": "max_tokens"}}


def test_format_chunk_metadata(model):
    event = {
        "chunk_type": "metadata", 
        "data": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }
    
    chunk = model.format_chunk(event)
    
    assert chunk == {
        "metadata": {
            "usage": {
                "inputTokens": 10,
                "outputTokens": 20,
                "totalTokens": 30,
            },
            "metrics": {
                "latencyMs": 0,
            },
        },
    }


def test_format_chunk_unknown_type(model):
    event = {"chunk_type": "unknown"}
    
    with pytest.raises(RuntimeError, match="chunk_type=<unknown> | unknown type"):
        model.format_chunk(event)


def test_stream(sagemaker_client, model, messages):
    # Mock the SageMaker response
    mock_response = {
        "Body": [
            {"PayloadPart": {"Bytes": json.dumps({
                "choices": [{"message": {"content": "Hello", "tool_calls": None}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            }).encode("utf-8")}}
        ]
    }
    sagemaker_client.invoke_endpoint_with_response_stream.return_value = mock_response
    
    request = model.format_request(messages)
    chunks = list(model.stream(request))
    
    # Verify the expected chunks
    assert len(chunks) == 5
    assert chunks[0] == {"chunk_type": "message_start"}
    assert chunks[1] == {"chunk_type": "content_start", "data_type": "text"}
    assert chunks[2] == {"chunk_type": "content_delta", "data_type": "text", "data": "Hello"}
    assert chunks[3] == {"chunk_type": "content_stop", "data_type": "text"}
    assert chunks[4] == {"chunk_type": "message_stop", "data": "stop"}
    
    # Verify the SageMaker client was called correctly
    sagemaker_client.invoke_endpoint_with_response_stream.assert_called_once_with(**request)


def test_stream_with_tool_calls(sagemaker_client, model, messages):
    # Mock the SageMaker response with tool calls
    tool_call = {
        "id": "tool123",
        "type": "function",
        "function": {
            "name": "calculator",
            "arguments": '{"expression": "2+2"}'
        }
    }
    
    mock_response = {
        "Body": [
            {"PayloadPart": {"Bytes": json.dumps({
                "choices": [{"message": {"content": "", "tool_calls": [tool_call]}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            }).encode("utf-8")}}
        ]
    }
    sagemaker_client.invoke_endpoint_with_response_stream.return_value = mock_response
    
    request = model.format_request(messages)
    chunks = list(model.stream(request))
    
    # Verify the expected chunks for tool calls
    assert chunks[0] == {"chunk_type": "message_start"}
    assert chunks[1] == {"chunk_type": "content_start", "data_type": "text"}
    
    # Tool call chunks
    assert chunks[2]["chunk_type"] == "content_start"
    assert chunks[2]["data_type"] == "tool"
    assert chunks[3]["chunk_type"] == "content_delta"
    assert chunks[3]["data_type"] == "tool"
    assert chunks[4]["chunk_type"] == "content_stop"
    
    # Content delta for "Thinking..." since content was empty
    assert chunks[5]["chunk_type"] == "content_delta"
    assert chunks[5]["data_type"] == "text"
    assert chunks[5]["data"] == "Thinking...\n"
    
    # Final chunks
    assert chunks[6]["chunk_type"] == "content_stop"
    assert chunks[7]["chunk_type"] == "message_stop"
    assert chunks[7]["data"] == "tool_use"