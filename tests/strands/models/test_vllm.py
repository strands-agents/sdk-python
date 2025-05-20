import pytest
import requests
from strands.models.vllm import VLLMModel


@pytest.fixture
def model_id():
    return "meta-llama/Llama-3.2-3B"


@pytest.fixture
def host():
    return "http://localhost:8000"


@pytest.fixture
def model(model_id, host):
    return VLLMModel(host, model_id=model_id, max_tokens=128)


@pytest.fixture
def messages():
    return [{"role": "user", "content": [{"text": "Hello"}]}]


@pytest.fixture
def system_prompt():
    return "You are a helpful assistant."


def test_init_sets_config(model, model_id):
    assert model.get_config()["model_id"] == model_id
    assert model.host == "http://localhost:8000"


def test_update_config_overrides(model):
    model.update_config(temperature=0.3)
    assert model.get_config()["temperature"] == 0.3


def test_format_request_basic(model, messages):
    request = model.format_request(messages)
    assert request["model"] == model.get_config()["model_id"]
    assert isinstance(request["messages"], list)
    assert request["messages"][0]["role"] == "user"
    assert request["messages"][0]["content"] == "Hello"
    assert request["stream"] is True


def test_format_request_with_system_prompt(model, messages, system_prompt):
    request = model.format_request(messages, system_prompt=system_prompt)
    assert request["messages"][0]["role"] == "system"
    assert request["messages"][0]["content"] == system_prompt


def test_format_chunk_text():
    chunk = {"choices": [{"delta": {"content": "World"}}]}
    formatted = VLLMModel.format_chunk(None, chunk)
    assert formatted == {"contentBlockDelta": {"delta": {"text": "World"}}}


def test_format_chunk_tool_call():
    chunk = {
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "id": "abc123",
                    "function": {
                        "name": "get_time",
                        "arguments": '{"timezone":"UTC"}'
                    }
                }]
            }
        }]
    }
    formatted = VLLMModel.format_chunk(None, chunk)
    assert formatted == {"toolCall": chunk["choices"][0]["delta"]["tool_calls"][0]}


def test_format_chunk_finish_reason():
    chunk = {"choices": [{"finish_reason": "stop"}]}
    formatted = VLLMModel.format_chunk(None, chunk)
    assert formatted == {"messageStop": {"stopReason": "stop"}}


def test_format_chunk_empty():
    chunk = {"choices": [{}]}
    formatted = VLLMModel.format_chunk(None, chunk)
    assert formatted == {}


def test_stream_response(monkeypatch, model, messages):
    mock_lines = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
        'data: {"choices":[{"finish_reason":"stop"}]}\n',
        "data: [DONE]\n",
    ]

    class MockResponse:
        def __init__(self):
            self.status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def iter_lines(self, decode_unicode=False):
            return iter(mock_lines)

    monkeypatch.setattr(requests, "post", lambda *a, **kw: MockResponse())

    request = model.format_request(messages)
    chunks = list(model.stream(request))

    assert {"chunk_type": "message_start"} in chunks
    assert any(chunk.get("chunk_type") == "content_delta" for chunk in chunks)
    assert {"chunk_type": "content_stop", "data_type": "text"} in chunks
    assert {"chunk_type": "message_stop", "data": "end_turn"} in chunks


def test_stream_tool_call(monkeypatch, model, messages):
    mock_lines = [
        'data: {"choices":[{"delta":{"tool_calls":[{"id":"abc","function":{"name":"current_time","arguments":"{\\"timezone\\": \\"UTC\\"}"}}]}}]}\n',
        "data: [DONE]\n",
    ]

    class MockResponse:
        def __init__(self):
            self.status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def iter_lines(self, decode_unicode=False):
            return iter(mock_lines)

    monkeypatch.setattr(requests, "post", lambda *a, **kw: MockResponse())

    request = model.format_request(messages)
    chunks = list(model.stream(request))

    assert any("toolCallStart" in c for c in chunks)
    assert any("toolCallDelta" in c for c in chunks)


def test_stream_server_error(monkeypatch, model, messages):
    class ErrorResponse:
        def __init__(self):
            self.status_code = 500
            self.text = "Internal Error"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def iter_lines(self, decode_unicode=False):
            return iter([])

    monkeypatch.setattr(requests, "post", lambda *a, **kw: ErrorResponse())

    with pytest.raises(Exception, match="Request failed: 500"):
        list(model.stream(model.format_request(messages)))
