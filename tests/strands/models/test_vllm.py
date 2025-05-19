
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
    assert request["prompt"].startswith("user: Hello")
    assert request["model"] == model.get_config()["model_id"]
    assert request["stream"] is False


def test_format_request_with_system_prompt(model, messages, system_prompt):
    request = model.format_request(messages, system_prompt=system_prompt)
    assert request["prompt"].startswith(f"system: {system_prompt}\nuser: Hello")


def test_format_chunk_text():
    chunk = {"choices": [{"text": "World"}]}
    formatted = VLLMModel.format_chunk(None, chunk)
    assert formatted == {"contentBlockDelta": {"delta": {"text": "World"}}}


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
        'data: {"choices":[{"text":"Hello"}]}\n',
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
    stream = list(model.stream(request))

    assert {"chunk_type": "message_start"} in stream
    assert any(chunk.get("chunk_type") == "content_delta" for chunk in stream)
    assert {"chunk_type": "content_stop", "data_type": "text"} in stream
    assert {"chunk_type": "message_stop", "data": "stop"} in stream


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
