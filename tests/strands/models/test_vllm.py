import pytest
import requests
import json

from types import SimpleNamespace
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
    chunk = {"chunk_type": "content_delta", "data_type": "text", "data": "World"}
    formatted = VLLMModel.format_chunk(None, chunk)
    assert formatted == {"contentBlockDelta": {"delta": {"text": "World"}}}


def test_format_chunk_tool_call_delta():
    chunk = {
        "chunk_type": "content_delta",
        "data_type": "tool",
        "data": SimpleNamespace(name="get_time", arguments={"timezone": "UTC"}),
    }

    formatted = VLLMModel.format_chunk(None, chunk)
    assert "contentBlockDelta" in formatted
    assert "toolUse" in formatted["contentBlockDelta"]["delta"]
    assert json.loads(formatted["contentBlockDelta"]["delta"]["toolUse"]["input"])["timezone"] == "UTC"


def test_stream_response(monkeypatch, model, messages):
    mock_lines = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
        'data: {"choices":[{"delta":{"content":" world"}}]}\n',
        "data: [DONE]\n",
    ]

    class MockResponse:
        def __init__(self):
            self.status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *a): pass

        def iter_lines(self, decode_unicode=False):
            return iter(mock_lines)

    monkeypatch.setattr(requests, "post", lambda *a, **kw: MockResponse())

    chunks = list(model.stream(model.format_request(messages)))
    chunk_types = [c.get("chunk_type") for c in chunks]

    assert "message_start" in chunk_types
    assert chunk_types.count("content_delta") == 2
    assert "content_stop" in chunk_types
    assert "message_stop" in chunk_types


def test_stream_tool_call(monkeypatch, model, messages):
    tool_call = {
        "name": "current_time",
        "arguments": {"timezone": "UTC"},
    }
    tool_call_json = json.dumps(tool_call)
    data_str = json.dumps({
        "choices": [
            {"delta": {"content": f"<tool_call>{tool_call_json}</tool_call>"}}
        ]
    })
    mock_lines = [
        'data: {"choices":[{"delta":{"content":"Some answer before tool."}}]}\n',
        f"data: {data_str}\n",
        "data: [DONE]\n",
    ]

    class MockResponse:
        def __init__(self): self.status_code = 200
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def iter_lines(self, decode_unicode=False): return iter(mock_lines)

    monkeypatch.setattr(requests, "post", lambda *a, **kw: MockResponse())

    chunks = list(model.stream(model.format_request(messages)))
    tool_chunks = [c for c in chunks if c.get("chunk_type") == "content_start" and c.get("data_type") == "tool"]

    assert tool_chunks
    assert any("tool_use" in c.get("chunk_type", "") or "tool" in c.get("data_type", "") for c in chunks)



def test_stream_server_error(monkeypatch, model, messages):
    class ErrorResponse:
        def __init__(self):
            self.status_code = 500
            self.text = "Internal Error"
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def iter_lines(self, decode_unicode=False): return iter([])

    monkeypatch.setattr(requests, "post", lambda *a, **kw: ErrorResponse())

    with pytest.raises(Exception, match="Request failed: 500"):
        list(model.stream(model.format_request(messages)))
