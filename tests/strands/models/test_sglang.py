import unittest.mock

import pytest

import strands
from strands.models.sglang import SGLangModel


@pytest.fixture
def httpx_client():
    with unittest.mock.patch.object(strands.models.sglang.httpx, "AsyncClient") as mock_client_cls:
        mock_client = unittest.mock.Mock()
        mock_client_cls.return_value = mock_client

        # httpx.AsyncClient.stream(...) returns an async context manager.
        stream_cm = unittest.mock.Mock()
        stream_cm.__aenter__ = unittest.mock.AsyncMock()
        stream_cm.__aexit__ = unittest.mock.AsyncMock(return_value=None)
        mock_client.stream.return_value = stream_cm

        yield mock_client


def _aline_iter(lines: list[str]):
    async def gen():
        for line in lines:
            yield line

    return gen()


@pytest.mark.asyncio
async def test_sglang_stream_parses_sse_and_emits_text_deltas(httpx_client, alist):
    # Mock /generate stream SSE
    resp = unittest.mock.Mock()
    resp.raise_for_status = unittest.mock.Mock()
    resp.aiter_lines = unittest.mock.Mock(
        return_value=_aline_iter(
            [
                'data: {"text":"h","output_ids":[1],"meta_info":{"finish_reason":{"type":"stop"},'
                '"prompt_tokens":2,"completion_tokens":1,"e2e_latency":0.01}}',
                'data: {"text":"hi","output_ids":[1,2],"meta_info":{"finish_reason":{"type":"stop"},'
                '"prompt_tokens":2,"completion_tokens":2,"e2e_latency":0.01}}',
                "data: [DONE]",
            ]
        )
    )

    # Async context manager for client.stream(...)
    httpx_client.stream.return_value.__aenter__.return_value = resp
    httpx_client.stream.return_value.__aexit__.return_value = None

    model = SGLangModel(base_url="http://localhost:30000", model_id=None, params=None, return_token_ids=False)
    events = await alist(model.stream([{"role": "user", "content": [{"text": "hi"}]}]))

    assert events[0] == {"messageStart": {"role": "assistant"}}
    assert events[1] == {"contentBlockStart": {"start": {}}}
    # deltas should be incremental: "h" then "i"
    deltas = [e["contentBlockDelta"]["delta"]["text"] for e in events if "contentBlockDelta" in e]
    assert deltas == ["h", "i"]

    stop = next(e for e in events if "messageStop" in e)["messageStop"]
    additional = stop["additionalModelResponseFields"]
    assert additional["token_ids"] == [1, 2]


@pytest.mark.asyncio
async def test_sglang_token_in_preserves_prompt_token_ids(httpx_client, alist):
    resp = unittest.mock.Mock()
    resp.raise_for_status = unittest.mock.Mock()
    resp.aiter_lines = unittest.mock.Mock(
        return_value=_aline_iter(
            [
                'data: {"text":"ok","output_ids":[9,10],"meta_info":{"finish_reason":{"type":"stop"},'
                '"prompt_tokens":3,"completion_tokens":2,"e2e_latency":0.01}}',
                "data: [DONE]",
            ]
        )
    )
    httpx_client.stream.return_value.__aenter__.return_value = resp
    httpx_client.stream.return_value.__aexit__.return_value = None

    model = SGLangModel(base_url="http://localhost:30000", model_id=None, params=None, return_token_ids=False)
    events = await alist(
        model.stream(
            [{"role": "user", "content": [{"text": "ignored"}]}],
            prompt_token_ids=[1, 2, 3],
            temperature=0,
        )
    )

    # Ensure token-in was sent as input_ids
    called = httpx_client.stream.call_args.kwargs["json"]
    assert called["input_ids"] == [1, 2, 3]
    assert called["stream"] is True

    stop = next(e for e in events if "messageStop" in e)["messageStop"]
    additional = stop["additionalModelResponseFields"]
    assert additional["prompt_token_ids"] == [1, 2, 3]
    assert additional["token_ids"] == [9, 10]


@pytest.mark.asyncio
async def test_sglang_text_prompt_token_out_uses_tokenize_when_enabled(httpx_client, alist):
    # Mock /tokenize
    tok_resp = unittest.mock.Mock()
    tok_resp.raise_for_status = unittest.mock.Mock()
    tok_resp.json = unittest.mock.Mock(return_value={"tokens": [101, 102]})
    httpx_client.post = unittest.mock.AsyncMock(return_value=tok_resp)

    # Mock /generate stream
    resp = unittest.mock.Mock()
    resp.raise_for_status = unittest.mock.Mock()
    resp.aiter_lines = unittest.mock.Mock(
        return_value=_aline_iter(
            [
                'data: {"text":"yo","output_ids":[7],"meta_info":{"finish_reason":{"type":"stop"}}}',
                "data: [DONE]",
            ]
        )
    )
    httpx_client.stream.return_value.__aenter__.return_value = resp
    httpx_client.stream.return_value.__aexit__.return_value = None

    model = SGLangModel(base_url="http://localhost:30000", model_id="m1", params=None, return_token_ids=True)
    events = await alist(model.stream([{"role": "user", "content": [{"text": "hello"}]}]))

    # tokenization called
    httpx_client.post.assert_awaited()
    assert httpx_client.post.call_args.args[0] == "/tokenize"

    stop = next(e for e in events if "messageStop" in e)["messageStop"]
    additional = stop["additionalModelResponseFields"]
    assert additional["prompt_token_ids"] == [101, 102]
    assert additional["token_ids"] == [7]
