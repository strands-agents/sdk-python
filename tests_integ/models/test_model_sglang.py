import pytest

from strands import Agent
from strands.models.sglang import SGLangModel
from tests_integ.models import providers

# These tests only run if a local SGLang server is reachable.
pytestmark = providers.sglang.mark


@pytest.fixture
def model() -> SGLangModel:
    return providers.sglang.create_model()  # type: ignore[return-value]


@pytest.fixture
def agent(model: SGLangModel) -> Agent:
    return Agent(model=model)


def _additional(result_message: dict) -> dict:
    additional = result_message.get("additionalModelResponseFields")
    assert isinstance(additional, dict), f"missing additionalModelResponseFields: {result_message}"
    return additional


def test_agent_invoke_preserves_token_ids(agent: Agent) -> None:
    result = agent("hi", invocation_state={"model_kwargs": {"return_token_ids": True}})
    additional = _additional(result.message)
    assert isinstance(additional.get("token_ids"), list) and additional["token_ids"]
    assert isinstance(additional.get("prompt_token_ids"), list) and additional["prompt_token_ids"]


@pytest.mark.asyncio
async def test_agent_stream_async_preserves_token_ids(agent: Agent) -> None:
    stream = agent.stream_async("hi", invocation_state={"model_kwargs": {"return_token_ids": True}})
    async for event in stream:
        _ = event
    result = event["result"]
    additional = _additional(result.message)
    assert isinstance(additional.get("token_ids"), list) and additional["token_ids"]
    assert isinstance(additional.get("prompt_token_ids"), list) and additional["prompt_token_ids"]


@pytest.mark.asyncio
async def test_token_in_round_trip_preserves_prompt_token_ids(agent: Agent) -> None:
    # Step 1: get prompt token ids from a text prompt
    res1 = await agent.invoke_async(
        "hi",
        invocation_state={
            "model_kwargs": {
                "return_token_ids": True,
                # Ensure the model stops naturally (avoid MaxTokensReachedException in Agent loop).
                "sampling_params": {"max_new_tokens": 64, "stop": ["\n"]},
            }
        },
    )
    add1 = _additional(res1.message)
    pti = add1["prompt_token_ids"]
    assert isinstance(pti, list) and pti

    # Step 2: token-in call using those prompt_token_ids
    res2 = await agent.invoke_async(
        "ignored",
        invocation_state={
            "model_kwargs": {
                "prompt_token_ids": pti,
                "sampling_params": {"max_new_tokens": 64, "stop": ["\n"]},
            }
        },
    )
    add2 = _additional(res2.message)
    assert add2.get("prompt_token_ids") == pti
    assert isinstance(add2.get("token_ids"), list) and add2["token_ids"]
