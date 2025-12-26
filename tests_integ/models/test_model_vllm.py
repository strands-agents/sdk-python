import os

import pytest

from strands import Agent, tool
from strands.event_loop.streaming import stream_messages
from strands.models.vllm import VLLMModel
from tests_integ.models import providers

# These tests only run if a local vLLM OpenAI-compatible server is reachable.
pytestmark = providers.vllm.mark


@pytest.fixture
def model() -> VLLMModel:
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    model_id = os.getenv("VLLM_MODEL_ID", "AMead10/Llama-3.2-3B-Instruct-AWQ")
    return VLLMModel(
        client_args={"api_key": "EMPTY", "base_url": base_url},
        model_id=model_id,
        params={"max_tokens": 32, "temperature": 0},
        return_token_ids=True,
    )


@pytest.fixture
def agent(model: VLLMModel) -> Agent:
    return Agent(model=model)


def _additional(result_message: dict) -> dict:
    additional = result_message.get("additionalModelResponseFields")
    assert isinstance(additional, dict), f"missing additionalModelResponseFields: {result_message}"
    return additional


def test_agent_invoke_preserves_token_ids(agent: Agent) -> None:
    result = agent(
        "hi",
        invocation_state={"model_kwargs": {"extra_body": {"return_token_ids": True}}},
    )

    additional = _additional(result.message)
    assert isinstance(additional.get("prompt_token_ids"), list) and additional["prompt_token_ids"]
    assert isinstance(additional.get("token_ids"), list) and additional["token_ids"]


@pytest.mark.asyncio
async def test_agent_invoke_async_preserves_token_ids(agent: Agent) -> None:
    result = await agent.invoke_async(
        "hi",
        invocation_state={"model_kwargs": {"extra_body": {"return_token_ids": True}}},
    )

    additional = _additional(result.message)
    assert isinstance(additional.get("prompt_token_ids"), list) and additional["prompt_token_ids"]
    assert isinstance(additional.get("token_ids"), list) and additional["token_ids"]


@pytest.mark.asyncio
async def test_agent_stream_async_preserves_token_ids(agent: Agent) -> None:
    stream = agent.stream_async(
        "hi",
        invocation_state={"model_kwargs": {"extra_body": {"return_token_ids": True}}},
    )

    async for event in stream:
        _ = event

    result = event["result"]
    additional = _additional(result.message)
    assert isinstance(additional.get("prompt_token_ids"), list) and additional["prompt_token_ids"]
    assert isinstance(additional.get("token_ids"), list) and additional["token_ids"]


@pytest.mark.asyncio
async def test_tool_use_stop_event_preserves_token_ids(model: VLLMModel) -> None:
    # Minimal tool; we only need tool specs, not tool execution.
    @tool
    def echo_tool(text: str) -> str:
        return text

    tool_specs = Agent(model=model, tools=[echo_tool]).tool_registry.get_all_tool_specs()

    events: list[dict] = []
    async for event in stream_messages(
        model,
        system_prompt=None,
        messages=[{"role": "user", "content": [{"text": "Call echo_tool with text='hello'. Return nothing else."}]}],
        tool_specs=tool_specs,
        tool_choice={"tool": {"name": "echo_tool"}},
        return_token_ids=True,
        logprobs=1,
        max_tokens=64,
    ):
        events.append(event)

    stop_events = [e["event"] for e in events if isinstance(e, dict) and "event" in e and "messageStop" in e["event"]]
    assert stop_events, f"no messageStop found; got: {events}"

    tool_stop = next((e for e in stop_events if e["messageStop"].get("stopReason") == "tool_use"), None)
    assert tool_stop is not None, "expected stopReason='tool_use' (tool calling may not be enabled on server)"

    additional = tool_stop["messageStop"].get("additionalModelResponseFields")
    assert isinstance(additional, dict), f"missing additionalModelResponseFields: {tool_stop}"
    assert isinstance(additional.get("prompt_token_ids"), list) and additional["prompt_token_ids"]
    assert isinstance(additional.get("token_ids"), list) and additional["token_ids"]
    assert additional.get("logprobs") is not None
