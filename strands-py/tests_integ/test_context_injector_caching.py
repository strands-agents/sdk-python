"""Integration tests for ContextInjector and Bedrock prompt caching, against a real model.

Injected context is ephemeral — it is present on one model call and gone (or re-rendered) on the
next. If it lands inside the cached prefix, the prefix diverges exactly where the injection sat
and the cache is lost from that point on. These tests assert the real Bedrock cache counters, so
they fail if the ephemeral boundary stops being honored end to end.
"""

import uuid

import pytest

from strands import Agent, tool
from strands.models import BedrockModel
from strands.models.model import CacheConfig
from strands.vended_plugins.context_injector import ContextInjector

_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


class _UsageCapturingBedrockModel(BedrockModel):
    """Records the per-call cache counters Bedrock reports.

    ``AgentResult.metrics.accumulated_usage`` sums across calls, which hides whether an individual
    call read from the cache; these tests need the per-call values.
    """

    def __init__(self, **config):
        super().__init__(**config)
        self.usage_per_call: list[dict[str, int]] = []

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        async for event in super().stream(messages, tool_specs, system_prompt, **kwargs):
            usage = event.get("metadata", {}).get("usage") if "metadata" in event else None
            if usage:
                self.usage_per_call.append(
                    {
                        "write": usage.get("cacheWriteInputTokens", 0),
                        "read": usage.get("cacheReadInputTokens", 0),
                    }
                )
            yield event


@pytest.fixture
def cacheable_prefix():
    """A prefix long enough to cache, unique per run so other runs cannot satisfy the read."""
    nonce = uuid.uuid4()
    return f"Reference dossier {nonce}. " + ("The subject prefers concise written answers. " * 400)


@pytest.fixture
def injector():
    """An injector whose text changes on every call — the cache-hostile case."""
    calls = {"n": 0}

    def render(context):
        calls["n"] += 1
        return f"<runtime-context>call={calls['n']}</runtime-context>"

    return ContextInjector(render)


def test_injected_context_does_not_prevent_cache_reads(cacheable_prefix, injector, quiet_strands_logging):
    """A second turn must read the prefix cached by the first, despite per-call injected text."""
    model = _UsageCapturingBedrockModel(model_id=_MODEL_ID, streaming=False, cache_config=CacheConfig(strategy="auto"))
    agent = Agent(
        model=model,
        plugins=[injector],
        load_tools_from_directory=False,
        callback_handler=None,
    )

    agent(f"{cacheable_prefix}\n\nQuestion one: reply with the single word ALPHA.")
    agent("Question two: reply with the single word BETA.")

    first_call, second_call = model.usage_per_call[0], model.usage_per_call[1]
    assert first_call["write"] > 0, f"expected the first call to write a cache entry, got {first_call}"
    assert second_call["read"] > 0, (
        f"expected the second call to read the cached prefix, got {second_call}. The injected block "
        "is likely inside the cached prefix again."
    )
    # The durable prefix is served from cache rather than rewritten wholesale.
    assert second_call["write"] < first_call["write"]


def test_injected_context_does_not_reach_durable_history(cacheable_prefix, injector, quiet_strands_logging):
    """The ephemerality contract holds against a real model, not just a mock."""
    agent = Agent(
        model=BedrockModel(model_id=_MODEL_ID, streaming=False),
        plugins=[injector],
        load_tools_from_directory=False,
        callback_handler=None,
    )

    agent(f"{cacheable_prefix}\n\nReply with the single word ALPHA.")

    durable_text = " ".join(
        block["text"] for message in agent.messages for block in message["content"] if "text" in block
    )
    assert "<runtime-context>" not in durable_text


def test_every_turn_injection_keeps_caching_across_a_tool_loop(cacheable_prefix, quiet_strands_logging):
    """``everyTurn`` re-renders on tool-result turns too, so every iteration must still cache.

    Also exercises the placement constraint: a tool result has to stay the first block of its
    message, and Bedrock rejects a cache point placed directly before one, so a bad boundary
    surfaces here as a ValidationException rather than a silent cache miss.
    """

    @tool
    def lookup_price(item: str) -> str:
        """Look up the price of an item.

        Args:
            item: The item to price.
        """
        return f"The price of {item} is 42 dollars."

    calls = {"n": 0}

    def render(context):
        calls["n"] += 1
        return f"<runtime-context>call={calls['n']}</runtime-context>"

    model = _UsageCapturingBedrockModel(model_id=_MODEL_ID, streaming=False, cache_config=CacheConfig(strategy="auto"))
    agent = Agent(
        model=model,
        tools=[lookup_price],
        plugins=[ContextInjector(render, trigger="everyTurn")],
        load_tools_from_directory=False,
        callback_handler=None,
    )

    agent(f"{cacheable_prefix}\n\nUse the lookup_price tool to find the price of a widget, then state it.")

    # The loop runs several model calls; every call after the first should read the warm prefix.
    assert len(model.usage_per_call) > 1, "expected the tool loop to make more than one model call"
    assert sum(call["read"] for call in model.usage_per_call) > 0, (
        f"expected cache reads across the tool loop, got {model.usage_per_call}"
    )
