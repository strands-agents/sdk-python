import uuid

import pytest

from strands import Agent, tool
from strands.models import BedrockModel, CacheConfig, CacheToolsConfig
from strands.types.content import ContentBlock, Messages

CACHING_MODEL_ID = "us.anthropic.claude-opus-4-8"


def durable_prefix() -> str:
    """Salted per run so a rerun cannot read an earlier run's entry, and sized past the cache minimum."""
    return f"Dossier {uuid.uuid4()}. " + ("The subject prefers concise written answers. " * 600)


def test_bedrock_cache_point(quiet_strands_logging):
    messages: Messages = [
        {
            "role": "user",
            "content": [
                {
                    # Salted: a fixed prefix stays warm across runs, so this call reads an earlier
                    # run's entry instead of writing its own.
                    "text": f"Some really long text {uuid.uuid4()}! " * 1000  # cachePoint needs >=1024 tokens
                },
                {"cachePoint": {"type": "default"}},
            ],
        },
        {"role": "assistant", "content": [{"text": "Blue!"}]},
    ]

    cache_point_usage = 0

    def cache_point_callback_handler(**kwargs):
        nonlocal cache_point_usage
        if "event" in kwargs and kwargs["event"] and "metadata" in kwargs["event"] and kwargs["event"]["metadata"]:
            metadata = kwargs["event"]["metadata"]
            if "usage" in metadata and metadata["usage"]:
                if "cacheReadInputTokens" in metadata["usage"] or "cacheWriteInputTokens" in metadata["usage"]:
                    cache_point_usage += 1

    agent = Agent(messages=messages, callback_handler=cache_point_callback_handler, load_tools_from_directory=False)
    agent("What is favorite color?")
    assert cache_point_usage > 0


def test_bedrock_multi_prompt_and_duplicate_cache_point(quiet_strands_logging):
    """Test multi-prompt system with cache point."""
    system_prompt_content = [
        {"text": "You are a helpful assistant." * 500},  # Long text for cache
        {"cachePoint": {"type": "default"}},
        {"text": "Always respond with enthusiasm!"},
    ]

    cache_point_usage = 0

    def cache_point_callback_handler(**kwargs):
        nonlocal cache_point_usage
        if "event" in kwargs and kwargs["event"] and "metadata" in kwargs["event"] and kwargs["event"]["metadata"]:
            metadata = kwargs["event"]["metadata"]
            if "usage" in metadata and metadata["usage"]:
                if "cacheReadInputTokens" in metadata["usage"] or "cacheWriteInputTokens" in metadata["usage"]:
                    cache_point_usage += 1

    agent = Agent(
        model=BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0", cache_prompt="default"),
        system_prompt=system_prompt_content,
        callback_handler=cache_point_callback_handler,
        load_tools_from_directory=False,
    )
    agent("Hello!")
    assert cache_point_usage > 0


def test_a_cache_boundary_keeps_per_call_content_out_of_the_cached_prefix(quiet_strands_logging):
    """Content behind a caller's boundary stays uncached instead of being folded into the prefix.

    A read alone does not show this: relocating the point to the end of the message also earns a read,
    because everything including the per-call tail gets cached. The tail's tokens are what tell the
    two apart - they are billed as ordinary input when the boundary is honored, and swallowed into
    cacheWriteInputTokens when it is not.
    """
    tail = f"Addendum {uuid.uuid4()}. " + ("Disregard this filler sentence entirely. " * 600)
    model = BedrockModel(model_id=CACHING_MODEL_ID, cache_config=CacheConfig(strategy="auto"))
    agent = Agent(model=model, load_tools_from_directory=False, callback_handler=None)

    first = agent([{"text": durable_prefix()}, {"cachePoint": {"type": "default"}}, {"text": tail}])
    first_usage = first.metrics.latest_agent_invocation.usage
    assert first_usage.get("cacheWriteInputTokens", 0) > 0, "first turn should have written the prefix"
    assert first_usage.get("inputTokens", 0) > 1000, (
        f"per-call tail was folded into the cached prefix instead of billed as input: {dict(first_usage)}"
    )

    fresh_tail = f"Addendum {uuid.uuid4()}. " + ("Disregard this filler sentence entirely. " * 600)
    second = agent([{"text": "Anything else?"}, {"cachePoint": {"type": "default"}}, {"text": fresh_tail}])
    assert second.metrics.latest_agent_invocation.usage.get("cacheReadInputTokens", 0) > 0, (
        "second turn rewrote the prefix instead of reading it"
    )


def test_a_leading_cache_point_is_accepted(quiet_strands_logging):
    """A cache point with nothing ahead of it is replaced by automatic placement.

    Bedrock rejects the shape outright with "There is nothing available to cache", so the point has to
    be dropped and re-placed for the request to survive at all.
    """
    model = BedrockModel(model_id=CACHING_MODEL_ID, cache_config=CacheConfig(strategy="auto"))
    agent = Agent(model=model, load_tools_from_directory=False, callback_handler=None)

    result = agent([{"cachePoint": {"type": "default"}}, {"text": durable_prefix()}])

    assert result.metrics.latest_agent_invocation.usage.get("cacheWriteInputTokens", 0) > 0


@pytest.mark.parametrize("configured_ttl", [None, "5m"])
def test_a_caller_ttl_does_not_conflict_with_the_tools_ttl(quiet_strands_logging, configured_ttl):
    """A caller's TTL is normalized so it cannot invert the TTL order Bedrock requires.

    Bedrock reads cache points in toolConfig, system, messages order and rejects a longer TTL that
    follows a shorter one, so a caller's ``1h`` behind tools at ``5m`` is what has to be neutralized.
    Both configurations reach the API: the caller's TTL is dropped when none is configured and replaced
    by the configured one otherwise. A tool cache point only reaches the request when a tool is
    registered, so the conflicting order this guards against needs one.
    """

    @tool
    def current_time() -> str:
        """Get the current time."""
        return "12:00"

    model = BedrockModel(
        model_id=CACHING_MODEL_ID,
        cache_config=CacheConfig(strategy="auto", ttl=configured_ttl),
        cache_tools=CacheToolsConfig(ttl="5m"),
    )
    agent = Agent(model=model, tools=[current_time], load_tools_from_directory=False, callback_handler=None)

    result = agent(
        [
            {"text": durable_prefix()},
            {"cachePoint": {"type": "default", "ttl": "1h"}},
            {"text": "Reply OK."},
        ]
    )

    assert result.stop_reason == "end_turn"


@pytest.mark.parametrize("document_format", ["csv", "pdf"])
def test_a_cache_point_after_a_document_is_accepted(quiet_strands_logging, letter_pdf, document_format):
    """Bedrock refuses a cache point directly after a non-PDF document but allows one after a PDF.

    The point steps back over an adjacent non-PDF document and stays put after a PDF, so both shapes
    have to reach the API intact.
    """
    sources = {"csv": b"a,b\n1,2\n", "pdf": letter_pdf}
    document: ContentBlock = {
        "document": {
            "format": document_format,
            "name": "attachment",
            "source": {"bytes": sources[document_format]},
        }
    }
    model = BedrockModel(model_id=CACHING_MODEL_ID, cache_config=CacheConfig(strategy="auto"))
    agent = Agent(model=model, load_tools_from_directory=False, callback_handler=None)

    result = agent([{"text": durable_prefix()}, document, {"cachePoint": {"type": "default"}}, {"text": "Summarize."}])

    assert result.stop_reason == "end_turn"
