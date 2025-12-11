from strands import Agent
from strands.models import BedrockModel
from strands.types.content import Messages


def test_bedrock_cache_point():
    messages: Messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": "Some really long text!" * 1000  # Minimum token count for cachePoint is 1024 tokens
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


def test_bedrock_multi_prompt_and_duplicate_cache_point():
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
