from strands import Agent
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
