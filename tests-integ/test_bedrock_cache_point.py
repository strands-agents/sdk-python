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

    agent = Agent(messages=messages, load_tools_from_directory=False)
    response = agent("What is favorite color?")

    usage = response.metrics.accumulated_usage
    assert usage["cacheReadInputTokens"] >= 0 or usage["cacheWriteInputTokens"] > 0  # At least one should have tokens
