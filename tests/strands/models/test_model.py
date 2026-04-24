from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from strands.hooks.events import AfterInvocationEvent
from strands.models import Model as SAModel
from strands.models.model import _ModelPlugin


class Person(BaseModel):
    name: str
    age: int


class TestModel(SAModel):
    def update_config(self, **model_config):
        return model_config

    def get_config(self):
        return

    async def structured_output(self, output_model, prompt=None, system_prompt=None, **kwargs):
        yield {"output": output_model(name="test", age=20)}

    async def stream(self, messages, tool_specs=None, system_prompt=None):
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": f"Processed {len(messages)} messages"}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        yield {
            "metadata": {
                "usage": {"inputTokens": 10, "outputTokens": 15, "totalTokens": 25},
                "metrics": {"latencyMs": 100},
            }
        }


@pytest.fixture
def model():
    return TestModel()


@pytest.fixture
def messages():
    return [
        {
            "role": "user",
            "content": [{"text": "hello"}],
        },
    ]


@pytest.fixture
def tool_specs():
    return [
        {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"},
                    },
                    "required": ["input"],
                },
            },
        },
    ]


@pytest.fixture
def model_plugin():
    return _ModelPlugin()


@pytest.fixture
def system_prompt():
    return "s1"


@pytest.mark.asyncio
async def test_stream(model, messages, tool_specs, system_prompt, alist):
    response = model.stream(messages, tool_specs, system_prompt)

    tru_events = await alist(response)
    exp_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockDelta": {"delta": {"text": "Processed 1 messages"}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
        {
            "metadata": {
                "usage": {"inputTokens": 10, "outputTokens": 15, "totalTokens": 25},
                "metrics": {"latencyMs": 100},
            }
        },
    ]
    assert tru_events == exp_events


@pytest.mark.asyncio
async def test_structured_output(model, messages, system_prompt, alist):
    response = model.structured_output(Person, prompt=messages, system_prompt=system_prompt)
    events = await alist(response)

    tru_output = events[-1]["output"]
    exp_output = Person(name="test", age=20)
    assert tru_output == exp_output


@pytest.mark.asyncio
async def test_stream_without_tool_choice_parameter(messages, alist):
    """Test that model implementations without tool_choice parameter are still valid."""

    class LegacyModel(SAModel):
        def update_config(self, **model_config):
            return model_config

        def get_config(self):
            return

        async def structured_output(self, output_model, prompt=None, system_prompt=None, **kwargs):
            yield {"output": output_model(name="test", age=20)}

        async def stream(self, messages, tool_specs=None, system_prompt=None):
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockDelta": {"delta": {"text": "Legacy model works"}}}
            yield {"messageStop": {"stopReason": "end_turn"}}

    model = LegacyModel()
    response = model.stream(messages)
    events = await alist(response)

    assert len(events) == 3
    assert events[1]["contentBlockDelta"]["delta"]["text"] == "Legacy model works"


@pytest.mark.asyncio
async def test_stream_with_tool_choice_parameter(messages, tool_specs, system_prompt, alist):
    """Test that model can accept tool_choice parameter."""

    class ModernModel(SAModel):
        def update_config(self, **model_config):
            return model_config

        def get_config(self):
            return

        async def structured_output(self, output_model, prompt=None, system_prompt=None, **kwargs):
            yield {"output": output_model(name="test", age=20)}

        async def stream(self, messages, tool_specs=None, system_prompt=None, *, tool_choice=None, **kwargs):
            yield {"messageStart": {"role": "assistant"}}
            if tool_choice:
                yield {"contentBlockDelta": {"delta": {"text": f"Tool choice: {tool_choice}"}}}
            else:
                yield {"contentBlockDelta": {"delta": {"text": "No tool choice"}}}
            yield {"messageStop": {"stopReason": "end_turn"}}

    model = ModernModel()

    # Test with tool_choice="auto"
    response = model.stream(messages, tool_specs, system_prompt, tool_choice="auto")
    events = await alist(response)
    assert events[1]["contentBlockDelta"]["delta"]["text"] == "Tool choice: auto"

    # Test with tool_choice="any"
    response = model.stream(messages, tool_specs, system_prompt, tool_choice="any")
    events = await alist(response)
    assert events[1]["contentBlockDelta"]["delta"]["text"] == "Tool choice: any"

    # Test with tool_choice={"type": "tool", "name": "test_tool"}
    response = model.stream(messages, tool_specs, system_prompt, tool_choice={"tool": {"name": "SampleModel"}})
    events = await alist(response)
    assert events[1]["contentBlockDelta"]["delta"]["text"] == "Tool choice: {'tool': {'name': 'SampleModel'}}"

    # Test without tool_choice
    response = model.stream(messages, tool_specs, system_prompt)
    events = await alist(response)
    assert events[1]["contentBlockDelta"]["delta"]["text"] == "No tool choice"


def test_context_window_limit_from_dict_config():
    class DictConfigModel(SAModel):
        def update_config(self, **model_config):
            pass

        def get_config(self):
            return {"context_window_limit": 200_000}

        async def structured_output(self, output_model, prompt=None, system_prompt=None, **kwargs):
            yield {}

        async def stream(self, messages, tool_specs=None, system_prompt=None):
            yield {}

    assert DictConfigModel().context_window_limit == 200_000


def test_context_window_limit_none_when_not_configured(model):
    assert model.context_window_limit is None


def test_stateful_false(model):
    """Model.stateful defaults to False."""
    assert not model.stateful


def test_model_plugin_clears_messages_when_stateful(model_plugin):
    """Messages are cleared when model is stateful."""
    agent = MagicMock()
    agent.model.stateful = True
    agent._model_state = {"response_id": "resp_123"}
    agent.messages = [{"role": "user", "content": [{"text": "hello"}]}]

    event = AfterInvocationEvent(agent=agent, invocation_state={})
    model_plugin._on_after_invocation(event)

    assert agent.messages == []


def test_model_plugin_preserves_messages_when_not_stateful(model_plugin):
    """Messages are preserved when model is not stateful."""
    agent = MagicMock()
    agent.model.stateful = False
    agent._model_state = {}
    agent.messages = [{"role": "user", "content": [{"text": "hello"}]}]

    event = AfterInvocationEvent(agent=agent, invocation_state={})
    model_plugin._on_after_invocation(event)

    assert len(agent.messages) == 1


@pytest.mark.asyncio
async def test_count_tokens_empty_messages(model):
    assert await model.count_tokens(messages=[]) == 0


@pytest.mark.asyncio
async def test_count_tokens_system_prompt_only(model):
    result = await model.count_tokens(messages=[], system_prompt="You are a helpful assistant.")
    assert result == 6


@pytest.mark.asyncio
async def test_count_tokens_text_messages(model, messages):
    result = await model.count_tokens(messages=messages)
    assert result == 1  # "hello"


@pytest.mark.asyncio
async def test_count_tokens_with_tool_specs(model, messages, tool_specs):
    without_tools = await model.count_tokens(messages=messages)
    with_tools = await model.count_tokens(messages=messages, tool_specs=tool_specs)
    assert without_tools == 1  # "hello"
    assert with_tools == 49  # "hello" (1) + tool_spec (48)


@pytest.mark.asyncio
async def test_count_tokens_with_system_prompt(model, messages, system_prompt):
    without_prompt = await model.count_tokens(messages=messages)
    with_prompt = await model.count_tokens(messages=messages, system_prompt=system_prompt)
    assert without_prompt == 1  # "hello"
    assert with_prompt == 3  # "hello" (1) + "s1" (2)


@pytest.mark.asyncio
async def test_count_tokens_combined(model, messages, tool_specs, system_prompt):
    result = await model.count_tokens(messages=messages, tool_specs=tool_specs, system_prompt=system_prompt)
    assert result == 51  # "hello" (1) + tool_spec (48) + "s1" (2)


@pytest.mark.asyncio
async def test_count_tokens_tool_use_block(model):
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "123",
                        "name": "my_tool",
                        "input": {"query": "test"},
                    }
                }
            ],
        }
    ]
    result = await model.count_tokens(messages=messages)
    # name "my_tool" (2) + json.dumps(input) (6) = 8
    assert result == 8


@pytest.mark.asyncio
async def test_count_tokens_tool_result_block(model):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "123",
                        "content": [{"text": "tool output here"}],
                        "status": "success",
                    }
                }
            ],
        }
    ]
    result = await model.count_tokens(messages=messages)
    assert result == 3  # "tool output here"


@pytest.mark.asyncio
async def test_count_tokens_reasoning_block(model):
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "reasoningContent": {
                        "reasoningText": {
                            "text": "Let me think about this step by step.",
                        }
                    }
                }
            ],
        }
    ]
    result = await model.count_tokens(messages=messages)
    assert result == 9  # "Let me think about this step by step."


@pytest.mark.asyncio
async def test_count_tokens_skips_binary_content(model):
    messages = [
        {
            "role": "user",
            "content": [{"image": {"format": "png", "source": {"bytes": b"fake image data"}}}],
        }
    ]
    assert await model.count_tokens(messages=messages) == 0


@pytest.mark.asyncio
async def test_count_tokens_tool_result_with_bytes_only(model):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "123",
                        "content": [{"image": {"format": "png", "source": {"bytes": b"image data"}}}],
                        "status": "success",
                    }
                }
            ],
        }
    ]
    result = await model.count_tokens(messages=messages)
    assert result == 0


@pytest.mark.asyncio
async def test_count_tokens_tool_result_with_text_and_bytes(model):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "123",
                        "content": [
                            {"text": "Here is the screenshot"},
                            {"image": {"format": "png", "source": {"bytes": b"image data"}}},
                        ],
                        "status": "success",
                    }
                }
            ],
        }
    ]
    result = await model.count_tokens(messages=messages)
    assert result > 0


@pytest.mark.asyncio
async def test_count_tokens_guard_content_block(model):
    messages = [
        {
            "role": "assistant",
            "content": [{"guardContent": {"text": {"text": "This content was filtered by guardrails."}}}],
        }
    ]
    result = await model.count_tokens(messages=messages)
    assert result == 8  # "This content was filtered by guardrails."


@pytest.mark.asyncio
async def test_count_tokens_tool_use_with_bytes(model):
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "123",
                        "name": "my_tool",
                        "input": {"data": b"binary data"},
                    }
                }
            ],
        }
    ]
    result = await model.count_tokens(messages=messages)
    # Should still count the tool name even though input has non-serializable bytes
    assert result == 2  # "my_tool" name only


@pytest.mark.asyncio
async def test_count_tokens_non_serializable_tool_spec(model, messages):
    tool_specs = [
        {
            "name": "test",
            "description": "a tool",
            "inputSchema": {"json": {"default": b"bytes"}},
        }
    ]
    result = await model.count_tokens(messages=messages, tool_specs=tool_specs)
    # Should still count the message tokens even though tool spec fails
    assert result == 1  # "hello" only, tool spec skipped


@pytest.mark.asyncio
async def test_count_tokens_citations_block(model):
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "citationsContent": {
                        "content": [{"text": "According to the document, the answer is 42."}],
                        "citations": [],
                    }
                }
            ],
        }
    ]
    result = await model.count_tokens(messages=messages)
    assert result == 11  # "According to the document, the answer is 42."


@pytest.mark.asyncio
async def test_count_tokens_system_prompt_content(model):
    result = await model.count_tokens(
        messages=[],
        system_prompt_content=[{"text": "You are a helpful assistant."}],
    )
    assert result == 6  # "You are a helpful assistant."


@pytest.mark.asyncio
async def test_count_tokens_system_prompt_content_with_cache_point(model):
    result = await model.count_tokens(
        messages=[],
        system_prompt_content=[
            {"text": "You are a helpful assistant."},
            {"cachePoint": {"type": "default"}},
        ],
    )
    assert result == 6  # "You are a helpful assistant.", cachePoint adds 0


@pytest.mark.asyncio
async def test_count_tokens_system_prompt_content_takes_priority(model):
    content_only = await model.count_tokens(
        messages=[],
        system_prompt_content=[{"text": "Short."}],
    )
    # When both are provided, system_prompt_content wins — system_prompt is ignored
    both = await model.count_tokens(
        messages=[],
        system_prompt="This is a much longer system prompt that should have more tokens.",
        system_prompt_content=[{"text": "Short."}],
    )
    assert content_only == 2  # "Short."
    assert content_only == both


@pytest.mark.asyncio
async def test_count_tokens_all_inputs(model):
    messages = [
        {"role": "user", "content": [{"text": "hello world"}]},
        {"role": "assistant", "content": [{"text": "hi there"}]},
    ]
    result = await model.count_tokens(
        messages=messages,
        tool_specs=[{"name": "test", "description": "a test tool", "inputSchema": {"json": {}}}],
        system_prompt="Be helpful.",
        system_prompt_content=[{"text": "Additional system context."}],
    )
    # system_prompt_content (4) + "hello world" (2) + "hi there" (2) + tool_spec (23) = 31
    assert result == 31


def test__get_encoding_falls_back_without_tiktoken(monkeypatch):
    """Test that _get_encoding returns None and count_tokens falls back to heuristic."""
    import strands.models.model as model_module

    model_module._get_encoding.cache_clear()
    original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _block_tiktoken(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("No module named 'tiktoken'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _block_tiktoken)

    try:
        assert model_module._get_encoding() is None

        # _estimate_tokens_with_tiktoken should raise when tiktoken is unavailable
        with pytest.raises(ImportError):
            model_module._estimate_tokens_with_tiktoken(
                messages=[{"role": "user", "content": [{"text": "hello world!"}]}],
            )

        # _estimate_tokens_with_heuristic uses chars/4 for text
        result = model_module._estimate_tokens_with_heuristic(
            messages=[{"role": "user", "content": [{"text": "hello world!"}]}],
        )
        assert result == 3  # ceil(12 / 4)
    finally:
        model_module._get_encoding.cache_clear()


class TestHeuristicEstimation:
    """Tests for _estimate_tokens_with_heuristic."""

    def test_all_content_types(self):
        """One call covering text, toolUse, toolResult, reasoning, guard, citations, system prompt, tool specs."""
        from strands.models.model import _estimate_tokens_with_heuristic

        messages = [
            {"role": "user", "content": [{"text": "hello world!"}]},
            {
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": "1", "name": "my_tool", "input": {"q": "test"}}},
                    {"reasoningContent": {"reasoningText": {"text": "Let me think."}}},
                    {"guardContent": {"text": {"text": "Filtered."}}},
                    {"citationsContent": {"content": [{"text": "Citation."}]}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "1", "content": [{"text": "tool output here"}]}},
                ],
            },
        ]
        result = _estimate_tokens_with_heuristic(
            messages=messages,
            tool_specs=[{"name": "test", "description": "a tool"}],
            system_prompt="ignored",
            system_prompt_content=[{"text": "Be helpful."}],
        )
        assert result > 0

    def test_non_serializable_inputs(self):
        """Heuristic gracefully handles non-serializable tool input and tool specs."""
        from strands.models.model import _estimate_tokens_with_heuristic

        result = _estimate_tokens_with_heuristic(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"toolUse": {"toolUseId": "1", "name": "my_tool", "input": {"data": b"bytes"}}},
                    ],
                },
            ],
            tool_specs=[{"name": "t", "inputSchema": {"json": {"default": b"bytes"}}}],
        )
        assert result == 2  # only tool name counted: ceil(len("my_tool") / 4)

    @pytest.mark.asyncio
    async def test_model_falls_back_to_heuristic(self, monkeypatch, model):
        """Model.count_tokens falls back to heuristic when tiktoken unavailable."""
        import strands.models.model as model_module

        model_module._get_encoding.cache_clear()
        original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def _block_tiktoken(name, *args, **kwargs):
            if name == "tiktoken":
                raise ImportError("No module named 'tiktoken'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _block_tiktoken)

        try:
            result = await model.count_tokens(messages=[{"role": "user", "content": [{"text": "hello world!"}]}])
            assert result == 3  # ceil(12 / 4)
        finally:
            model_module._get_encoding.cache_clear()
