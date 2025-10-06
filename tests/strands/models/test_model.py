import pytest
from pydantic import BaseModel

from strands.models import Model as SAModel


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
