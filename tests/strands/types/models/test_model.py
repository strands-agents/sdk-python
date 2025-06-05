import pytest

from strands.types.models import Model as SAModel


class TestModel(SAModel):
    def update_config(self, **model_config):
        return model_config

    def get_config(self):
        return

    def format_request(self, messages, tool_specs, system_prompt):
        return {
            "messages": messages,
            "tool_specs": tool_specs,
            "system_prompt": system_prompt,
        }

    def format_chunk(self, event):
        return {"event": event}

    def stream(self, request):
        yield {"request": request}


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


def test_converse(model, messages, tool_specs, system_prompt):
    response = model.converse(messages, tool_specs, system_prompt)

    tru_events = list(response)
    exp_events = [
        {
            "event": {
                "request": {
                    "messages": messages,
                    "tool_specs": tool_specs,
                    "system_prompt": system_prompt,
                },
            },
        },
    ]
    assert tru_events == exp_events
