from typing import Type

import pytest
from pydantic import BaseModel

from strands.types.models import Model as SAModel


class Person(BaseModel):
    name: str
    age: int


class TestModel(SAModel):
    def update_config(self, **model_config):
        return model_config

    def get_config(self):
        return

    def structured_output(self, output_model: Type[BaseModel]) -> BaseModel:
        return output_model(name="test", age=20)

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


def test_structured_output(model):
    response = model.structured_output(Person)

    assert response == Person(name="test", age=20)


def test_converse_logging(model, messages, tool_specs, system_prompt, caplog):
    """Test that converse method logs the formatted request at debug level."""
    import logging

    # Set the logger to debug level to capture debug messages
    caplog.set_level(logging.DEBUG, logger="strands.types.models.model")

    # Execute the converse method
    response = model.converse(messages, tool_specs, system_prompt)
    list(response)  # Consume the generator to trigger all logging

    # Check that the expected log messages are present
    assert "formatting request" in caplog.text
    assert "formatted request=" in caplog.text
    assert "invoking model" in caplog.text
    assert "got response from model" in caplog.text
    assert "finished streaming response from model" in caplog.text

    # Check that the formatted request is logged with the expected content
    expected_request_str = str(
        {
            "messages": messages,
            "tool_specs": tool_specs,
            "system_prompt": system_prompt,
        }
    )
    assert expected_request_str in caplog.text
