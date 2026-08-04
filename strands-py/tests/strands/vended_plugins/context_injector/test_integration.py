"""End-to-end tests for ContextInjector against a real Agent + event loop.

These verify the ephemerality contract: the injected text reaches the model for one call but
never enters the agent's durable conversation history.
"""

import pytest

from strands import Agent
from strands.vended_plugins.context_injector import ContextInjector
from tests.fixtures.mocked_model_provider import MockedModelProvider


def _texts(messages):
    """Flatten all text blocks across a conversation into one string."""
    return " ".join(block["text"] for message in messages for block in message["content"] if "text" in block)


@pytest.fixture
def model():
    return MockedModelProvider([{"role": "assistant", "content": [{"text": "Hello!"}]}])


@pytest.fixture
def model_seen(model):
    """Capture the messages the model actually receives, wrapping the mock's stream()."""
    seen: list[list] = []
    original_stream = model.stream

    def stream(messages, *args, **kwargs):
        seen.append([dict(message) for message in messages])
        return original_stream(messages, *args, **kwargs)

    model.stream = stream
    return seen


def test_injected_text_reaches_model_but_not_durable_history(model, model_seen):
    agent = Agent(
        model=model,
        callback_handler=None,
        plugins=[ContextInjector(lambda context: "<ctx>EPHEMERAL</ctx>")],
    )

    agent("what is the weather?")

    # (a) The model saw the injected block folded into the user message.
    assert len(model_seen) == 1
    assert "<ctx>EPHEMERAL</ctx>" in _texts(model_seen[0])
    assert "what is the weather?" in _texts(model_seen[0])

    # (b) The durable conversation never contains the injected text.
    assert "<ctx>EPHEMERAL</ctx>" not in _texts(agent.messages)
    assert "what is the weather?" in _texts(agent.messages)


def test_injected_text_is_marked_ephemeral_for_the_model():
    """The model receives the injected text behind a cache-point boundary, every call.

    The boundary is what keeps per-call content out of a provider's cached prefix, so it must be
    present on each call and must not accumulate across calls.
    """
    model = MockedModelProvider(
        [
            {"role": "assistant", "content": [{"text": "first"}]},
            {"role": "assistant", "content": [{"text": "second"}]},
        ]
    )
    seen: list[list] = []
    original_stream = model.stream

    def stream(messages, *args, **kwargs):
        seen.append([dict(message) for message in messages])
        return original_stream(messages, *args, **kwargs)

    model.stream = stream

    agent = Agent(
        model=model,
        callback_handler=None,
        plugins=[ContextInjector(lambda context: "<ctx>EPHEMERAL</ctx>")],
    )
    agent("first ask")
    agent("second ask")

    for call in seen:
        last_content = call[-1]["content"]
        assert last_content[-2] == {"cachePoint": {"type": "default"}}
        assert last_content[-1] == {"text": "<ctx>EPHEMERAL</ctx>"}
        # Exactly one boundary per call, so the provider's cache-point budget is not exhausted.
        assert sum(1 for msg in call for block in msg["content"] if "cachePoint" in block) == 1
