import unittest.mock

import pytest

from strands.hooks import Interrupt, InterruptException


@pytest.fixture
def interrupt():
    return Interrupt(
        name="test",
        event_name="test_event",
        reasons=[],
    )


@pytest.fixture
def agent():
    instance = unittest.mock.Mock()
    instance._interrupts = {}
    return instance


def test_interrupt__call__(interrupt):
    with pytest.raises(InterruptException) as exception:
        interrupt("test reason")

    tru_interrupt = exception.value.interrupt
    exp_interrupt = Interrupt(
        name="test",
        event_name="test_event",
        reasons=["test reason"],
        activated=True,
    )
    assert tru_interrupt == exp_interrupt


def test_interrupt__call__with_response(interrupt):
    interrupt.activated = True
    interrupt.response = "test response"

    tru_response = interrupt("test reason")
    exp_response = "test response"

    assert tru_response == exp_response
    assert not interrupt.activated


@pytest.mark.parametrize(
    ("reasons", "exp_content"),
    [
        (
            ["test reason"],
            [{"json": {"interrupt": {"name": "test", "event_name": "test_event", "reasons": ["test reason"]}}}],
        ),
        ([], []),
    ],
)
def test_interrupt_to_tool_result_content(reasons, exp_content, interrupt):
    interrupt.reasons = reasons

    tru_content = interrupt.to_tool_result_content()
    assert tru_content == exp_content


def test_interrupt_from_agent(agent):
    exp_interrupt = Interrupt(name="test", event_name="test_event", reasons=["test reason"], response="test response")
    agent._interrupts = {("test", "test_event"): exp_interrupt}

    tru_interrupt = Interrupt.from_agent("test", "test_event", agent)
    assert tru_interrupt == exp_interrupt


def test_interrupt_from_agent_empty(agent):
    tru_interrupt = Interrupt.from_agent("test", "test_event", agent)
    exp_interrupt = Interrupt(
        name="test",
        event_name="test_event",
        reasons=[],
    )
    assert tru_interrupt == exp_interrupt
