import pytest

from strands.agent.interrupt import InterruptState
from strands.interrupt import Interrupt


@pytest.fixture
def interrupt():
    return Interrupt(id="test_id", name="test_name", reason="test reason")


def test_interrupt_state__contains__(interrupt):
    interrupt_state = InterruptState(interrupts={"test_id": interrupt})

    assert "test_id" in interrupt_state


def test_interrupt_state__getitem__(interrupt):
    interrupt_state = InterruptState(interrupts={"test_id": interrupt})

    tru_interrupt = interrupt_state["test_id"]
    exp_interrupt = interrupt
    assert tru_interrupt == exp_interrupt


def test_interrupt_state__setitem__(interrupt):
    interrupt_state = InterruptState()
    interrupt_state["test_id"] = interrupt

    tru_interrupt = interrupt_state["test_id"]
    exp_interrupt = interrupt
    assert tru_interrupt == exp_interrupt


def test_interrupt_state_setdefault(interrupt):
    interrupt_state = InterruptState()

    tru_interrupt = interrupt_state.setdefault("test_id", interrupt)
    exp_interrupt = interrupt
    assert tru_interrupt == exp_interrupt

    tru_interrupt = interrupt_state.setdefault("test_id", Interrupt(id="dummy_id", name="dummy_name"))
    exp_interrupt = interrupt
    assert tru_interrupt == exp_interrupt


def test_interrupt_activate():
    interrupt_state = InterruptState()

    interrupt_state.activate(context={"test": "context"})

    assert interrupt_state.activated

    tru_context = interrupt_state.context
    exp_context = {"test": "context"}
    assert tru_context == exp_context


def test_interrupt_deactivate():
    interrupt_state = InterruptState(context={"test": "context"}, activated=True)

    interrupt_state.deactivate()

    assert not interrupt_state.activated

    tru_context = interrupt_state.context
    exp_context = {}
    assert tru_context == exp_context


def test_interrupt_state_to_dict(interrupt):
    interrupt_state = InterruptState(interrupts={"test_id": interrupt}, context={"test": "context"}, activated=True)

    tru_data = interrupt_state.to_dict()
    exp_data = {
        "interrupts": {"test_id": {"id": "test_id", "name": "test_name", "reason": "test reason", "response": None}},
        "context": {"test": "context"},
        "activated": True,
    }
    assert tru_data == exp_data


def test_interrupt_state_from_dict():
    data = {
        "interrupts": {"test_id": {"id": "test_id", "name": "test_name", "reason": "test reason", "response": None}},
        "context": {"test": "context"},
        "activated": True,
    }

    tru_state = InterruptState.from_dict(data)
    exp_state = InterruptState(
        interrupts={"test_id": Interrupt(id="test_id", name="test_name", reason="test reason")},
        context={"test": "context"},
        activated=True,
    )
    assert tru_state == exp_state
