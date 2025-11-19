import pytest

from strands.interrupt import Interrupt, _InterruptState


@pytest.fixture
def interrupt():
    return Interrupt(
        id="test_id:test_name",
        name="test_name",
        reason={"reason": "test"},
        response={"response": "test"},
    )


def test_interrupt_to_dict(interrupt):
    tru_dict = interrupt.to_dict()
    exp_dict = {
        "id": "test_id:test_name",
        "name": "test_name",
        "reason": {"reason": "test"},
        "response": {"response": "test"},
    }
    assert tru_dict == exp_dict


def test_interrupt_state_activate():
    interrupt_state = _InterruptState()

    interrupt_state.activate()
    assert interrupt_state.activated


def test_interrupt_state_deactivate():
    interrupt_state = _InterruptState(context={"test": "context"}, activated=True)

    interrupt_state.deactivate()

    assert not interrupt_state.activated

    tru_context = interrupt_state.context
    exp_context = {}
    assert tru_context == exp_context


def test_interrupt_state_to_dict():
    interrupt_state = _InterruptState(
        interrupts={"test_id": Interrupt(id="test_id", name="test_name", reason="test reason")},
        context={"test": "context"},
        activated=True,
    )

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

    tru_state = _InterruptState.from_dict(data)
    exp_state = _InterruptState(
        interrupts={"test_id": Interrupt(id="test_id", name="test_name", reason="test reason")},
        context={"test": "context"},
        activated=True,
    )
    assert tru_state == exp_state


def test_interrupt_state_resume():
    interrupt_state = _InterruptState(
        interrupts={"test_id": Interrupt(id="test_id", name="test_name", reason="test reason")},
        activated=True,
    )

    prompt = [
        {
            "interruptResponse": {
                "interruptId": "test_id",
                "response": "test response",
            }
        }
    ]
    interrupt_state.resume(prompt)

    tru_response = interrupt_state.interrupts["test_id"].response
    exp_response = "test response"
    assert tru_response == exp_response


def test_interrupt_state_resumse_deactivated():
    interrupt_state = _InterruptState(activated=False)
    interrupt_state.resume([])


def test_interrupt_state_resume_invalid_prompt():
    interrupt_state = _InterruptState(activated=True)

    exp_message = r"prompt_type=<class 'str'> \| must resume from interrupt with list of interruptResponse's"
    with pytest.raises(TypeError, match=exp_message):
        interrupt_state.resume("invalid")


def test_interrupt_state_resume_invalid_content():
    interrupt_state = _InterruptState(activated=True)

    exp_message = r"content_types=<\['text'\]> \| must resume from interrupt with list of interruptResponse's"
    with pytest.raises(TypeError, match=exp_message):
        interrupt_state.resume([{"text": "invalid"}])


def test_interrupt_resume_invalid_id():
    interrupt_state = _InterruptState(activated=True)

    exp_message = r"interrupt_id=<invalid> \| no interrupt found"
    with pytest.raises(KeyError, match=exp_message):
        interrupt_state.resume([{"interruptResponse": {"interruptId": "invalid", "response": None}}])
