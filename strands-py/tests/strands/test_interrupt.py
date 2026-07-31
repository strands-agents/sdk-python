import pytest

from strands.interrupt import AGENT_STREAM_INTERRUPT_ID_PREFIX, Interrupt, _InterruptState


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

    tru_context = interrupt_state.context
    exp_context = {"responses": prompt}
    assert tru_context == exp_context


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


# ============================================================================
# Version Tracking Tests
# ============================================================================


def test_interrupt_state_version_is_zero_after_initialization():
    """Test that _get_version() returns 0 after initialization."""
    interrupt_state = _InterruptState()
    assert interrupt_state._get_version() == 0


def test_interrupt_state_version_increments_after_activate():
    """Test that _get_version() increments after activate() is called."""
    interrupt_state = _InterruptState()
    assert interrupt_state._get_version() == 0

    interrupt_state.activate()
    assert interrupt_state._get_version() == 1


def test_interrupt_state_version_increments_after_deactivate():
    """Test that _get_version() increments after deactivate() is called."""
    interrupt_state = _InterruptState(activated=True)
    initial_version = interrupt_state._get_version()

    interrupt_state.deactivate()
    assert interrupt_state._get_version() == initial_version + 1


def test_interrupt_state_version_increments_after_resume():
    """Test that _get_version() increments after resume() is called."""
    interrupt_state = _InterruptState(
        interrupts={"test_id": Interrupt(id="test_id", name="test_name", reason="test reason")},
        activated=True,
    )
    initial_version = interrupt_state._get_version()

    prompt = [{"interruptResponse": {"interruptId": "test_id", "response": "test response"}}]
    interrupt_state.resume(prompt)
    assert interrupt_state._get_version() == initial_version + 1


def test_interrupt_state_version_increments_independently():
    """Test that version increments independently for each operation."""
    interrupt_state = _InterruptState()
    assert interrupt_state._get_version() == 0

    interrupt_state.activate()
    assert interrupt_state._get_version() == 1

    interrupt_state.deactivate()
    assert interrupt_state._get_version() == 2


def test_interrupt_state_version_not_in_to_dict():
    """Test that _version is not included in to_dict() output."""
    interrupt_state = _InterruptState()
    interrupt_state.activate()

    data = interrupt_state.to_dict()
    assert "_version" not in data
    assert "version" not in data


def test_interrupt_state_complete_tool_resume():
    """Answered agent-stream interrupts outlive a tool cycle; everything else is cleared."""
    answered_gate = Interrupt(id=f"{AGENT_STREAM_INTERRUPT_ID_PREFIX}answered", name="gate", response="approved")
    unanswered_gate = Interrupt(id=f"{AGENT_STREAM_INTERRUPT_ID_PREFIX}unanswered", name="gate2")
    tool_interrupt = Interrupt(id="v1:tool_call:t1:abc", name="tool_gate", response="approved")
    interrupt_state = _InterruptState(
        interrupts={
            answered_gate.id: answered_gate,
            unanswered_gate.id: unanswered_gate,
            tool_interrupt.id: tool_interrupt,
        },
        context={"tool_use_message": {"role": "assistant"}, "tool_results": []},
        activated=True,
    )

    interrupt_state.complete_tool_resume()

    tru_interrupts = interrupt_state.interrupts
    exp_interrupts = {answered_gate.id: answered_gate}
    assert tru_interrupts == exp_interrupts
    assert interrupt_state.context == {}
    assert not interrupt_state.activated


def test_interrupt_state_clear_agent_stream_interrupts():
    """Agent-stream interrupts are dropped; tool interrupts and context are untouched."""
    answered_gate = Interrupt(id=f"{AGENT_STREAM_INTERRUPT_ID_PREFIX}answered", name="gate", response="approved")
    tool_interrupt = Interrupt(id="v1:tool_call:t1:abc", name="tool_gate")
    interrupt_state = _InterruptState(
        interrupts={answered_gate.id: answered_gate, tool_interrupt.id: tool_interrupt},
        context={"tool_use_message": {"role": "assistant"}},
        activated=True,
    )

    interrupt_state.clear_agent_stream_interrupts()

    tru_interrupts = interrupt_state.interrupts
    exp_interrupts = {tool_interrupt.id: tool_interrupt}
    assert tru_interrupts == exp_interrupts
    assert interrupt_state.context == {"tool_use_message": {"role": "assistant"}}
    assert interrupt_state.activated


def test_interrupt_state_clear_agent_stream_interrupts_no_agent_stream_interrupts():
    """With nothing to drop the state is left alone, version included."""
    tool_interrupt = Interrupt(id="v1:tool_call:t1:abc", name="tool_gate")
    interrupt_state = _InterruptState(interrupts={tool_interrupt.id: tool_interrupt})
    version = interrupt_state._get_version()

    interrupt_state.clear_agent_stream_interrupts()

    assert interrupt_state.interrupts == {tool_interrupt.id: tool_interrupt}
    assert interrupt_state._get_version() == version


def test_interrupt_state_version_increments_after_complete_tool_resume():
    interrupt_state = _InterruptState(activated=True)
    version = interrupt_state._get_version()

    interrupt_state.complete_tool_resume()

    assert interrupt_state._get_version() > version
