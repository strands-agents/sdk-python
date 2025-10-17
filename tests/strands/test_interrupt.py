import pytest

from strands.interrupt import Interrupt


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
