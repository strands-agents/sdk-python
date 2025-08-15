import pytest

from strands.agent import identifier


def test_validate():
    tru_id = identifier.validate("abc")
    exp_id = "abc"
    assert tru_id == exp_id


@pytest.mark.parametrize(
    "agent_id",
    [
        "a/../b",
        "a/b",
    ],
)
def test_validate_invalid(agent_id):
    with pytest.raises(ValueError):
        identifier.validate(agent_id)
