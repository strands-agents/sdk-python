import pytest

from strands import _identifier


@pytest.mark.parametrize("type_", list(_identifier.Identifier))
def test_validate(type_):
    tru_id = _identifier.validate("abc", type_)
    exp_id = "abc"
    assert tru_id == exp_id


@pytest.mark.parametrize("type_", list(_identifier.Identifier))
def test_validate_invalid(type_):
    id_ = "a/../b"
    with pytest.raises(ValueError, match=f"{type_.value}={id_} | id cannot contain path separators"):
        _identifier.validate(id_, type_)
