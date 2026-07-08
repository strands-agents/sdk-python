"""Tests that the deprecated ``LocalMemoryStore`` aliases stay backwards compatible.

``LocalMemoryStore`` (and its ``*Config`` / ``*AddResult`` siblings) was renamed to
``JsonMemoryStore``. The old names must keep resolving to the exact same objects while emitting a
``DeprecationWarning``, so existing imports work unchanged.
"""

from __future__ import annotations

import pytest

from strands.vended_memory_stores.local import (
    JsonMemoryAddResult,
    JsonMemoryStore,
    JsonMemoryStoreConfig,
)

_ALIASES = {
    "LocalMemoryStore": JsonMemoryStore,
    "LocalMemoryStoreConfig": JsonMemoryStoreConfig,
    "LocalMemoryAddResult": JsonMemoryAddResult,
}


@pytest.mark.parametrize("old_name", list(_ALIASES))
def test_subpackage_alias_returns_new_symbol_with_warning(old_name, captured_warnings):
    import strands.vended_memory_stores.local as local_module

    resolved = getattr(local_module, old_name)

    assert resolved is _ALIASES[old_name]
    assert len(captured_warnings) == 1
    assert issubclass(captured_warnings[0].category, DeprecationWarning)
    assert old_name in str(captured_warnings[0].message)
    assert _ALIASES[old_name].__name__ in str(captured_warnings[0].message)


def test_parent_package_alias_returns_new_store_with_warning(captured_warnings):
    import strands.vended_memory_stores as vended_memory_stores

    resolved = vended_memory_stores.LocalMemoryStore

    assert resolved is JsonMemoryStore
    assert len(captured_warnings) == 1
    assert issubclass(captured_warnings[0].category, DeprecationWarning)
    assert "LocalMemoryStore" in str(captured_warnings[0].message)
    assert "JsonMemoryStore" in str(captured_warnings[0].message)


def test_new_names_do_not_warn(captured_warnings):
    import strands.vended_memory_stores as vended_memory_stores
    import strands.vended_memory_stores.local as local_module

    assert vended_memory_stores.JsonMemoryStore is JsonMemoryStore
    assert local_module.JsonMemoryStore is JsonMemoryStore
    assert local_module.JsonMemoryStoreConfig is JsonMemoryStoreConfig
    assert local_module.JsonMemoryAddResult is JsonMemoryAddResult
    assert len(captured_warnings) == 0


def test_unknown_subpackage_attribute_raises():
    import strands.vended_memory_stores.local as local_module

    with pytest.raises(AttributeError):
        _ = local_module.NoSuchSymbol
