"""Tests that the deprecated ``LocalMemoryStore`` aliases stay backwards compatible.

``LocalMemoryStore`` (and its ``*Config`` / ``*AddResult`` siblings) was renamed to
``TestMemoryStore``. The old names must keep resolving to the exact same objects while emitting a
``DeprecationWarning``, so existing imports work unchanged.
"""

from __future__ import annotations

import pytest

from strands.vended_memory_stores.local import (
    TestMemoryAddResult,
    TestMemoryStore,
    TestMemoryStoreConfig,
)

_ALIASES = {
    "LocalMemoryStore": TestMemoryStore,
    "LocalMemoryStoreConfig": TestMemoryStoreConfig,
    "LocalMemoryAddResult": TestMemoryAddResult,
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

    assert resolved is TestMemoryStore
    assert len(captured_warnings) == 1
    assert issubclass(captured_warnings[0].category, DeprecationWarning)
    assert "LocalMemoryStore" in str(captured_warnings[0].message)
    assert "TestMemoryStore" in str(captured_warnings[0].message)


def test_new_names_do_not_warn(captured_warnings):
    import strands.vended_memory_stores as vended_memory_stores
    import strands.vended_memory_stores.local as local_module

    assert vended_memory_stores.TestMemoryStore is TestMemoryStore
    assert local_module.TestMemoryStore is TestMemoryStore
    assert local_module.TestMemoryStoreConfig is TestMemoryStoreConfig
    assert local_module.TestMemoryAddResult is TestMemoryAddResult
    assert len(captured_warnings) == 0


def test_unknown_subpackage_attribute_raises():
    import strands.vended_memory_stores.local as local_module

    with pytest.raises(AttributeError):
        _ = local_module.NoSuchSymbol


def test_test_prefixed_classes_are_not_collected_by_pytest():
    # The ``Test`` prefix would otherwise make pytest try to collect these classes as test suites
    # (emitting a PytestCollectionWarning). ``__test__ = False`` opts them out; assert it stays set so
    # a future edit that drops a guard fails here instead of silently re-enabling collection.
    assert TestMemoryStore.__test__ is False
    assert TestMemoryStoreConfig.__test__ is False
    assert TestMemoryAddResult.__test__ is False
