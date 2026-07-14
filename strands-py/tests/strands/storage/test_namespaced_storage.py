"""Tests for the namespace() factory that scopes a storage under a key prefix."""

import pytest

from strands.storage import InMemoryStorage
from strands.storage.base import NAMESPACED, namespace


@pytest.fixture
def backend():
    return InMemoryStorage()


@pytest.fixture
def namespaced(backend):
    return namespace(backend, "prefix")


@pytest.mark.asyncio
async def test_prepends_namespace_to_write_keys(backend, namespaced):
    await namespaced.write("key", bytes([1, 2, 3]))
    assert await backend.read("prefix/key") == bytes([1, 2, 3])


@pytest.mark.asyncio
async def test_prepends_namespace_to_read_keys(backend, namespaced):
    await backend.write("prefix/key", bytes([4, 5]))
    assert await namespaced.read("key") == bytes([4, 5])


@pytest.mark.asyncio
async def test_returns_none_for_missing_keys(namespaced):
    assert await namespaced.read("nonexistent") is None


@pytest.mark.asyncio
async def test_prepends_namespace_to_delete_keys(backend, namespaced):
    await backend.write("prefix/key", bytes([1]))
    await namespaced.delete("key")
    assert await backend.read("prefix/key") is None


@pytest.mark.asyncio
async def test_lists_keys_with_namespace_stripped(backend, namespaced):
    await backend.write("prefix/a", bytes([1]))
    await backend.write("prefix/b", bytes([2]))
    await backend.write("other/c", bytes([3]))

    assert await namespaced.list("") == ["a", "b"]


@pytest.mark.asyncio
async def test_lists_keys_with_sub_prefix(backend, namespaced):
    await backend.write("prefix/session/abc", bytes([1]))
    await backend.write("prefix/session/def", bytes([2]))
    await backend.write("prefix/offloader/xyz", bytes([3]))

    assert await namespaced.list("session/") == ["session/abc", "session/def"]


@pytest.mark.asyncio
async def test_composes_nested_namespaces(backend):
    nested = namespace(namespace(backend, "prefix"), "sub")
    await nested.write("key", bytes([9]))
    assert await backend.read("prefix/sub/key") == bytes([9])


@pytest.mark.asyncio
async def test_composes_nested_namespaces_via_method(backend):
    nested = namespace(backend, "prefix").namespace("sub")
    await nested.write("key", bytes([9]))
    assert await backend.read("prefix/sub/key") == bytes([9])


@pytest.mark.asyncio
async def test_handles_empty_namespace_as_no_op_prefix(backend):
    empty = namespace(backend, "")
    await empty.write("key", bytes([7]))
    assert await backend.read("key") == bytes([7])


def test_sets_namespaced_marker_on_returned_view(namespaced):
    assert getattr(namespaced, NAMESPACED, False) is True


def test_does_not_set_namespaced_marker_on_raw_storage(backend):
    assert getattr(backend, NAMESPACED, False) is False
