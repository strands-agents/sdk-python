"""Tests for the ``strands.vended_memory_stores`` lazy re-export barrel."""

import pytest

import strands.vended_memory_stores as vended_memory_stores
from strands.vended_memory_stores.bedrock_knowledge_base import BedrockKnowledgeBaseStore


def test_reexports_the_store_from_the_top_level():
    """The convenience import resolves to the same object as the deep import."""
    assert vended_memory_stores.BedrockKnowledgeBaseStore is BedrockKnowledgeBaseStore


def test_lists_its_lazy_exports_in_all():
    """Every name in ``__all__`` is resolvable via the lazy ``__getattr__``."""
    for name in vended_memory_stores.__all__:
        assert getattr(vended_memory_stores, name) is not None


def test_raises_attribute_error_for_unknown_name():
    """An unknown attribute raises ``AttributeError`` rather than masking the typo."""
    with pytest.raises(AttributeError, match="cannot import name 'Nope'"):
        _ = vended_memory_stores.Nope
