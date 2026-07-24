"""Tests for documentation search indexing."""

import pytest

from strands_mcp_server.utils.indexer import Doc, IndexSearch


@pytest.mark.parametrize(
    ("indexed_title", "query"),
    [
        ("Streaming guide", "stream"),
        ("Stream guide", "streaming"),
        ("Guardrails guide", "guardrail"),
        ("Guardrail guide", "guardrails"),
        ("Running agents", "run"),
        ("Run agents", "running"),
        ("Policies", "policy"),
        ("Policy", "policies"),
        ("Processed events", "process"),
        ("Process events", "processed"),
        ("Boxes", "box"),
        ("Box", "boxes"),
        ("Classes", "class"),
        ("Class", "classes"),
    ],
)
def test_search_matches_common_word_forms(indexed_title: str, query: str) -> None:
    """Search matches common inflections in either direction (#3322)."""
    index = IndexSearch()
    doc = Doc(uri="https://strandsagents.com/guide", display_title=indexed_title, content="", index_title=indexed_title)
    index.add(doc)

    tru_results = index.search(query)
    tru_docs = [result_doc for _, result_doc in tru_results]

    assert tru_docs == [doc]
    assert tru_results[0][0] > 0
