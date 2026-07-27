"""Tests ensuring IndexSearch stores one posting per token and document.

Regression coverage for https://github.com/strands-agents/harness-sdk/issues/3325.
"""

from strands_mcp_server.utils.indexer import Doc, IndexSearch


def test_add_deduplicates_postings_per_document():
    index = IndexSearch()
    index.add(Doc(uri="first", display_title="First", content="agent agent", index_title="agent"))
    index.add(Doc(uri="second", display_title="Second", content="agent", index_title=""))

    exp_postings = [0, 1]
    assert index.doc_indices["agent"] == exp_postings
    assert index.doc_frequency["agent"] == 2


def test_search_scores_each_posting_once():
    index = IndexSearch()
    index.add(Doc(uri="doc", display_title="Doc", content="agent agent agent", index_title=""))

    tru_results = index.search("agent")

    assert len(tru_results) == 1
    assert tru_results[0][0] == 3.0
