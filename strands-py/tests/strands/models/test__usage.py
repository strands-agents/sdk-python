"""Tests for the shared token-usage normalization helper.

Every token count here comes from a live provider response or from that provider's own published
documentation, noted per case, so the tests encode what providers actually send rather than what
the SDK assumes they send.
"""

from types import SimpleNamespace

import pytest

from strands.models._usage import normalize_usage, token_detail
from strands.types.event_loop import Usage


def _billed_total(usage: Usage) -> int:
    """Sum the four disjoint billed counters, the way a cost calculation would."""
    return (
        usage["inputTokens"]
        + usage["outputTokens"]
        + usage.get("cacheReadInputTokens", 0)
        + usage.get("cacheWriteInputTokens", 0)
    )


def test_disjoint_provider_keeps_input_tokens_and_adds_cache_to_total():
    # Captured from Bedrock Converse, eu.anthropic.claude-haiku-4-5, eu-north-1: 10 + 4 + 5848
    # equals the 5862 the provider reported as its own total.
    tru_usage = normalize_usage(input_tokens=10, output_tokens=4, cache_write_tokens=5848, input_includes_cache=False)
    exp_usage = {"inputTokens": 10, "outputTokens": 4, "totalTokens": 5862, "cacheWriteInputTokens": 5848}

    assert tru_usage == exp_usage


def test_inclusive_provider_subtracts_cache_from_input_tokens():
    # The cache-hit example published in the Bedrock prompt-caching guide, under
    # "Cache Management for Models from OpenAI": 2048 + 256 == 2304, with 1920 of the 2048 cached.
    tru_usage = normalize_usage(input_tokens=2048, output_tokens=256, cache_read_tokens=1920, input_includes_cache=True)
    exp_usage = {"inputTokens": 128, "outputTokens": 256, "totalTokens": 2304, "cacheReadInputTokens": 1920}

    assert tru_usage == exp_usage


def test_cache_and_reasoning_keys_are_omitted_when_zero():
    tru_usage = normalize_usage(input_tokens=100, output_tokens=20, input_includes_cache=True)
    exp_usage = {"inputTokens": 100, "outputTokens": 20, "totalTokens": 120}

    assert tru_usage == exp_usage


def test_reasoning_tokens_are_reported_without_inflating_the_total():
    tru_usage = normalize_usage(input_tokens=100, output_tokens=500, reasoning_tokens=400, input_includes_cache=True)
    exp_usage = {
        "inputTokens": 100,
        "outputTokens": 500,
        "totalTokens": 600,
        "reasoningOutputTokens": 400,
    }

    assert tru_usage == exp_usage


def test_cache_exceeding_input_clamps_to_zero_rather_than_going_negative(caplog):
    tru_usage = normalize_usage(input_tokens=100, output_tokens=10, cache_read_tokens=500, input_includes_cache=True)
    exp_usage = {"inputTokens": 0, "outputTokens": 10, "totalTokens": 510, "cacheReadInputTokens": 500}

    assert tru_usage == exp_usage
    assert "clamping net input to zero" in caplog.text


@pytest.mark.parametrize(
    "value, exp_count",
    [
        # Values a provider or OpenAI-compatible gateway can realistically send. The expected
        # counts match the TypeScript SDK's Number()/Math.trunc coercion exactly.
        (7, 7),
        ("42", 42),
        ("12.5", 12),
        ("1e3", 1000),
        (float("inf"), 0),
        (float("-inf"), 0),
        (float("nan"), 0),
        (-5, 0),
        (True, 0),
        (None, 0),
        ("abc", 0),
        (object(), 0),
        # Forms only one language's own parser accepts, which each SDK rejects so the two agree:
        # float() takes the separator, the non-ASCII digits and the prefixes belong to Number().
        ("1_000", 0),
        ("0x10", 0),
        ("0o17", 0),
        ("0b101", 0),
        ("٤٢", 0),
        ("１２", 0),
    ],
)
def test_token_counts_are_coerced_consistently_with_the_typescript_sdk(value, exp_count):
    usage = normalize_usage(input_tokens=value, output_tokens=0, input_includes_cache=True)

    assert usage["inputTokens"] == exp_count


@pytest.mark.parametrize("value", [None, "not a number", object(), float("nan")])
def test_non_numeric_counts_are_treated_as_absent(value):
    tru_usage = normalize_usage(input_tokens=value, output_tokens=value, input_includes_cache=True)
    exp_usage = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}

    assert tru_usage == exp_usage


@pytest.mark.parametrize(
    "kwargs, input_includes_cache, provider_total",
    [
        # Bedrock Converse, eu.anthropic.claude-haiku-4-5, eu-north-1: cache write then read.
        # Converse reports the cache counters on top of inputTokens.
        ({"input_tokens": 10, "output_tokens": 4, "cache_write_tokens": 5848}, False, 5862),
        ({"input_tokens": 10, "output_tokens": 4, "cache_read_tokens": 5848}, False, 5862),
        # Bedrock Mantle Responses, openai.gpt-5.6-luna, us-east-1: cache write then read. Both
        # counters are subsets of input_tokens here.
        ({"input_tokens": 6452, "output_tokens": 5, "cache_write_tokens": 6450}, True, 6457),
        ({"input_tokens": 6452, "output_tokens": 5, "cache_read_tokens": 6450}, True, 6457),
    ],
)
def test_provider_payloads_satisfy_the_invariant(kwargs, input_includes_cache, provider_total):
    usage = normalize_usage(**kwargs, input_includes_cache=input_includes_cache)

    assert _billed_total(usage) == usage["totalTokens"] == provider_total


@pytest.mark.parametrize(
    "details",
    [
        # Vendor SDKs model token breakdowns as objects...
        SimpleNamespace(cached_tokens=80),
        # ...while OpenAI-compatible gateways returning raw JSON leave them as mappings.
        {"cached_tokens": 80},
    ],
)
def test_token_detail_reads_object_and_mapping_payloads(details):
    assert token_detail(details, "cached_tokens") == 80


@pytest.mark.parametrize("details", [None, {}, SimpleNamespace(), {"other": 1}, 0])
def test_token_detail_treats_absent_counts_as_zero(details):
    assert token_detail(details, "cached_tokens") == 0
