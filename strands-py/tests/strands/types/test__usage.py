"""Tests for the internal helpers that read and combine ``Usage`` counts."""

import time
from unittest.mock import MagicMock

import pytest

from strands.types._usage import (
    accumulate_usage,
    as_token_count,
    coerce_usage_counters,
    context_token_count,
    prompt_token_count,
)
from strands.types.event_loop import Usage


def _billed_total(usage: Usage) -> int:
    """Sum the four disjoint billed counters, the way a cost calculation would."""
    return (
        usage["inputTokens"]
        + usage["outputTokens"]
        + usage.get("cacheReadInputTokens", 0)
        + usage.get("cacheWriteInputTokens", 0)
    )


def test_accumulating_cached_usage_preserves_the_invariant():
    """Every counter must be carried across, or totalTokens counts tokens nothing reports."""
    target = Usage(inputTokens=0, outputTokens=0, totalTokens=0)
    source = Usage(inputTokens=10, outputTokens=4, totalTokens=5862, cacheReadInputTokens=5848)

    accumulate_usage(target, source)

    assert target == source
    assert _billed_total(target) == target["totalTokens"]


def test_accumulating_many_usages_preserves_the_invariant():
    target = Usage(inputTokens=0, outputTokens=0, totalTokens=0)
    source = Usage(
        inputTokens=2,
        outputTokens=5,
        totalTokens=6457,
        cacheReadInputTokens=6000,
        cacheWriteInputTokens=450,
        reasoningOutputTokens=3,
    )

    for _ in range(3):
        accumulate_usage(target, source)

    exp_usage = Usage(
        inputTokens=6,
        outputTokens=15,
        totalTokens=19371,
        cacheReadInputTokens=18000,
        cacheWriteInputTokens=1350,
        reasoningOutputTokens=9,
    )

    assert target == exp_usage
    assert _billed_total(target) == target["totalTokens"]


def test_accumulating_partial_usage_treats_missing_counters_as_zero():
    """A multi-agent node may report usage assembled outside a model provider."""
    target = Usage(inputTokens=1, outputTokens=2, totalTokens=3)

    accumulate_usage(target, {})  # type: ignore[typeddict-item]

    assert target == Usage(inputTokens=1, outputTokens=2, totalTokens=3)


def test_accumulating_omits_optional_counters_the_source_never_reports():
    target = Usage(inputTokens=0, outputTokens=0, totalTokens=0)

    accumulate_usage(target, Usage(inputTokens=10, outputTokens=5, totalTokens=15))

    assert target == Usage(inputTokens=10, outputTokens=5, totalTokens=15)


@pytest.mark.parametrize(
    "usage, exp_count",
    [
        # Captured from live provider responses: Bedrock Converse, Bedrock Mantle Responses, Google.
        ({"inputTokens": 10, "outputTokens": 4, "totalTokens": 5862, "cacheReadInputTokens": 5848}, 5862),
        ({"inputTokens": 2, "outputTokens": 5, "totalTokens": 6457, "cacheReadInputTokens": 6450}, 6457),
        ({"inputTokens": 8, "outputTokens": 35, "totalTokens": 23327, "cacheReadInputTokens": 23284}, 23327),
        ({"inputTokens": 100, "outputTokens": 20, "totalTokens": 120}, 120),
        # A total that accounts for nothing must not shrink the estimate below the counters.
        ({"inputTokens": 100, "outputTokens": 20, "totalTokens": 0}, 120),
    ],
)
def test_context_token_count_counts_the_whole_prompt(usage, exp_count):
    """Understating the window skips compaction and overflows the model instead."""
    assert context_token_count(usage) == exp_count


@pytest.mark.parametrize(
    "usage, min_count",
    [
        # Recorded before cache tokens became separate counters, so inputTokens already contained
        # them. Reading high is the safe direction and self-corrects after the next model call.
        ({"inputTokens": 6452, "outputTokens": 10, "totalTokens": 6462, "cacheReadInputTokens": 6450}, 6462),
        ({"inputTokens": 5000, "outputTokens": 100, "totalTokens": 5100, "cacheReadInputTokens": 1000}, 6100),
    ],
)
def test_context_token_count_never_understates_usage_recorded_before_the_contract(usage, min_count):
    assert context_token_count(usage) >= min_count


@pytest.mark.parametrize(
    "usage, exp_count",
    [
        # A gateway reporting counts as strings must not concatenate them into a nonsense size.
        # Expected values match the TypeScript SDK's contextTokenCount exactly.
        ({"inputTokens": "10", "outputTokens": "5", "totalTokens": "15"}, 15),
        ({"inputTokens": 10, "outputTokens": 5, "totalTokens": 105, "cacheReadInputTokens": "90"}, 105),
        ({"inputTokens": "abc", "outputTokens": {}, "totalTokens": []}, 0),
    ],
)
def test_context_token_count_coerces_malformed_counts(usage, exp_count):
    tru_count = context_token_count(usage)

    assert tru_count == exp_count
    assert type(tru_count) is int
    assert type(prompt_token_count(usage)) is int


@pytest.mark.parametrize("count, exp_count", [(2**53 - 1, 2**53 - 1), (2**53, 0), (int("9" * 400), 0)])
def test_reading_a_count_larger_than_a_float_can_hold(count, exp_count):
    """JSON carries arbitrary-precision ints, so a count need not fit in a float.

    Coercing through a float would overflow rather than degrade, and the counts are read from
    payloads nothing validates: a model implementation's own usage, and persisted session state.
    Past what JavaScript represents exactly the count is read as absent, since one the TypeScript
    SDK cannot carry is not one this SDK should report either.
    """
    assert as_token_count(count) == exp_count
    assert coerce_usage_counters({"inputTokens": count, "outputTokens": 4, "totalTokens": 4}) == {
        "inputTokens": exp_count,
        "outputTokens": 4,
        "totalTokens": 4,
    }


def test_reading_counters_from_a_vendor_usage_object():
    """A model implementation may hand back its vendor's usage object rather than a mapping.

    Reading only mappings would zero every counter with nothing logged, so a cost cap would stop
    tripping and the context window would measure as empty.
    """

    class VendorUsage:
        inputTokens = 1000
        outputTokens = 500
        totalTokens = 1500

    assert coerce_usage_counters(VendorUsage()) == {
        "inputTokens": 1000,
        "outputTokens": 500,
        "totalTokens": 1500,
    }


@pytest.mark.parametrize("payload", ["not-a-mapping", 42, [1, 2, 3], None, True])
def test_reading_counters_from_a_payload_that_carries_none(payload):
    assert coerce_usage_counters(payload) == {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}


def test_accumulating_past_the_reported_count_ceiling_keeps_the_running_total():
    """A running total is a sum this SDK computed, not a count a provider claimed.

    Applying the ceiling meant for a reported count to an accumulated one would reset the total
    once it grew past that ceiling, silently discarding everything counted so far.
    """
    target = Usage(inputTokens=0, outputTokens=0, totalTokens=0)
    source = Usage(inputTokens=0, outputTokens=0, totalTokens=2**53 - 1)

    totals = []
    for _ in range(4):
        accumulate_usage(target, source)
        totals.append(target["totalTokens"])

    assert totals == [(2**53 - 1) * turn for turn in (1, 2, 3, 4)]


@pytest.mark.parametrize(
    "count, exp_count",
    [
        (7, 7),
        ("42", 42),
        ("12.5", 12),
        ("1e3", 1000),
        (-5, 0),
        ("abc", 0),
        # A count that is not a finite number reads as absent rather than poisoning every sum it
        # reaches, since one NaN makes an accumulated total NaN for the rest of the session.
        (float("nan"), 0),
        (float("inf"), 0),
        (float("-inf"), 0),
        # Forms only one language's own parser accepts, which each SDK rejects so the two agree:
        # JavaScript's Number() takes the prefixes and Python's float() takes the separator.
        ("0x10", 0),
        ("0o17", 0),
        ("0b101", 0),
        ("1_000", 0),
        # Separators str.strip() removes but JavaScript's trim() does not, and a non-ASCII digit
        # that \\d would match.
        ("\x1c12", 0),
        ("\x8512", 0),
        ("٤٢", 0),
    ],
)
def test_reading_a_count_agrees_with_the_typescript_sdk(count, exp_count):
    """Any divergence here silently changes reported cost for one language only.

    Expected values match the TypeScript SDK's ``asTokenCount`` exactly.
    """
    assert as_token_count(count) == exp_count


def test_reading_a_count_from_a_mock_standing_in_for_one():
    """A mock answers ``isinstance`` for the type it stands in for without supporting arithmetic.

    This is how a test doubles a model's usage, so it reaches the coercion in practice. A count only
    describes an invocation whose reply is already complete, so an unusable one reports zero rather
    than taking the reply with it.
    """
    assert as_token_count(MagicMock(spec=int)) == 0


def test_reading_a_long_digit_run_with_an_invalid_tail_is_linear():
    """A count arrives as characters a gateway chose, so the pattern must not backtrack on them.

    Splitting the integer and fraction alternatives lets the engine retry every split point, which
    costs time quadratic in the length. Elapsed time is the only observable, since the regex engine
    exposes no step count, but the threshold is generous: this reads in tens of milliseconds, while
    the shape it guards against takes seconds at a thirtieth of this length.
    """
    started = time.monotonic()

    assert as_token_count("0" * 500_000 + "x") == 0

    assert time.monotonic() - started < 5
