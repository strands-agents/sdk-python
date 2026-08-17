"""End-to-end integration tests for classifier-driven model routing."""

import logging
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from typing_extensions import override

from strands import Agent
from strands.models import BedrockModel, InputComplexityStrategy, ModelRouter
from strands.types.exceptions import ModelThrottledException
from strands.types.streaming import StreamEvent
from tests_integ.conftest import retry_on_flaky

_HAIKU_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_SONNET_MODEL_ID = "global.anthropic.claude-sonnet-4-6"


class _InvocationTrackingBedrockModel(BedrockModel):
    """Bedrock model that appends its model ID whenever it is invoked."""

    def __init__(self, model_id: str, invoked_model_ids: list[str]) -> None:
        super().__init__(model_id=model_id, max_tokens=256, streaming=False)
        self._tracked_model_id = model_id
        self._invoked_model_ids = invoked_model_ids

    @override
    async def stream(self, *args: Any, **kwargs: Any) -> AsyncGenerator[StreamEvent, None]:
        self._invoked_model_ids.append(self._tracked_model_id)
        async for event in super().stream(*args, **kwargs):
            yield event


@retry_on_flaky(
    "Bedrock throttling can be transient",
    max_attempts=2,
    retry_on=[ModelThrottledException],
)
@pytest.mark.parametrize(
    ("user_prompt", "expected_model_id", "candidate_model_ids"),
    [
        (
            "What is the capital of France? Reply with only the city name.",
            _HAIKU_MODEL_ID,
            (_SONNET_MODEL_ID, _HAIKU_MODEL_ID),
        ),
        (
            "Design a backward-compatible migration from region-local to globally unique idempotency keys for an "
            "active-active payment service. Account for concurrent requests, mixed-version deployments, rollback, "
            "data reconciliation, and monitoring. Return exactly five numbered steps, each under twelve words.",
            _SONNET_MODEL_ID,
            (_HAIKU_MODEL_ID, _SONNET_MODEL_ID),
        ),
    ],
    ids=["factual-request-selects-haiku", "distributed-systems-request-selects-sonnet"],
)
def test_agent_invokes_only_expected_model_for_request_complexity(
    caplog, user_prompt, expected_model_id, candidate_model_ids
):
    """The default classifier invokes exactly the appropriate candidate, independent of declaration order."""
    invoked_candidate_model_ids: list[str] = []
    candidate_models = [
        _InvocationTrackingBedrockModel(model_id, invoked_candidate_model_ids) for model_id in candidate_model_ids
    ]
    router = ModelRouter(
        models=candidate_models,
        strategy=InputComplexityStrategy(),
    )
    agent = Agent(model=router, load_tools_from_directory=False)

    with caplog.at_level(logging.WARNING, logger="strands.models.routing.input_complexity_strategy"):
        result = agent(user_prompt)

    assert result.stop_reason == "end_turn"
    assert str(result).strip()
    assert invoked_candidate_model_ids == [expected_model_id], (
        "Live model routing was inconclusive: expected exactly one invocation of the selected candidate"
    )
    assert not any("classification failed" in record.getMessage() for record in caplog.records), (
        "Live model routing was inconclusive: classification degraded"
    )
