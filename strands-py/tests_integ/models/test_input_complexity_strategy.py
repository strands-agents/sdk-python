"""End-to-end integration tests for classifier-driven model routing."""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from typing_extensions import override

from strands import Agent
from strands.models import BedrockModel, InputComplexityStrategy, ModelRouter, RoutingCandidate
from strands.types.content import Messages, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec

_CLASSIFIER_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_ROUTINE_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_COMPLEX_MODEL_ID = "global.anthropic.claude-sonnet-4-6"


class _InvocationTrackingBedrockModel(BedrockModel):
    """Bedrock model that records whether the router invoked it."""

    def __init__(self, model_id: str) -> None:
        super().__init__(model_id=model_id, max_tokens=256, streaming=False)
        self.invocation_count = 0

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        self.invocation_count += 1
        async for event in super().stream(
            messages,
            tool_specs,
            system_prompt,
            tool_choice=tool_choice,
            system_prompt_content=system_prompt_content,
            **kwargs,
        ):
            yield event


@pytest.mark.parametrize(
    ("user_prompt", "expected_candidate_name"),
    [
        (
            "What is the capital of France? Reply with only the city name.",
            "routine",
        ),
        (
            "Design a backward-compatible migration from region-local to globally unique idempotency keys for an "
            "active-active payment service. Account for concurrent requests, mixed-version deployments, rollback, "
            "data reconciliation, and monitoring. Return exactly five numbered steps, each under twelve words.",
            "complex",
        ),
    ],
    ids=["routine-factual-request", "complex-distributed-systems-request"],
)
def test_agent_routes_to_expected_model_for_request_complexity(
    tmp_path,
    caplog,
    user_prompt,
    expected_candidate_name,
):
    """A real classifier routes routine and complex requests to their expected real models."""
    catalog_path = tmp_path / "model-routing.json"
    catalog_path.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "routine": {
                        "description": "Use for direct factual requests that need no tradeoff analysis or planning.",
                        "relative_latency": "low",
                        "capabilities": ["short_factual_answers"],
                        "limitations": ["systems_architecture", "multi_constraint_planning"],
                    },
                    "complex": {
                        "description": "Use for systems architecture and multi-constraint planning with tradeoffs.",
                        "relative_latency": "high",
                        "capabilities": ["systems_architecture", "multi_constraint_planning"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    classifier_model = BedrockModel(
        model_id=_CLASSIFIER_MODEL_ID,
        max_tokens=64,
        streaming=False,
    )
    candidate_models = {
        "routine": _InvocationTrackingBedrockModel(_ROUTINE_MODEL_ID),
        "complex": _InvocationTrackingBedrockModel(_COMPLEX_MODEL_ID),
    }
    router = ModelRouter(
        models=[
            RoutingCandidate(model=candidate_models["routine"], name="routine"),
            RoutingCandidate(model=candidate_models["complex"], name="complex"),
        ],
        strategy=InputComplexityStrategy(
            classifier_model=classifier_model,
            model_catalog_path=catalog_path,
        ),
    )
    agent = Agent(model=router, load_tools_from_directory=False)

    with caplog.at_level(logging.WARNING, logger="strands.models.routing.input_complexity_strategy"):
        result = agent(user_prompt)

    assert result.stop_reason == "end_turn"
    assert str(result).strip()
    assert candidate_models[expected_candidate_name].invocation_count > 0
    assert all(
        model.invocation_count == 0
        for candidate_name, model in candidate_models.items()
        if candidate_name != expected_candidate_name
    )
    assert not any("classification failed" in record.getMessage() for record in caplog.records)
