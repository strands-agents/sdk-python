"""End-to-end integration tests for classifier-driven model routing."""

import logging
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from typing_extensions import override

from strands import Agent
from strands.models import BedrockModel, InputComplexityStrategy, ModelCatalog, ModelRouter
from strands.types.content import Messages, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec

_CLASSIFIER_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_HAIKU_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_SONNET_MODEL_ID = "global.anthropic.claude-sonnet-4-6"


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
    ("user_prompt", "expected_model_id"),
    [
        (
            "What is the capital of France? Reply with only the city name.",
            _HAIKU_MODEL_ID,
        ),
        (
            "Design a backward-compatible migration from region-local to globally unique idempotency keys for an "
            "active-active payment service. Account for concurrent requests, mixed-version deployments, rollback, "
            "data reconciliation, and monitoring. Return exactly five numbered steps, each under twelve words.",
            _SONNET_MODEL_ID,
        ),
    ],
    ids=["factual-request-selects-haiku", "distributed-systems-request-selects-sonnet"],
)
def test_agent_routes_to_expected_model_for_request_complexity(caplog, user_prompt, expected_model_id):
    """A real classifier routes distinct request complexities to the expected real models."""
    classifier_model = BedrockModel(
        model_id=_CLASSIFIER_MODEL_ID,
        max_tokens=64,
        streaming=False,
        temperature=0,
    )
    candidate_models = {
        model_id: _InvocationTrackingBedrockModel(model_id) for model_id in (_HAIKU_MODEL_ID, _SONNET_MODEL_ID)
    }
    model_catalog = ModelCatalog(
        {
            model_id: {
                "litellm_provider": "bedrock",
                "mode": "chat",
                "max_input_tokens": 200_000,
                "supports_tool_calling": True,
            }
            for model_id in candidate_models
        }
    )
    router = ModelRouter(
        models=list(candidate_models.values()),
        strategy=InputComplexityStrategy(
            classifier_model=classifier_model,
            model_catalog=model_catalog,
        ),
    )
    agent = Agent(model=router, load_tools_from_directory=False)

    with caplog.at_level(logging.WARNING, logger="strands.models.routing.input_complexity_strategy"):
        result = agent(user_prompt)

    assert result.stop_reason == "end_turn"
    assert str(result).strip()
    assert candidate_models[expected_model_id].invocation_count > 0
    assert all(
        model.invocation_count == 0 for model_id, model in candidate_models.items() if model_id != expected_model_id
    )
    assert not any("classification failed" in record.getMessage() for record in caplog.records)
