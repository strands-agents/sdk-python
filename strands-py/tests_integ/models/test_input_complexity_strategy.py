"""End-to-end integration tests for classifier-driven model routing."""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from typing_extensions import override

from strands import Agent
from strands.models import BedrockModel, InputComplexityStrategy, ModelRouter, RoutingCandidate
from strands.types.content import Messages, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec

_CLASSIFIER_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_ROUTINE_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_COMPLEX_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"


class _InvocationTrackingBedrockModel(BedrockModel):
    """Bedrock model that records whether the router invoked it."""

    def __init__(self, model_id: str) -> None:
        super().__init__(model_id=model_id, max_tokens=64, streaming=False)
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


def test_agent_routes_with_real_classifier_and_catalog(tmp_path, caplog):
    """A real classifier selects one real candidate through the public Agent API."""
    catalog_path = tmp_path / "model-routing.json"
    catalog_path.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "routine": {
                        "description": "Best for short factual questions and simple arithmetic.",
                        "relative_latency": "low",
                        "capabilities": ["short_factual_answers"],
                    },
                    "complex": {
                        "description": "Best for ambiguous requests requiring multi-step reasoning.",
                        "relative_latency": "high",
                        "capabilities": ["complex_reasoning"],
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
    routine_model = _InvocationTrackingBedrockModel(_ROUTINE_MODEL_ID)
    complex_model = _InvocationTrackingBedrockModel(_COMPLEX_MODEL_ID)
    router = ModelRouter(
        models=[
            RoutingCandidate(model=routine_model, name="routine"),
            RoutingCandidate(model=complex_model, name="complex"),
        ],
        strategy=InputComplexityStrategy(
            classifier_model=classifier_model,
            model_catalog_path=catalog_path,
        ),
    )
    agent = Agent(model=router, load_tools_from_directory=False)

    with caplog.at_level(logging.WARNING, logger="strands.models.routing.input_complexity_strategy"):
        result = agent("What is 2 + 2? Reply with one short sentence.")

    assert result.stop_reason == "end_turn"
    assert str(result).strip()
    assert sum(model.invocation_count > 0 for model in (routine_model, complex_model)) == 1
    assert not any("classification failed" in record.getMessage() for record in caplog.records)
