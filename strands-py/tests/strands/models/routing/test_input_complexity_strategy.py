"""Tests for classifier-driven proactive model selection."""

import json
from typing import Any

import pytest

from strands.models import InputComplexityStrategy, ModelCatalog, ModelRouter, RoutingAttempt, RoutingContext
from tests.fixtures.mocked_model_provider import MockedModelProvider


class _ConfiguredModel(MockedModelProvider):
    def __init__(self, model_id: str):
        super().__init__([{"role": "assistant", "content": [{"text": model_id}]}])
        self.model_id = model_id

    def get_config(self):
        return {
            "model_id": self.model_id,
            "api_key": "provider-secret",
            "api_base": "https://provider.internal",
        }


class _ClassifierModel(MockedModelProvider):
    def __init__(self, selected_index: int = 0, error: Exception | None = None):
        super().__init__([])
        self.selected_index = selected_index
        self.error = error
        self.calls = 0
        self.system_prompt: str | None = None

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs: Any):
        self.calls += 1
        self.system_prompt = system_prompt
        if self.error is not None:
            raise self.error
        yield {"output": output_model(selected_candidate_index=self.selected_index)}


def _context(router, attempts=()):
    return RoutingContext(
        messages=[{"role": "user", "content": [{"text": "Plan a safe migration"}]}],
        system_prompt="Be precise",
        tool_specs=[],
        candidates=router.candidates,
        invocation_state={},
        attempts=attempts,
    )


@pytest.mark.asyncio
async def test_select_supports_multiple_bare_candidates_and_exact_model_id_metadata():
    classifier = _ClassifierModel(selected_index=2)
    models = [_ConfiguredModel(f"model-{index}") for index in range(3)]
    catalog = ModelCatalog(
        {
            "model-2": {"input_cost_per_token": 0.000003, "supports_tool_calling": True},
            "candidate_2": {"input_cost_per_token": 999},
        }
    )
    strategy = InputComplexityStrategy(classifier_model=classifier, model_catalog=catalog)
    router = ModelRouter(models=models, strategy=strategy)

    selected = await strategy.select(_context(router))

    assert selected is router.candidates[2]
    assert classifier.calls == 1
    assert classifier.system_prompt is not None
    assert "provider-secret" not in classifier.system_prompt
    assert "https://provider.internal" not in classifier.system_prompt
    context = json.loads(classifier.system_prompt.split("Classification context: ", 1)[1])
    assert context["candidates"][2] == {
        "candidate_index": 2,
        "name": "candidate_2",
        "description": None,
        "model_id": "model-2",
        "model_profile": {"input_cost_per_token": 0.000003, "supports_tool_calling": True},
    }


@pytest.mark.asyncio
async def test_single_candidate_bypasses_classifier_and_attempts_decline_reselection():
    classifier = _ClassifierModel(selected_index=0)
    strategy = InputComplexityStrategy(classifier_model=classifier)
    router = ModelRouter(models=[_ConfiguredModel("only")], strategy=strategy)

    assert await strategy.select(_context(router)) is router.candidates[0]
    attempts = (RoutingAttempt(router.candidates[0], ValueError("down")),)
    assert await strategy.select(_context(router, attempts=attempts)) is None
    assert classifier.calls == 0


@pytest.mark.asyncio
async def test_classifier_failure_selects_first_candidate(caplog):
    classifier = _ClassifierModel(error=RuntimeError("classifier unavailable"))
    strategy = InputComplexityStrategy(classifier_model=classifier)
    router = ModelRouter(models=[_ConfiguredModel("first"), _ConfiguredModel("second")], strategy=strategy)

    with caplog.at_level("WARNING", logger="strands.models.routing.input_complexity_strategy"):
        selected = await strategy.select(_context(router))

    assert selected is router.candidates[0]
    assert "classification failed" in caplog.text
    assert "classifier unavailable" not in caplog.text


def test_constructor_requires_public_model_catalog():
    with pytest.raises(TypeError, match="model_catalog must be a ModelCatalog"):
        InputComplexityStrategy(classifier_model=_ClassifierModel(), model_catalog={})
