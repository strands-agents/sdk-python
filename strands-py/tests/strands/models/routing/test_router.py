"""Tests for ModelRouter core: candidate validation, default resolution, guards."""

import pytest

from strands import Agent, Plugin
from strands.models import BedrockModel
from strands.models.routing import ModelRouter
from tests.fixtures.mocked_model_provider import MockedModelProvider


class StatefulModel(MockedModelProvider):
    @property
    def stateful(self):
        return True


def _model(text="hi"):
    return MockedModelProvider([{"role": "assistant", "content": [{"text": text}]}])


# --- plugin identity ---


def test_router_is_a_plugin_with_stable_name():
    router = ModelRouter(models=[_model()])

    assert isinstance(router, Plugin)
    assert router.name == "strands:model-router"


# --- default resolution (first candidate) ---


def test_default_model_is_first_candidate():
    m0, m1 = _model("0"), _model("1")
    router = ModelRouter(models=[m0, m1])

    assert router.default_model is m0


def test_nested_router_default_resolves_recursively():
    inner_model = _model()
    inner = ModelRouter(models=[inner_model])
    outer = ModelRouter(models=[inner, _model("x")])

    assert outer.default_model is inner_model


def test_repeated_model_object_is_allowed():
    m = _model()
    router = ModelRouter(models=[m, m])

    assert router.default_model is m


# --- guards ---


def test_empty_models_raises():
    with pytest.raises(ValueError, match="at least one"):
        ModelRouter(models=[])


def test_stateful_candidate_raises():
    with pytest.raises(ValueError, match=r"StatefulModel.*stateful"):
        ModelRouter(models=[StatefulModel([])])


def test_mapping_models_raises():
    with pytest.raises(TypeError, match="sequence of candidates"):
        ModelRouter(models={"cheap": _model()})


def test_bare_string_models_raises():
    with pytest.raises(TypeError, match="sequence of candidates"):
        ModelRouter(models="my-model-id")


def test_string_candidate_is_rejected():
    with pytest.raises(TypeError, match="candidate must be"):
        ModelRouter(models=["my-model-id"])


def test_invalid_candidate_raises():
    with pytest.raises(TypeError, match="candidate must be"):
        ModelRouter(models=[object()])


# --- agent integration ---


def test_agent_accepts_model_router_and_exposes_default():
    m = _model("routed")
    router = ModelRouter(models=[m])
    agent = Agent(model=router, callback_handler=None)

    assert agent.model is m
    assert agent._model_router is router


def test_agent_registers_router_as_plugin():
    router = ModelRouter(models=[_model()])
    agent = Agent(model=router, callback_handler=None)

    assert router.name in agent._plugin_registry._plugins


def test_router_via_plugins_is_rejected():
    router = ModelRouter(models=[_model()])
    with pytest.raises(ValueError, match=r"model=.*not plugins"):
        Agent(plugins=[router], callback_handler=None)


def test_agent_runs_with_router_using_first_candidate():
    router = ModelRouter(models=[_model("routed")])
    agent = Agent(model=router, callback_handler=None)

    result = agent("hello")

    assert result.message["content"][0]["text"] == "routed"


def test_bedrock_model_object_is_a_valid_candidate():
    haiku = BedrockModel(model_id="haiku")
    router = ModelRouter(models=[haiku, BedrockModel(model_id="opus")])

    assert router.default_model is haiku
