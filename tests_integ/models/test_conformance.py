from unittest import SkipTest

import pytest
from pydantic import BaseModel

from strands import Agent
from strands.models import Model
from tests_integ.models.providers import ProviderInfo, all_providers, cohere, llama, mistral


def get_models():
    return [
        pytest.param(
            provider_info,
            id=provider_info.id,  # Adds the provider name to the test name
            marks=provider_info.mark,  # ignores tests that don't have the requirements
        )
        for provider_info in all_providers
    ]


@pytest.fixture(params=get_models())
def provider_info(request) -> ProviderInfo:
    return request.param


@pytest.fixture()
def skip_for(provider_info: list[ProviderInfo]):
    """A fixture which provides a function to skip the test if the provider is one of the providers specified."""

    def skip_for_any_provider_in_list(providers: list[ProviderInfo], description: str):
        """Skips the current test is the provider is one of those provided."""
        if provider_info in providers:
            raise SkipTest(f"Skipping test for {provider_info.id}: {description}")

    return skip_for_any_provider_in_list


@pytest.fixture()
def model(provider_info):
    return provider_info.create_model()


def test_model_can_be_constructed(model: Model, skip_for):
    assert model is not None
    pass


def test_structured_output_is_forced(skip_for, model):
    """Tests that structured_output is always forced to return a value even if model doesn't have any information."""
    skip_for([mistral, cohere, llama], "structured_output is not forced for provider ")

    class Weather(BaseModel):
        time: str
        weather: str

    agent = Agent(model)

    result = agent.structured_output(Weather, "How are you?")

    assert len(result.time) > 0
    assert len(result.weather) > 0
