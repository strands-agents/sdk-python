import pytest

from strands.models import Model
from tests_integ.models.providers import ProviderInfo, all_providers


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
def model(provider_info):
    return provider_info.create_model()


def test_model_can_be_constructed(model: Model):
    assert model is not None
    pass
