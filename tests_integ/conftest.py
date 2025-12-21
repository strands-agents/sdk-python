import json
import logging
import os

import boto3
import pytest
from botocore.exceptions import NoRegionError

logger = logging.getLogger(__name__)


def pytest_sessionstart(session):
    _load_api_keys_from_secrets_manager()


## Data


@pytest.fixture
def yellow_img(pytestconfig):
    path = pytestconfig.rootdir / "tests_integ/yellow.png"
    with open(path, "rb") as fp:
        return fp.read()


@pytest.fixture
def letter_pdf(pytestconfig):
    path = pytestconfig.rootdir / "tests_integ/letter.pdf"
    with open(path, "rb") as fp:
        return fp.read()


## Async


@pytest.fixture(scope="session")
def agenerator():
    async def agenerator(items):
        for item in items:
            yield item

    return agenerator


@pytest.fixture(scope="session")
def alist():
    async def alist(items):
        return [item async for item in items]

    return alist


## Models


def _load_api_keys_from_secrets_manager():
    """Load API keys as environment variables from AWS Secrets Manager."""
    secret_name = os.getenv("STRANDS_TEST_API_KEYS_SECRET_NAME")
    if secret_name:
        try:
            region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
            if not region:
                logger.warning(
                    "STRANDS_TEST_API_KEYS_SECRET_NAME is set but AWS region is not configured; "
                    "skipping Secrets Manager lookup"
                )
            else:
                session = boto3.session.Session(region_name=region)
                client = session.client(service_name="secretsmanager", region_name=region)
                response = client.get_secret_value(SecretId=secret_name)

                if "SecretString" in response:
                    secret = json.loads(response["SecretString"])
                    for key, value in secret.items():
                        os.environ[f"{key.upper()}_API_KEY"] = str(value)

        except NoRegionError:
            logger.warning("AWS region not configured; skipping Secrets Manager lookup")
        except Exception as e:
            logger.warning("Error retrieving secret", exc_info=e)

    """
    Validate that required environment variables are set when running in GitHub Actions.
    This prevents tests from being unintentionally skipped due to missing credentials.
    """
    if os.environ.get("GITHUB_ACTIONS") != "true":
        logger.warning("Tests running outside GitHub Actions, skipping required provider validation")
        return

    required_providers = {
        "ANTHROPIC_API_KEY",
        "COHERE_API_KEY",
        "MISTRAL_API_KEY",
        "OPENAI_API_KEY",
        "WRITER_API_KEY",
    }
    for provider in required_providers:
        if provider not in os.environ or not os.environ[provider]:
            raise ValueError(f"Missing required environment variables for {provider}")
