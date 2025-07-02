import configparser
import logging
import os
import sys

import boto3
import moto
import pytest

## Moto

# Get the log level from the environment variable
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.getLogger("strands").setLevel(log_level)
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s", handlers=[logging.StreamHandler(stream=sys.stdout)]
)


@pytest.fixture
def moto_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)


@pytest.fixture
def moto_mock_aws():
    with moto.mock_aws():
        yield


@pytest.fixture
def moto_cloudwatch_client():
    return boto3.client("cloudwatch")


## Boto3


@pytest.fixture
def boto3_profile_name():
    return "test-profile"


@pytest.fixture
def boto3_profile(boto3_profile_name):
    config = configparser.ConfigParser()
    config[boto3_profile_name] = {
        "aws_access_key_id": "test",
        "aws_secret_access_key": "test",
    }

    return config


@pytest.fixture
def boto3_profile_path(boto3_profile, tmp_path, monkeypatch):
    path = tmp_path / ".aws/credentials"
    path.parent.mkdir(exist_ok=True)
    with path.open("w") as fp:
        boto3_profile.write(fp)

    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(path))

    return path
