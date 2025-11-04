"""Pytest fixtures for bidirectional streaming integration tests."""

import logging

import pytest

from .utils.audio_generator import AudioGenerator

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def audio_generator():
    """Provide AudioGenerator instance for tests."""
    return AudioGenerator(region="us-east-1")


@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for tests."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    # Reduce noise from some loggers
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
