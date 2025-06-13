"""Pytest configuration for A2A tests."""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_uvicorn():
    """Mock uvicorn.run to prevent actual server startup during tests."""
    with patch("uvicorn.run") as mock:
        yield mock
