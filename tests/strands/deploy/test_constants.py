"""Tests for deployment constants."""

import sys

import pytest

from strands.deploy._constants import PYTHON_RUNTIME_MAP, SUPPORTED_REGIONS, get_python_runtime


class TestGetPythonRuntime:
    def test_returns_runtime_for_current_python(self):
        version_key = (sys.version_info.major, sys.version_info.minor)
        if version_key in PYTHON_RUNTIME_MAP:
            result = get_python_runtime()
            assert result.startswith("PYTHON_3_")
        else:
            with pytest.raises(ValueError, match="not supported"):
                get_python_runtime()

    def test_all_supported_versions_have_entries(self):
        for version, runtime in PYTHON_RUNTIME_MAP.items():
            assert runtime.startswith("PYTHON_")
            assert len(version) == 2  # (major, minor) tuple


class TestSupportedRegions:
    def test_contains_common_regions(self):
        assert "us-east-1" in SUPPORTED_REGIONS
        assert "us-west-2" in SUPPORTED_REGIONS
        assert "eu-west-1" in SUPPORTED_REGIONS

    def test_has_17_regions(self):
        assert len(SUPPORTED_REGIONS) == 17
