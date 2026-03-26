"""Tests for deployment constants."""

import sys

import pytest

from strands.experimental.deploy._constants import PYTHON_RUNTIME_MAP, agentcore_runtime_name, get_python_runtime


class TestAgentcoreRuntimeName:
    def test_prefixes_with_strands(self):
        assert agentcore_runtime_name("my-agent") == "strands_my-agent"

    def test_empty_name(self):
        assert agentcore_runtime_name("") == "strands_"


class TestGetPythonRuntime:
    def test_returns_runtime_for_current_python(self):
        version_key = (sys.version_info.major, sys.version_info.minor)
        max_supported = max(PYTHON_RUNTIME_MAP.keys())
        if version_key in PYTHON_RUNTIME_MAP:
            result = get_python_runtime()
            assert result.startswith("PYTHON_3_")
        elif version_key > max_supported:
            # Newer Python falls back to highest supported version
            result = get_python_runtime()
            assert result == PYTHON_RUNTIME_MAP[max_supported]
        else:
            with pytest.raises(ValueError, match="not supported"):
                get_python_runtime()

    def test_all_supported_versions_have_entries(self):
        for version, runtime in PYTHON_RUNTIME_MAP.items():
            assert runtime.startswith("PYTHON_")
            assert len(version) == 2  # (major, minor) tuple
