"""Static configuration for agent deployment."""

import sys

# Maps Python (major, minor) to AgentCore runtime identifiers
PYTHON_RUNTIME_MAP: dict[tuple[int, int], str] = {
    (3, 10): "PYTHON_3_10",
    (3, 11): "PYTHON_3_11",
    (3, 12): "PYTHON_3_12",
    (3, 13): "PYTHON_3_13",
    (3, 14): "PYTHON_3_14",
}

# Directories and files to exclude when packaging agent code
PACKAGING_EXCLUDES = {
    ".strands",
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    ".env",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "dist",
    "build",
    "*.egg-info",
}


def get_python_runtime() -> str:
    """Get the AgentCore Python runtime identifier for the current Python version."""
    version_key = (sys.version_info.major, sys.version_info.minor)
    runtime = PYTHON_RUNTIME_MAP.get(version_key)
    if runtime is None:
        supported = ", ".join(f"{m}.{n}" for m, n in sorted(PYTHON_RUNTIME_MAP.keys()))
        raise ValueError(
            f"Python {sys.version_info.major}.{sys.version_info.minor} is not supported by AgentCore. "
            f"Supported versions: {supported}"
        )
    return runtime
