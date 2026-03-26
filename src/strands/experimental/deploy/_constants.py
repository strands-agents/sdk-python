"""Static configuration for agent deployment."""

import sys

# Maps Python (major, minor) to AgentCore runtime identifiers
PYTHON_RUNTIME_MAP: dict[tuple[int, int], str] = {
    (3, 10): "PYTHON_3_10",
    (3, 11): "PYTHON_3_11",
    (3, 12): "PYTHON_3_12",
    (3, 13): "PYTHON_3_13",
}

# Resource naming prefix for AgentCore deployments
AGENTCORE_NAME_PREFIX = "strands_"

# Generated entrypoint filename written to CWD during deployment
ENTRYPOINT_FILENAME = "_strands_entrypoint.py"

# Local state directory name (relative to CWD)
STATE_DIR_NAME = ".strands_deploy"

# AgentCore deployment type identifier
DEPLOYMENT_TYPE = "direct_code_deploy"

# Default requirements for AgentCore deployments
AGENTCORE_BASE_REQUIREMENTS = ["bedrock-agentcore", "strands-agents"]

# Toolkit build artifacts to clean up after deployment
TOOLKIT_BUILD_ARTIFACTS = ["dependencies.hash", "dependencies.zip"]


def agentcore_runtime_name(name: str) -> str:
    """Build the AgentCore runtime name from a deploy name."""
    return f"{AGENTCORE_NAME_PREFIX}{name}"


def get_python_runtime() -> str:
    """Get the AgentCore Python runtime identifier for the current Python version."""
    version_key = (sys.version_info.major, sys.version_info.minor)
    runtime = PYTHON_RUNTIME_MAP.get(version_key)
    if runtime is None:
        # Fall back to the highest supported version if local Python is newer
        max_supported = max(PYTHON_RUNTIME_MAP.keys())
        if version_key > max_supported:
            return PYTHON_RUNTIME_MAP[max_supported]
        supported = ", ".join(f"{m}.{n}" for m, n in sorted(PYTHON_RUNTIME_MAP.keys()))
        raise ValueError(
            f"Python {sys.version_info.major}.{sys.version_info.minor} is not supported by AgentCore. "
            f"Supported versions: {supported}"
        )
    return runtime
