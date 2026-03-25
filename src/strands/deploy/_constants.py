"""Static configuration for agent deployment."""

import sys

SUPPORTED_REGIONS = [
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
    "ap-south-1",
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-northeast-3",
    "ap-southeast-1",
    "ap-southeast-2",
    "ca-central-1",
    "eu-central-1",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "eu-north-1",
    "sa-east-1",
]

# Maps Python (major, minor) to AgentCore runtime identifiers
PYTHON_RUNTIME_MAP: dict[tuple[int, int], str] = {
    (3, 10): "PYTHON_3_10",
    (3, 11): "PYTHON_3_11",
    (3, 12): "PYTHON_3_12",
    (3, 13): "PYTHON_3_13",
    (3, 14): "PYTHON_3_14",
}

# IAM trust policy for Bedrock AgentCore
AGENTCORE_TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ],
}

# IAM execution policy for AgentCore runtimes
AGENTCORE_EXECUTION_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BedrockModelInvocation",
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
            ],
            "Resource": "*",
        },
        {
            "Sid": "CloudWatchLogs",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
            ],
            "Resource": "arn:aws:logs:*:*:/aws/bedrock-agentcore/runtimes/*",
        },
        {
            "Sid": "XRayTracing",
            "Effect": "Allow",
            "Action": [
                "xray:PutTraceSegments",
                "xray:PutTelemetryRecords",
            ],
            "Resource": "*",
        },
        {
            "Sid": "CloudWatchMetrics",
            "Effect": "Allow",
            "Action": ["cloudwatch:PutMetricData"],
            "Resource": "*",
            "Condition": {"StringEquals": {"cloudwatch:namespace": "bedrock-agentcore"}},
        },
    ],
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

# S3 bucket naming prefix for deployment artifacts
S3_BUCKET_PREFIX = "strands-deploy"

# Polling configuration for waiting on runtime readiness
RUNTIME_POLL_INTERVAL_SECONDS = 5
RUNTIME_POLL_MAX_ATTEMPTS = 60  # 5 minutes at 5s intervals

# IAM propagation retry configuration
IAM_PROPAGATION_DELAY_SECONDS = 2
IAM_PROPAGATION_MAX_ATTEMPTS = 5


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
