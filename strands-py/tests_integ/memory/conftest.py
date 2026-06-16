"""Shared fixtures for the live Bedrock Knowledge Base integration tests.

The SSM test-infra parameters are resolved once in a session-scoped fixture
(:func:`bedrock_kb_context`) and reused across tests. Tests ``pytest.skip(...)`` when the context's
``should_skip`` flag is set (SSM unreachable or the KB id missing).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import boto3
import pytest

logger = logging.getLogger(__name__)

# SSM parameter names for the shared test infrastructure.
_SSM_PARAMS = {
    "knowledge_base_id": "/strands/test-infra/bedrock-knowledge-base/knowledge-base-id",
    "custom_data_source_id": "/strands/test-infra/bedrock-knowledge-base/custom-data-source-id",
    "s3_data_source_id": "/strands/test-infra/bedrock-knowledge-base/s3-data-source-id",
    "s3_bucket": "/strands/test-infra/bedrock-knowledge-base/s3-source-bucket-name",
}

# Optional local-dev overrides. Set these env vars to point the tests at your own resources without
# touching SSM.
_OVERRIDE_ENV = {
    "knowledge_base_id": "STRANDS_TEST_KB_ID",
    "custom_data_source_id": "STRANDS_TEST_KB_CUSTOM_DS_ID",
    "s3_data_source_id": "STRANDS_TEST_KB_S3_DS_ID",
    "s3_bucket": "STRANDS_TEST_KB_S3_BUCKET",
}


@dataclass
class BedrockKnowledgeBaseContext:
    """Resolved Bedrock KB test-infra parameters plus a skip flag."""

    should_skip: bool
    knowledge_base_id: str | None = None
    custom_data_source_id: str | None = None
    s3_data_source_id: str | None = None
    s3_bucket: str | None = None


def _resolve_ssm_parameters() -> dict[str, str | None] | None:
    """Batch-read the SSM parameters by name; returns a key->value map, or ``None`` on failure.

    Parameters that don't exist in SSM resolve to ``None``.
    """
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    try:
        client = boto3.Session(region_name=region).client("ssm")
        response = client.get_parameters(Names=list(_SSM_PARAMS.values()))
    except Exception as error:  # noqa: BLE001 - any failure means "skip", not "error".
        logger.warning("error=<%s> | failed to resolve Bedrock KB SSM parameters", error)
        return None

    by_name = {param["Name"]: param["Value"] for param in response.get("Parameters", [])}
    return {key: by_name.get(name) for key, name in _SSM_PARAMS.items()}


@pytest.fixture(scope="session")
def bedrock_kb_context() -> BedrockKnowledgeBaseContext:
    """Resolve the Bedrock KB test-infra parameters once per session.

    Local-dev overrides (the ``STRANDS_TEST_KB_*`` env vars) take precedence over SSM. If SSM is
    unreachable or the knowledge base id can't be resolved, returns a context with ``should_skip`` set
    so dependent tests skip cleanly.
    """
    overrides = {key: os.environ.get(env_var) for key, env_var in _OVERRIDE_ENV.items()}

    resolved = _resolve_ssm_parameters() or {}

    merged = {key: overrides.get(key) or resolved.get(key) for key in _SSM_PARAMS}

    if not merged.get("knowledge_base_id"):
        logger.info("Bedrock KB id not available (SSM/overrides) - KB integration tests will be skipped")
        return BedrockKnowledgeBaseContext(should_skip=True)

    return BedrockKnowledgeBaseContext(
        should_skip=False,
        knowledge_base_id=merged["knowledge_base_id"],
        custom_data_source_id=merged["custom_data_source_id"],
        s3_data_source_id=merged["s3_data_source_id"],
        s3_bucket=merged["s3_bucket"],
    )


@pytest.fixture
def cleanup_registrar() -> list:
    """A list tests append best-effort cleanup callbacks to; drained here in teardown.

    Each registered callback runs once at test teardown; exceptions are swallowed so cleanup never
    masks a real assertion failure.
    """
    callbacks: list = []
    yield callbacks
    for callback in callbacks:
        try:
            callback()
        except Exception:  # noqa: BLE001 - best-effort cleanup.
            pass
