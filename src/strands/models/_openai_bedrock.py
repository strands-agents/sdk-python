"""Internal helpers for routing OpenAI-compatible clients to Bedrock Mantle.

Converts a ``bedrock_mantle_config`` dict into the ``base_url`` and ``api_key`` that the
OpenAI Python SDK consumes. Tokens are minted on demand via
``aws_bedrock_token_generator.provide_token`` so long-running agents survive the
bearer token's maximum lifetime.

``aws_bedrock_token_generator`` is part of the ``openai`` extras group
(``pip install strands-agents[openai]``) but is *not* included in the ``litellm``
or ``sagemaker`` extras, which also pull in the ``openai`` package. The import is
therefore lazy — it happens inside :func:`resolve_bedrock_client_args` so that
those other extras never trigger an ``ImportError`` at module load.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, TypedDict

import boto3
from botocore.credentials import CredentialProvider

_MANTLE_BASE_URL_TEMPLATE = "https://bedrock-mantle.{region}.api.aws/v1"
_MANTLE_DOCS_URL = "https://docs.aws.amazon.com/bedrock/latest/userguide/inference-openai.html"


class BedrockMantleConfig(TypedDict, total=False):
    """Config for routing an OpenAI-compatible client through Bedrock Mantle.

    Attributes:
        region: AWS region hosting the Bedrock Mantle endpoint. If omitted, resolved
            from ``boto_session`` (if provided) or the standard boto3 chain
            (``AWS_REGION`` / ``AWS_DEFAULT_REGION`` / active profile / EC2 metadata).
            A :class:`ValueError` is raised if none resolve.
        boto_session: Optional :class:`boto3.Session` used to resolve the region when
            ``region`` is not provided. Useful for picking up a non-default profile
            without exporting env vars.
        credentials_provider: Optional botocore :class:`~botocore.credentials.CredentialProvider`
            forwarded to ``provide_token``. Omit to let the token generator use the
            standard AWS credential chain.
        expiry: Optional ``timedelta`` for the bearer token's lifetime, forwarded to
            ``provide_token``. Defaults to the generator's built-in lifetime when
            omitted.
    """

    region: str
    boto_session: boto3.Session
    credentials_provider: CredentialProvider
    expiry: timedelta


def _resolve_region(config: BedrockMantleConfig) -> str:
    """Resolve the AWS region, preferring explicit config then falling back to boto3.

    Raises:
        ValueError: If no region can be resolved from the config, an attached session,
            or the standard boto3 credential chain.
    """
    region = config.get("region")
    if region:
        return region

    session = config.get("boto_session")
    if session is not None and session.region_name:
        return str(session.region_name)

    # ``boto3.Session()`` with no args reads ``AWS_REGION`` / ``AWS_DEFAULT_REGION``,
    # the active profile, and falls back to EC2 instance metadata — the same chain
    # :class:`BedrockModel` uses.
    default_region = boto3.Session().region_name
    if default_region:
        return str(default_region)

    raise ValueError(
        "Could not resolve an AWS region for Bedrock Mantle. Pass 'region' in "
        "bedrock_mantle_config, attach a boto_session with a configured region, or set "
        f"AWS_REGION in the environment. See {_MANTLE_DOCS_URL} for supported regions."
    )


def resolve_bedrock_client_args(
    config: BedrockMantleConfig, client_args: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Resolve a ``BedrockMantleConfig`` (plus optional ``client_args``) into OpenAI client kwargs.

    Mints a fresh bearer token on every call. Callers are expected to validate that
    ``client_args`` does not contain ``base_url`` or ``api_key`` before calling this
    function (typically at ``__init__`` time for fail-fast behavior).

    Raises:
        ValueError: If no region can be resolved.
        ImportError: If ``aws-bedrock-token-generator`` is not installed.
        RuntimeError: If token minting fails (e.g. missing AWS credentials).
    """
    region = _resolve_region(config)

    # ``aws-bedrock-token-generator`` is included in the ``openai`` extras group but not in
    # ``litellm`` or ``sagemaker`` (which also depend on the ``openai`` package). The lazy
    # import keeps those extras from hitting an ImportError at module load.
    try:
        from aws_bedrock_token_generator import provide_token
    except ImportError as e:
        raise ImportError(
            "bedrock_mantle_config requires the 'aws-bedrock-token-generator' package. "
            "Install it with: pip install strands-agents[openai]"
        ) from e

    # Only forward kwargs the user set; provide_token rejects expiry=None.
    token_kwargs: dict[str, Any] = {"region": region}
    if "credentials_provider" in config:
        token_kwargs["aws_credentials_provider"] = config["credentials_provider"]
    if "expiry" in config:
        token_kwargs["expiry"] = config["expiry"]

    try:
        token = provide_token(**token_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to mint Bedrock Mantle bearer token for region '{region}'. "
            "Verify your AWS credentials and network connectivity."
        ) from e

    resolved: dict[str, Any] = dict(client_args or {})
    resolved["base_url"] = _MANTLE_BASE_URL_TEMPLATE.format(region=region)
    resolved["api_key"] = token
    return resolved
