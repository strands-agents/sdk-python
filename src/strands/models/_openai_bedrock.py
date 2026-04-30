"""Internal helpers for routing OpenAI-compatible clients to Bedrock Mantle.

Converts an ``aws_config`` dict into the ``base_url`` and ``api_key`` that the
OpenAI Python SDK consumes. Tokens are minted on demand via
``aws_bedrock_token_generator.provide_token`` so long-running agents keep working
past the bearer token's maximum lifetime.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, TypedDict

from aws_bedrock_token_generator import provide_token

_MANTLE_BASE_URL_TEMPLATE = "https://bedrock-mantle.{region}.api.aws/v1"


class AwsConfig(TypedDict, total=False):
    """AWS-side config for reaching Bedrock Mantle via an OpenAI-compatible client.

    Attributes:
        region: AWS region hosting the Bedrock Mantle endpoint (required).
        credentials_provider: Optional botocore ``CredentialProvider`` forwarded to
            ``provide_token``. Defaults to the AWS credential chain.
        expiry: Optional ``timedelta`` for the bearer token's lifetime, forwarded
            to ``provide_token``.
    """

    region: str
    credentials_provider: Any
    expiry: timedelta


def resolve_bedrock_client_args(aws_config: AwsConfig, client_args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve an ``AwsConfig`` (plus optional ``client_args``) into OpenAI client kwargs.

    Mints a fresh bearer token on every call. When ``client_args`` is provided, its
    entries are preserved except for ``base_url`` and ``api_key``, which are always
    overridden by the values derived from ``aws_config``.

    Raises:
        ValueError: If ``aws_config['region']`` is missing.
    """
    region = aws_config.get("region")
    if not region:
        raise ValueError("aws_config must include a non-empty 'region'.")

    # Only forward optional kwargs when explicitly set, so provide_token's own
    # defaults apply. Passing expiry=None in particular crashes the library.
    token_kwargs: dict[str, Any] = {"region": region}
    if "credentials_provider" in aws_config:
        token_kwargs["aws_credentials_provider"] = aws_config["credentials_provider"]
    if "expiry" in aws_config:
        token_kwargs["expiry"] = aws_config["expiry"]
    token = provide_token(**token_kwargs)

    resolved: dict[str, Any] = dict(client_args or {})
    resolved["base_url"] = _MANTLE_BASE_URL_TEMPLATE.format(region=region)
    resolved["api_key"] = token
    return resolved
