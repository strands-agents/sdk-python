"""Internal helpers for routing OpenAI-compatible clients to Bedrock Mantle.

Converts a ``bedrock_mantle_config`` dict into the ``base_url`` and ``api_key`` that the
OpenAI Python SDK consumes. Tokens are minted on demand via
``aws_bedrock_token_generator.provide_token`` so long-running agents survive the
bearer token's maximum lifetime.

``aws_bedrock_token_generator`` is part of the ``openai`` extras group
(``pip install strands-agents[openai]``) but is *not* included in the ``litellm``
or ``sagemaker`` extras, which also pull in the ``openai`` package. The import is
therefore lazy â€” it happens inside :func:`resolve_bedrock_client_args` so that
those other extras never trigger an ``ImportError`` at module load.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, TypedDict

import boto3
from botocore.credentials import CredentialProvider

_MANTLE_BASE_URL_TEMPLATE = "https://bedrock-mantle.{region}.api.aws/{path_prefix}"
_MANTLE_DOCS_URL = "https://docs.aws.amazon.com/bedrock/latest/userguide/inference-openai.html"

# Model families that require the /openai/v1 path prefix instead of /v1.
# See: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-openai.html
# - gpt-oss-* models use /v1 (standard path)
# - gpt-5.* models use /openai/v1 (OpenAI-specific path)
_OPENAI_PATH_MODEL_PREFIXES = (
    "openai.gpt-5",
)


def _resolve_mantle_base_url(region: str, model_id: str | None = None) -> str:
    """Resolve the correct Bedrock Mantle base URL for a given model.

    The Bedrock Mantle endpoint uses different path prefixes depending on the
    model family:
    - ``openai.gpt-5.*`` models require ``/openai/v1``
    - All other models (e.g. ``openai.gpt-oss-*``) use ``/v1``

    Args:
        region: AWS region hosting the Bedrock Mantle endpoint.
        model_id: The model identifier (e.g. ``openai.gpt-5.4``). When ``None``,
            falls back to the default ``/v1`` path.

    Returns:
        The fully-qualified Mantle base URL for the given model and region.
    """
    path_prefix = "v1"
    if model_id:
        for prefix in _OPENAI_PATH_MODEL_PREFIXES:
            if model_id.startswith(prefix):
                path_prefix = "openai/v1"
                break

    return _MANTLE_BASE_URL_TEMPLATE.format(region=region, path_prefix=path_prefix)


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
        base_url: Optional base URL override. When set, bypasses the automatic
            model-family-aware URL resolution. Useful for new model families
            not yet mapped in ``_OPENAI_PATH_MODEL_PREFIXES``.
    """

    region: str
    boto_session: boto3.Session
    credentials_provider: CredentialProvider
    expiry: timedelta
    base_url: str


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
    # the active profile, and falls back to EC2 instance metadata â€” the same chain
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
    config: BedrockMantleConfig,
    client_args: dict[str, Any] | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Resolve a ``BedrockMantleConfig`` (plus optional ``client_args``) into OpenAI client kwargs.

    Mints a fresh bearer token on every call. Callers are expected to validate that
    ``client_args`` does not contain ``base_url`` or ``api_key`` before calling this
    function (typically at ``__init__`` time for fail-fast behavior).

    The ``model_id`` parameter enables model-family-aware URL resolution:
    - ``openai.gpt-5.*`` models are routed to ``/openai/v1``
    - All other models use the default ``/v1`` path

    A ``base_url`` key in ``config`` overrides automatic resolution for models
    not yet mapped.

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

    # Allow explicit base_url override in config for unmapped model families.
    if "base_url" in config:
        resolved["base_url"] = config["base_url"]
    else:
        resolved["base_url"] = _resolve_mantle_base_url(region, model_id)

    resolved["api_key"] = token
    return resolved
