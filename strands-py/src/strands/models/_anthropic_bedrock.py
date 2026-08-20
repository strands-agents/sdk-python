"""Internal helpers for routing the Anthropic client to Bedrock Mantle.

Converts a ``bedrock_mantle_config`` dict into the keyword arguments that
``anthropic.AsyncAnthropicBedrockMantle`` consumes. That client derives the Mantle base URL
from the region, sends the required ``anthropic-version`` header, and signs every request
with SigV4 on its own, so nothing here mints or caches a credential.
"""

from __future__ import annotations

from typing import Any, TypedDict

import boto3

from ._validation import validate_region

_MANTLE_DOCS_URL = "https://docs.aws.amazon.com/bedrock/latest/userguide/inference-messages-api.html"

# Client kwargs this module derives from the config. A caller that also accepts raw client
# args must reject these, or the two sources would silently disagree.
MANTLE_DERIVED_CLIENT_ARGS = ("aws_region", "aws_profile", "api_key")


class BedrockMantleConfig(TypedDict, total=False):
    """Config for routing the Anthropic client through Bedrock Mantle.

    Attributes:
        region: AWS region hosting the Bedrock Mantle endpoint. If omitted, resolved from
            ``profile`` (if provided) or the standard boto3 chain (``AWS_REGION`` /
            ``AWS_DEFAULT_REGION`` / active profile / EC2 metadata). A :class:`ValueError`
            is raised if none resolve.
        profile: Optional AWS profile to authenticate with. Selects SigV4 authentication.
        api_key: Optional Amazon Bedrock API key. When set, requests carry it as a bearer
            token instead of being signed with SigV4. Omit to use the standard AWS
            credential chain.
    """

    region: str
    profile: str
    api_key: str


def _resolve_region(config: BedrockMantleConfig) -> str:
    """Resolve the AWS region, preferring explicit config then falling back to boto3.

    The resolved region is validated before it is returned, since
    ``AsyncAnthropicBedrockMantle`` interpolates it into the Mantle endpoint URL.

    Args:
        config: The Bedrock Mantle config to resolve the region from.

    Returns:
        The resolved AWS region.

    Raises:
        ValueError: If no region can be resolved from the config, the named profile, or the
            standard boto3 credential chain, or if the resolved region is not a well-formed
            AWS region identifier.
    """
    region = config.get("region")
    if region:
        return validate_region(region)

    # ``boto3.Session()`` reads ``AWS_REGION`` / ``AWS_DEFAULT_REGION``, the active profile,
    # and EC2 instance metadata. The Anthropic client's own fallback reads only the two env
    # vars, so a profile-configured region would otherwise go unseen.
    profile = config.get("profile")
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    if session.region_name:
        return validate_region(str(session.region_name))

    raise ValueError(
        "Could not resolve an AWS region for Bedrock Mantle. Pass 'region' in "
        "bedrock_mantle_config, configure a region on the named profile, or set AWS_REGION "
        f"in the environment. See {_MANTLE_DOCS_URL} for supported regions."
    )


def resolve_bedrock_client_args(
    config: BedrockMantleConfig, client_args: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Resolve a ``BedrockMantleConfig`` (plus optional ``client_args``) into Mantle client kwargs.

    Callers are expected to validate that ``client_args`` carries none of
    :data:`MANTLE_DERIVED_CLIENT_ARGS` before calling this function (typically at
    ``__init__`` time for fail-fast behavior).

    Args:
        config: Config for routing the Anthropic client through Bedrock Mantle.
        client_args: Additional arguments for the underlying Anthropic client.

    Returns:
        Keyword arguments for ``anthropic.AsyncAnthropicBedrockMantle``.

    Raises:
        ValueError: If no region can be resolved.
    """
    resolved: dict[str, Any] = dict(client_args or {})
    resolved["aws_region"] = _resolve_region(config)

    # Only forward keys the user set; the Anthropic client reads auth precedence off which
    # arguments are not None, so passing an explicit None would change the selected mode.
    if "profile" in config:
        resolved["aws_profile"] = config["profile"]
    if "api_key" in config:
        resolved["api_key"] = config["api_key"]
    return resolved
