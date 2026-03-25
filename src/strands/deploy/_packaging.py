"""Packaging utilities for agent deployment.

Handles zipping agent code, generating entrypoints, and uploading to S3.
"""

import io
import logging
import os
import zipfile
from typing import TYPE_CHECKING

import boto3

from ._constants import PACKAGING_EXCLUDES, S3_BUCKET_PREFIX
from ._exceptions import DeployPackagingException

if TYPE_CHECKING:
    from ..agent.agent import Agent

logger = logging.getLogger(__name__)

AGENTCORE_ENTRYPOINT_TEMPLATE = '''\
"""Auto-generated Strands Agent entrypoint for AgentCore."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent

app = BedrockAgentCoreApp()

agent = Agent(
    model={model_repr},
    system_prompt={system_prompt_repr},
    name={name_repr},
)


@app.entrypoint
def invoke(payload):
    prompt = payload.get("prompt", "Hello!")
    result = agent(prompt)
    return {{"result": str(result), "stop_reason": result.stop_reason}}


if __name__ == "__main__":
    app.run()
'''


def extract_agent_config(agent: "Agent") -> dict:
    """Extract serializable configuration from an Agent instance."""
    config: dict = {
        "name": agent.name,
        "system_prompt": agent.system_prompt if isinstance(agent.system_prompt, str) else None,
    }

    # Extract model ID from the model provider
    model = agent.model
    if hasattr(model, "config"):
        model_config = model.config
        if hasattr(model_config, "get"):
            config["model_id"] = model_config.get("model_id")
        elif hasattr(model_config, "model_id"):
            config["model_id"] = model_config.model_id

    return config


def generate_agentcore_entrypoint(agent: "Agent") -> str:
    """Generate a BedrockAgentCoreApp entrypoint that reconstructs the agent."""
    config = extract_agent_config(agent)

    model_id = config.get("model_id")
    model_repr = repr(model_id) if model_id else "None"
    system_prompt = config.get("system_prompt")
    system_prompt_repr = repr(system_prompt) if system_prompt else "None"
    name_repr = repr(config.get("name", "Strands Agent"))

    return AGENTCORE_ENTRYPOINT_TEMPLATE.format(
        model_repr=model_repr,
        system_prompt_repr=system_prompt_repr,
        name_repr=name_repr,
    )


def _should_exclude(path: str, base_dir: str) -> bool:
    """Check if a path should be excluded from the deployment zip."""
    rel = os.path.relpath(path, base_dir)
    parts = rel.split(os.sep)
    for part in parts:
        if part in PACKAGING_EXCLUDES:
            return True
        # Match wildcard patterns like *.egg-info
        for pattern in PACKAGING_EXCLUDES:
            if pattern.startswith("*") and part.endswith(pattern[1:]):
                return True
    return False


def create_code_zip(entrypoint_code: str, base_dir: str | None = None) -> bytes:
    """Create a zip of the working directory plus the generated entrypoint.

    Args:
        entrypoint_code: Generated Python entrypoint source code.
        base_dir: Directory to package. Defaults to CWD.

    Returns:
        Zip file contents as bytes.
    """
    base_dir = base_dir or os.getcwd()
    buffer = io.BytesIO()
    file_count = 0

    try:
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(base_dir):
                # Prune excluded directories in-place
                dirs[:] = [d for d in dirs if not _should_exclude(os.path.join(root, d), base_dir)]

                for filename in files:
                    if filename.endswith(".pyc"):
                        continue
                    full_path = os.path.join(root, filename)
                    if _should_exclude(full_path, base_dir):
                        continue
                    arc_name = os.path.relpath(full_path, base_dir)
                    zf.write(full_path, arc_name)
                    file_count += 1

            # Add generated entrypoint
            zf.writestr("_strands_entrypoint.py", entrypoint_code)
            file_count += 1
    except OSError as e:
        raise DeployPackagingException(f"Failed to create deployment zip: {e}") from e

    zip_bytes = buffer.getvalue()
    size_kb = len(zip_bytes) / 1024
    print(f"  Packaging agent code ({file_count} files, {size_kb:.0f}KB)")

    return zip_bytes


def ensure_s3_bucket(bucket_name: str, region: str) -> str:
    """Create the S3 bucket if it doesn't exist.

    Returns the bucket name.
    """
    s3 = boto3.client("s3", region_name=region)
    try:
        s3.head_bucket(Bucket=bucket_name)
    except s3.exceptions.ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("404", "NoSuchBucket"):
            create_args: dict = {"Bucket": bucket_name}
            # us-east-1 doesn't accept LocationConstraint
            if region != "us-east-1":
                create_args["CreateBucketConfiguration"] = {"LocationConstraint": region}
            s3.create_bucket(**create_args)
            # Block public access
            s3.put_public_access_block(
                Bucket=bucket_name,
                PublicAccessBlockConfiguration={
                    "BlockPublicAcls": True,
                    "IgnorePublicAcls": True,
                    "BlockPublicPolicy": True,
                    "RestrictPublicBuckets": True,
                },
            )
            logger.debug("bucket=<%s> | created S3 bucket", bucket_name)
        else:
            raise
    return bucket_name


def upload_to_s3(zip_bytes: bytes, bucket: str, key: str, region: str) -> str:
    """Upload zip bytes to S3.

    Returns the S3 URI.
    """
    s3 = boto3.client("s3", region_name=region)
    s3.put_object(Bucket=bucket, Key=key, Body=zip_bytes)
    s3_uri = f"s3://{bucket}/{key}"
    print(f"  Uploading to {s3_uri}")
    return s3_uri


def get_s3_bucket_name(account_id: str, region: str) -> str:
    """Generate the S3 bucket name for deployment artifacts."""
    return f"{S3_BUCKET_PREFIX}-{account_id}-{region}"
