"""AgentCore deployment target for Bedrock AgentCore runtimes."""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import boto3

from ._base import DeployTarget
from ._constants import (
    AGENTCORE_EXECUTION_POLICY,
    AGENTCORE_TRUST_POLICY,
    IAM_PROPAGATION_DELAY_SECONDS,
    IAM_PROPAGATION_MAX_ATTEMPTS,
    RUNTIME_POLL_INTERVAL_SECONDS,
    RUNTIME_POLL_MAX_ATTEMPTS,
    get_python_runtime,
)
from ._exceptions import DeployException, DeployTargetException
from ._packaging import (
    create_code_zip,
    ensure_s3_bucket,
    generate_agentcore_entrypoint,
    get_s3_bucket_name,
    upload_to_s3,
)
from ._state import DeployState

if TYPE_CHECKING:
    from ..agent.agent import Agent
    from . import DeployConfig, DeployResult
    from ._state import StateManager

logger = logging.getLogger(__name__)


class AgentCoreTarget(DeployTarget):
    """Deploy a Strands agent to AWS Bedrock AgentCore.

    Uses the bedrock-agentcore-control API to create/update runtimes
    and endpoints. Agent code is uploaded to S3 as a zip archive.
    """

    def validate(self, config: "DeployConfig") -> None:
        """Check prerequisites for AgentCore deployment."""
        # Verify boto3 has the bedrock-agentcore-control service
        try:
            boto3.client("bedrock-agentcore-control", region_name="us-east-1")
        except Exception as e:
            raise DeployException(
                "Failed to create bedrock-agentcore-control client. "
                "Ensure your boto3 version supports AgentCore "
                f"(pip install --upgrade boto3 botocore): {e}"
            ) from e

        # Verify AWS credentials are configured
        try:
            sts = boto3.client("sts")
            sts.get_caller_identity()
        except Exception as e:
            raise DeployException(f"AWS credentials not configured: {e}") from e

    def deploy(self, agent: "Agent", config: "DeployConfig", state_manager: "StateManager") -> "DeployResult":
        """Deploy agent to AgentCore."""
        from . import DeployResult

        region = self._resolve_region(agent, config)
        print(f"  Region: {region}")

        existing = state_manager.load(config.name)
        account_id = self._get_account_id()

        # Step 1: IAM role
        role_arn = self._ensure_iam_role(config.name, existing, account_id)

        # Step 2: Generate entrypoint and package code
        entrypoint_code = generate_agentcore_entrypoint(agent)
        zip_bytes = create_code_zip(entrypoint_code)

        # Step 3: Upload to S3
        bucket_name = get_s3_bucket_name(account_id, region)
        ensure_s3_bucket(bucket_name, region)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        s3_key = f"{config.name}/{timestamp}.zip"
        upload_to_s3(zip_bytes, bucket_name, s3_key, region)

        # Step 4: Create or update runtime
        control = boto3.client("bedrock-agentcore-control", region_name=region)
        python_runtime = get_python_runtime()

        artifact = {
            "codeConfiguration": {
                "code": {"s3": {"bucket": bucket_name, "prefix": s3_key}},
                "runtime": python_runtime,
                "entryPoint": ["python", "_strands_entrypoint.py"],
            }
        }

        network_config = {"networkMode": "PUBLIC"}

        if existing and existing.get("agent_runtime_id"):
            # Update existing runtime
            runtime_id = existing["agent_runtime_id"]
            print(f"  Updating AgentCore Runtime: {config.name}")
            response = control.update_agent_runtime(
                agentRuntimeId=runtime_id,
                agentRuntimeArtifact=artifact,
                roleArn=role_arn,
                networkConfiguration=network_config,
                **({"description": config.description} if config.description else {}),
                **({"environmentVariables": config.environment_variables} if config.environment_variables else {}),
            )
            created = False
        else:
            # Create new runtime
            runtime_name = f"strands-{config.name}"
            print(f"  Creating AgentCore Runtime: {runtime_name}")
            create_kwargs: dict = {
                "agentRuntimeName": runtime_name,
                "agentRuntimeArtifact": artifact,
                "roleArn": role_arn,
                "networkConfiguration": network_config,
                "protocolConfiguration": {"serverProtocol": "HTTP"},
            }
            if config.description:
                create_kwargs["description"] = config.description
            if config.environment_variables:
                create_kwargs["environmentVariables"] = config.environment_variables

            response = control.create_agent_runtime(**create_kwargs)
            created = True

        runtime_id = response["agentRuntimeId"]
        runtime_arn = response["agentRuntimeArn"]
        runtime_version = response.get("agentRuntimeVersion")

        # Step 5: Wait for runtime to be ready
        self._wait_for_runtime(control, runtime_id)

        # Step 6: Create endpoint (only for new deployments)
        endpoint_arn = None
        if existing and existing.get("agent_runtime_endpoint_arn"):
            endpoint_arn = existing["agent_runtime_endpoint_arn"]
        elif created:
            endpoint_arn = self._create_endpoint(control, runtime_id, config.name)

        # Step 7: Save state
        state = DeployState(
            target="agentcore",
            region=region,
            agent_runtime_id=runtime_id,
            agent_runtime_arn=runtime_arn,
            agent_runtime_version=runtime_version or "",
            agent_runtime_endpoint_arn=endpoint_arn or "",
            role_arn=role_arn,
            s3_bucket=bucket_name,
            s3_key=s3_key,
        )
        state_manager.save(config.name, state)

        return DeployResult(
            target="agentcore",
            name=config.name,
            region=region,
            created=created,
            agent_runtime_id=runtime_id,
            agent_runtime_arn=runtime_arn,
            agent_runtime_endpoint_arn=endpoint_arn,
            role_arn=role_arn,
        )

    def destroy(self, name: str, state_manager: "StateManager", region: str | None = None) -> None:
        """Tear down AgentCore resources for a deployment."""
        existing = state_manager.load(name)
        if not existing:
            print(f"  No deployment found for '{name}'")
            return

        deploy_region = region or existing.get("region", "us-east-1")
        control = boto3.client("bedrock-agentcore-control", region_name=deploy_region)

        runtime_id = existing.get("agent_runtime_id")
        if runtime_id:
            try:
                print(f"  Deleting AgentCore Runtime: {runtime_id}")
                control.delete_agent_runtime(agentRuntimeId=runtime_id)
            except Exception as e:
                logger.warning("Failed to delete runtime %s: %s", runtime_id, e)

        role_arn = existing.get("role_arn")
        if role_arn:
            self._delete_iam_role(role_arn)

        state_manager.delete(name)
        print(f"  Deployment '{name}' destroyed")

    def _resolve_region(self, agent: "Agent", config: "DeployConfig") -> str:
        """Resolve the AWS region from config, agent model, or environment."""
        if config.region:
            return config.region

        # Try to get region from the agent's model
        model = agent.model
        if hasattr(model, "config") and hasattr(model.config, "get"):
            model_region: str | None = model.config.get("region")
            if model_region:
                return model_region

        # Fall back to boto3 session / environment
        session = boto3.Session()
        region = session.region_name or os.environ.get("AWS_REGION") or "us-east-1"
        return region

    def _get_account_id(self) -> str:
        """Get the AWS account ID."""
        sts = boto3.client("sts")
        account_id: str = sts.get_caller_identity()["Account"]
        return account_id

    def _ensure_iam_role(self, name: str, existing: DeployState | None, account_id: str) -> str:
        """Create or reuse the IAM execution role for AgentCore."""
        if existing and existing.get("role_arn"):
            existing_role: str = existing["role_arn"]
            print(f"  Using existing IAM role: {existing_role}")
            return existing_role

        role_name = f"strands-{name}-agentcore-role"
        iam = boto3.client("iam")

        print(f"  Creating IAM role: {role_name}")
        try:
            response = iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(AGENTCORE_TRUST_POLICY),
                Description=f"Execution role for Strands agent '{name}' on AgentCore",
                Tags=[
                    {"Key": "CreatedBy", "Value": "strands-agents-deploy"},
                    {"Key": "AgentName", "Value": name},
                ],
            )
            role_arn: str = response["Role"]["Arn"]
        except iam.exceptions.EntityAlreadyExistsException:
            response = iam.get_role(RoleName=role_name)
            found_arn: str = response["Role"]["Arn"]
            print(f"  IAM role already exists: {found_arn}")
            return found_arn

        # Attach inline execution policy
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName="strands-agentcore-execution",
            PolicyDocument=json.dumps(AGENTCORE_EXECUTION_POLICY),
        )

        # Wait for IAM propagation
        print("  Waiting for IAM role propagation...")
        time.sleep(IAM_PROPAGATION_DELAY_SECONDS * IAM_PROPAGATION_MAX_ATTEMPTS)

        return role_arn

    def _delete_iam_role(self, role_arn: str) -> None:
        """Best-effort deletion of an IAM role."""
        iam = boto3.client("iam")
        role_name = role_arn.split("/")[-1]
        try:
            # Delete inline policies first
            policies = iam.list_role_policies(RoleName=role_name)
            for policy_name in policies.get("PolicyNames", []):
                iam.delete_role_policy(RoleName=role_name, PolicyName=policy_name)
            iam.delete_role(RoleName=role_name)
            print(f"  Deleted IAM role: {role_name}")
        except Exception as e:
            logger.warning("Failed to delete IAM role %s: %s", role_name, e)

    def _wait_for_runtime(self, control_client: Any, runtime_id: str) -> None:
        """Poll until the runtime reaches READY status."""
        for _attempt in range(RUNTIME_POLL_MAX_ATTEMPTS):
            response = control_client.get_agent_runtime(agentRuntimeId=runtime_id)
            status = response.get("status", "UNKNOWN")

            if status == "READY":
                print("  Runtime ready!")
                return
            elif status in ("CREATE_FAILED", "UPDATE_FAILED"):
                reason = response.get("failureReason", "Unknown reason")
                raise DeployTargetException("agentcore", f"Runtime failed with status {status}: {reason}")

            print(f"  Waiting for runtime to be ready... ({status})")
            time.sleep(RUNTIME_POLL_INTERVAL_SECONDS)

        raise DeployTargetException(
            "agentcore",
            f"Runtime did not reach READY status within {RUNTIME_POLL_MAX_ATTEMPTS * RUNTIME_POLL_INTERVAL_SECONDS}s",
        )

    def _create_endpoint(self, control_client: Any, runtime_id: str, name: str) -> str:
        """Create an endpoint for the runtime."""
        endpoint_name = f"strands-{name}-endpoint"
        print(f"  Creating endpoint: {endpoint_name}")
        response = control_client.create_agent_runtime_endpoint(
            agentRuntimeId=runtime_id,
            name=endpoint_name,
        )
        endpoint_arn: str = response["agentRuntimeEndpointArn"]
        return endpoint_arn
