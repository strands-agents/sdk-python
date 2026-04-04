"""Local state management for tracking deployed resources."""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, TypedDict

from ._constants import STATE_DIR_NAME
from ._exceptions import DeployStateException

logger = logging.getLogger(__name__)

STATE_VERSION = "1"


class DeployState(TypedDict, total=False):
    """State for a single deployment."""

    target: str
    region: str
    # AgentCore fields
    agent_runtime_id: str
    agent_runtime_arn: str
    agent_runtime_version: str
    agent_runtime_endpoint_arn: str
    role_arn: str
    s3_bucket: str
    s3_key: str
    # Timestamps
    last_deployed: str
    created_at: str


class StateManager:
    """Manages .strands_deploy/state.json for tracking deployed AWS resources.

    Uses atomic writes (tmp file + os.replace) to prevent corruption.
    """

    def __init__(self, base_dir: str | None = None):
        self._base_dir = base_dir or os.getcwd()
        self._state_dir = os.path.join(self._base_dir, STATE_DIR_NAME)
        self._state_file = os.path.join(self._state_dir, "state.json")

    def load(self, deployment_name: str) -> DeployState | None:
        """Load state for a named deployment.

        Returns None if no state exists for this deployment.
        """
        all_state = self._read_all()
        deployments: dict[str, DeployState] = all_state.get("deployments", {})
        return deployments.get(deployment_name)

    def save(self, deployment_name: str, state: DeployState) -> None:
        """Save state for a named deployment with atomic write."""
        state["last_deployed"] = datetime.now(timezone.utc).isoformat()
        if "created_at" not in state:
            existing = self.load(deployment_name)
            if existing and "created_at" in existing:
                state["created_at"] = existing["created_at"]
            else:
                state["created_at"] = state["last_deployed"]

        all_state = self._read_all()
        all_state.setdefault("deployments", {})[deployment_name] = state
        self._write_all(all_state)
        logger.debug("state_file=<%s> | saved deployment '%s'", self._state_file, deployment_name)

    def delete(self, deployment_name: str) -> None:
        """Remove state for a named deployment."""
        all_state = self._read_all()
        deployments = all_state.get("deployments", {})
        if deployment_name in deployments:
            del deployments[deployment_name]
            self._write_all(all_state)

    def _read_all(self) -> dict[str, Any]:
        """Read the full state file. Returns empty state if file doesn't exist."""
        if not os.path.exists(self._state_file):
            return {"version": STATE_VERSION, "deployments": {}}
        try:
            with open(self._state_file, encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)
                return data
        except (json.JSONDecodeError, OSError) as e:
            raise DeployStateException(f"Failed to read state file {self._state_file}: {e}") from e

    def _write_all(self, data: dict[str, Any]) -> None:
        """Atomic write using tmp file + os.replace."""
        data["version"] = STATE_VERSION
        os.makedirs(self._state_dir, exist_ok=True)
        tmp = f"{self._state_file}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self._state_file)
        except OSError as e:
            # Clean up tmp file on failure
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise DeployStateException(f"Failed to write state file: {e}") from e
