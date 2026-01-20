"""A2A client for communicating with remote A2A agents."""

import logging
import uuid
from typing import Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)


def build_agentcore_url(agent_arn: str) -> str:
    """Build the invocation URL from an AgentCore Runtime ARN.

    Args:
        agent_arn: The ARN of the AgentCore Runtime agent.

    Returns:
        The full invocation URL for the agent.

    Raises:
        ValueError: If the ARN format is invalid.
    """
    if not agent_arn.startswith("arn:aws:bedrock-agentcore:"):
        raise ValueError(
            f"Invalid AgentCore ARN format. Expected 'arn:aws:bedrock-agentcore:...' but got '{agent_arn}'"
        )

    parts = agent_arn.split(":")
    if len(parts) < 6:
        raise ValueError(f"Invalid ARN format: {agent_arn}")

    region = parts[3]
    encoded_arn = quote(agent_arn, safe="")

    return f"https://bedrock-agentcore.{region}.amazonaws.com/runtimes/{encoded_arn}/invocations"


def extract_region_from_arn(agent_arn: str) -> str:
    """Extract the AWS region from an AgentCore Runtime ARN.

    Args:
        agent_arn: The ARN of the AgentCore Runtime agent.

    Returns:
        The AWS region (e.g., "us-east-1").

    Raises:
        ValueError: If the ARN format is invalid.
    """
    parts = agent_arn.split(":")
    if len(parts) < 4:
        raise ValueError(f"Invalid ARN format: {agent_arn}")
    return parts[3]


class A2AError(Exception):
    """Exception raised for A2A protocol errors."""

    def __init__(self, code: int, message: str):
        """Initialize an A2AError."""
        self.code = code
        self.message = message
        super().__init__(f"A2A Error {code}: {message}")


class A2AClient:
    """Client for communicating with remote A2A agents.

    This client implements the A2A protocol for sending tasks to remote agents.
    It supports synchronous APIs for sending tasks and retrieving agent metadata.
    """

    def __init__(
        self,
        url: str,
        auth: Optional[httpx.Auth] = None,
        timeout: float = 300.0,
        headers: Optional[dict[str, str]] = None,
    ):
        """Initialize an A2A client.

        Args:
            url: The base URL of the A2A agent.
            auth: Optional authentication object (e.g., SigV4 auth).
            timeout: Request timeout in seconds. Defaults to 300.
            headers: Optional additional HTTP headers.
        """
        self._url = url.rstrip("/")
        self._auth = auth
        self._timeout = timeout
        self._headers = headers or {}
        self._agent_card: Optional[dict] = None

    @classmethod
    def from_agentcore_arn(
        cls,
        agent_arn: str,
        region: Optional[str] = None,
        timeout: float = 300.0,
    ) -> "A2AClient":
        """Create a client from an AgentCore Runtime ARN with IAM authentication.

        Args:
            agent_arn: The ARN of the AgentCore Runtime agent.
            region: AWS region for authentication. If None, extracted from ARN.
            timeout: Request timeout in seconds. Defaults to 300.

        Returns:
            An A2AClient configured for the specified AgentCore agent.

        Raises:
            ImportError: If mcp-proxy-for-aws is not installed.
        """
        try:
            from mcp_proxy_for_aws.sigv4_helper import SigV4HTTPXAuth, create_aws_session
        except ImportError as e:
            raise ImportError(
                "mcp-proxy-for-aws is required for IAM authentication. "
                "Please install it with: pip install mcp-proxy-for-aws"
            ) from e

        url = build_agentcore_url(agent_arn)

        if region is None:
            region = extract_region_from_arn(agent_arn)

        session = create_aws_session()
        credentials = session.get_credentials()
        auth = SigV4HTTPXAuth(credentials, "bedrock-agentcore", region)

        logger.debug("Created A2AClient with SigV4 auth for region=%s", region)

        return cls(url=url, auth=auth, timeout=timeout)

    @classmethod
    def from_url(
        cls,
        url: str,
        timeout: float = 300.0,
    ) -> "A2AClient":
        """Create a client from a URL without authentication.

        Args:
            url: The URL of the A2A agent.
            timeout: Request timeout in seconds. Defaults to 300.

        Returns:
            An A2AClient configured for the specified URL.
        """
        return cls(url=url, auth=None, timeout=timeout)

    @property
    def url(self) -> str:
        """Get the base URL of the A2A agent."""
        return self._url

    def get_agent_card(self, force_refresh: bool = False) -> dict:
        """Get the agent card (metadata).

        Args:
            force_refresh: If True, bypass the cache and fetch fresh data.

        Returns:
            The agent card as a dictionary.
        """
        if self._agent_card is not None and not force_refresh:
            return self._agent_card

        with httpx.Client(auth=self._auth, timeout=self._timeout) as http_client:
            response = http_client.get(
                f"{self._url}/.well-known/agent.json",
                headers=self._headers,
            )
            response.raise_for_status()
            self._agent_card = response.json()
            return self._agent_card

    def send_task(
        self,
        message: str,
        session_id: Optional[str] = None,
    ) -> str:
        """Send a task and wait for the result.

        Args:
            message: The message to send to the agent.
            session_id: Optional session ID for conversation continuity.

        Returns:
            The agent's response as a string.

        Raises:
            A2AError: If the agent returns an error response.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        task_id = str(uuid.uuid4())
        request_body = self._build_task_request(task_id, session_id, message)

        with httpx.Client(auth=self._auth, timeout=self._timeout) as http_client:
            response = http_client.post(
                self._url,
                json=request_body,
                headers={
                    "Content-Type": "application/json",
                    **self._headers,
                },
            )
            response.raise_for_status()
            result = response.json()

        return self._extract_text_from_response(result)

    def _build_task_request(self, task_id: str, session_id: str, message: str) -> dict:
        """Build a JSON-RPC request for tasks/send."""
        return {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "id": task_id,
            "params": {
                "id": task_id,
                "sessionId": session_id,
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": message}],
                },
            },
        }

    def _extract_text_from_response(self, response: dict) -> str:
        """Extract text content from a JSON-RPC response."""
        if "error" in response:
            error = response["error"]
            raise A2AError(
                code=error.get("code", -1),
                message=error.get("message", "Unknown error"),
            )

        result = response.get("result", {})
        artifacts = result.get("artifacts", [])
        texts = []

        for artifact in artifacts:
            parts = artifact.get("parts", [])
            for part in parts:
                if part.get("kind") == "text":
                    texts.append(part.get("text", ""))

        return "".join(texts)
