"""Fixtures for SSH sandbox integration tests.

Provides a session-scoped :class:`~strands.sandbox.ssh.SshSandbox` pointed at
the shared ``ssh-ec2`` test instance via an SSM port-forward. The entire module
is skipped when the infrastructure isn't available — individual tests never
carry skip logic.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import secrets
import shutil
import socket
import stat
import subprocess
import tempfile
import time
from collections.abc import Iterator

import pytest

from strands.sandbox.ssh import SshSandbox
from tests_integ._ssm import read_ssm_parameter, resolve_ssm_parameters, ssm_parameter_path

logger = logging.getLogger(__name__)

_SSH_USER = "ec2-user"
_TUNNEL_READY_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# SSM tunnel lifecycle
# ---------------------------------------------------------------------------


class SsmTunnel:
    """Manages an SSM Session Manager port-forward with proper cleanup.

    Captures the session ID from the process's stdout so cleanup can explicitly
    call ``aws ssm terminate-session`` rather than relying solely on SIGTERM
    (which may not fire if the runner is OOM-killed or times out).
    """

    def __init__(self, instance_id: str, local_port: int, region: str) -> None:
        self.instance_id = instance_id
        self.local_port = local_port
        self.region = region
        self._process: subprocess.Popen | None = None
        self._session_id: str | None = None

    def open(self) -> bool:
        """Start the tunnel and wait for the local port to accept connections.

        Returns:
            True if the tunnel is ready, False if it failed to come up.
        """
        self._process = subprocess.Popen(
            [
                "aws",
                "ssm",
                "start-session",
                "--region",
                self.region,
                "--target",
                self.instance_id,
                "--document-name",
                "AWS-StartPortForwardingSession",
                "--parameters",
                json.dumps({"portNumber": ["22"], "localPortNumber": [str(self.local_port)]}),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        atexit.register(self.close)

        output_lines: list[str] = []
        deadline = time.monotonic() + _TUNNEL_READY_TIMEOUT
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                break
            if _port_open(self.local_port):
                self._extract_session_id(output_lines)
                return True
            # Non-blocking read of stdout to capture session ID
            if self._process.stdout and self._process.stdout.readable():
                import selectors

                sel = selectors.DefaultSelector()
                sel.register(self._process.stdout, selectors.EVENT_READ)
                if sel.select(timeout=0.5):
                    line = self._process.stdout.readline()
                    if line:
                        output_lines.append(line)
                sel.close()
            else:
                time.sleep(0.5)

        self._extract_session_id(output_lines)
        return False

    def close(self) -> None:
        """Terminate the session and kill the local process."""
        if self._session_id:
            subprocess.run(
                ["aws", "ssm", "terminate-session", "--region", self.region, "--session-id", self._session_id],
                capture_output=True,
            )
            self._session_id = None
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    def _extract_session_id(self, lines: list[str]) -> None:
        for line in lines:
            match = re.search(r"SessionId[:\s]+(\S+)", line)
            if match:
                self._session_id = match.group(1)
                return


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _tooling_available() -> bool:
    return shutil.which("aws") is not None and shutil.which("session-manager-plugin") is not None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ssh_sandbox() -> Iterator[SshSandbox]:
    """Session-scoped SSH sandbox connected to the ssh-ec2 instance.

    Resolves infrastructure from SSM, opens an SSM tunnel, creates an isolated
    remote working directory, and yields a ready-to-use :class:`SshSandbox`.
    Skips the entire session when infrastructure isn't available.
    """
    if not _tooling_available():
        pytest.skip("aws CLI or session-manager-plugin not on PATH")

    resolved = resolve_ssm_parameters(
        {
            "instance_id": ssm_parameter_path("ssh-ec2", "instance-id"),
            "private_key_parameter_name": ssm_parameter_path("ssh-ec2", "private-key-parameter-name"),
        },
        overrides={
            "instance_id": "STRANDS_TEST_SSH_INSTANCE_ID",
            "private_key_parameter_name": "STRANDS_TEST_SSH_KEY_PARAM",
        },
    )
    instance_id = resolved["instance_id"]
    key_param_name = resolved["private_key_parameter_name"]
    if not instance_id or not key_param_name:
        pytest.skip("ssh-ec2 SSM parameters not available")

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    private_key = read_ssm_parameter(key_param_name, with_decryption=True, region=region)
    if not private_key:
        pytest.skip("ssh-ec2 private key not readable")

    key_dir = tempfile.mkdtemp(prefix="strands-ssh-ec2-")
    key_path = os.path.join(key_dir, "key.pem")
    with open(key_path, "w", encoding="ascii") as f:
        f.write(private_key if private_key.endswith("\n") else private_key + "\n")
    os.chmod(key_path, stat.S_IRUSR | stat.S_IWUSR)

    local_port = _free_port()
    tunnel = SsmTunnel(instance_id, local_port, region)
    if not tunnel.open():
        tunnel.close()
        pytest.skip("SSM port-forward did not come up")

    # Isolate from concurrent runners (Py + TS hit the same instance simultaneously).
    working_dir = f"/home/{_SSH_USER}/strands-integ-ssh-{secrets.token_hex(4)}"

    sandbox = SshSandbox(
        host=f"{_SSH_USER}@127.0.0.1",
        working_dir=working_dir,
        identity_file=key_path,
        port=local_port,
        allow_unknown_hosts=True,
    )

    try:
        mkdir = subprocess.run(
            [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "BatchMode=yes",
                "-p",
                str(local_port),
                "-i",
                key_path,
                "--",
                f"{_SSH_USER}@127.0.0.1",
                f"mkdir -p {working_dir}",
            ],
            capture_output=True,
            text=True,
        )
        if mkdir.returncode != 0:
            tunnel.close()
            pytest.skip(f"could not create remote working dir: {mkdir.stderr.strip()}")

        yield sandbox

        # Best-effort cleanup of remote files while tunnel is still up.
        subprocess.run(
            [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "BatchMode=yes",
                "-p",
                str(local_port),
                "-i",
                key_path,
                "--",
                f"{_SSH_USER}@127.0.0.1",
                f"rm -rf {working_dir}",
            ],
            capture_output=True,
        )
    finally:
        tunnel.close()
        shutil.rmtree(key_dir, ignore_errors=True)
