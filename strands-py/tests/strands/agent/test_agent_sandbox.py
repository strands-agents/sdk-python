"""Tests for the ``Agent.sandbox`` getter.

A configured sandbox is returned as-is; an unconfigured agent falls back to a per-agent
:class:`NotASandboxLocalEnvironment`. Unlike the TS oracle's module-level singleton, each
agent gets its own default instance so state can't leak across agents. The TS browser-throw
case has no Python analog (core Python always runs on a host), so it is omitted.
"""

import pytest

from strands import Agent
from strands.sandbox.not_a_sandbox_local_environment import NotASandboxLocalEnvironment


def test_returns_configured_sandbox():
    sandbox = NotASandboxLocalEnvironment()
    assert Agent(model="nonsense", sandbox=sandbox).sandbox is sandbox


def test_falls_back_to_host_default_when_unconfigured():
    assert isinstance(Agent(model="nonsense").sandbox, NotASandboxLocalEnvironment)


def test_default_is_stable_within_one_agent():
    # All reads on a single agent return the same instance (tools share one sandbox).
    agent = Agent(model="nonsense")
    assert agent.sandbox is agent.sandbox


def test_default_is_not_shared_across_agents():
    # Each agent gets its own host default, so a future stateful default can't leak across agents.
    assert Agent(model="nonsense").sandbox is not Agent(model="nonsense").sandbox


def test_invalid_sandbox_raises_type_error():
    # Bad input is rejected at construction rather than silently falling back to the host default.
    with pytest.raises(TypeError, match="sandbox must be a Sandbox instance"):
        Agent(model="nonsense", sandbox="not-a-sandbox")
