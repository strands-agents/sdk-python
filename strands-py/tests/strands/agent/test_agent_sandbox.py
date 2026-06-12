"""Tests for the ``Agent.sandbox`` getter.

Mirrors ``strands-ts/src/sandbox/__tests__/default.test.node.ts``: a configured
sandbox is returned as-is, and an unconfigured agent (or ``sandbox=False``) falls
back to the host default. The TS browser-throw case has no Python analog (core
Python always runs on a host), so it is omitted.
"""

from strands import Agent
from strands.sandbox import NotASandboxLocalEnvironment, default_sandbox


def test_returns_configured_sandbox():
    sandbox = NotASandboxLocalEnvironment()
    assert Agent(model="nonsense", sandbox=sandbox).sandbox is sandbox


def test_falls_back_to_host_default_when_unconfigured():
    assert isinstance(Agent(model="nonsense").sandbox, NotASandboxLocalEnvironment)


def test_sandbox_false_treated_as_unconfigured():
    assert isinstance(Agent(model="nonsense", sandbox=False).sandbox, NotASandboxLocalEnvironment)


def test_default_sandbox_is_host_default():
    assert isinstance(default_sandbox(), NotASandboxLocalEnvironment)


def test_host_default_is_shared_across_agents():
    # Mirrors the TS oracle's module-level singleton: unconfigured agents share one instance.
    assert Agent(model="nonsense").sandbox is Agent(model="nonsense", sandbox=False).sandbox is default_sandbox()
