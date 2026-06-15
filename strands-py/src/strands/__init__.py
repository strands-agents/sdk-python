"""A framework for building, deploying, and managing AI agents."""

from . import agent, models, telemetry, types
from .agent.agent import Agent
from .agent.base import AgentBase
from .event_loop._retry import ModelRetryStrategy
from .interventions import InterventionHandler
from .plugins import MultiAgentPlugin, Plugin
from .sandbox import (
    ExecutionResult,
    FileInfo,
    OutputFile,
    PosixShellSandbox,
    Sandbox,
    StreamChunk,
    StreamType,
)
from .sandbox.errors import SandboxPathNotFoundError, SandboxTimeoutError
from .tools.decorator import tool
from .types._snapshot import Snapshot
from .types.tools import ToolContext
from .vended_plugins.skills import AgentSkills, Skill

__all__ = [
    "Agent",
    "AgentBase",
    "AgentSkills",
    "ExecutionResult",
    "FileInfo",
    "InterventionHandler",
    "agent",
    "models",
    "ModelRetryStrategy",
    "MultiAgentPlugin",
    "OutputFile",
    "Plugin",
    "PosixShellSandbox",
    "Sandbox",
    "SandboxPathNotFoundError",
    "SandboxTimeoutError",
    "Skill",
    "Snapshot",
    "StreamChunk",
    "StreamType",
    "tool",
    "ToolContext",
    "types",
    "telemetry",
]
