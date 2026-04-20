"""Experimental checkpoint types for durable agent execution.

This module is experimental and subject to change in future revisions without notice.

Checkpoints enable crash-resilient agent execution by breaking the agent loop into
discrete steps (one model call or one tool execution per step). Each step returns either
a Checkpoint (keep going) or an AgentResult (done).
"""

from .invoke import invoke_with_checkpoint
from .types import Checkpoint, CheckpointPosition

__all__ = ["Checkpoint", "CheckpointPosition", "invoke_with_checkpoint"]
