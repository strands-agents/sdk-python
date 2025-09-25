"""Trajectory capture system for agent execution traces.

This module provides functionality to capture and store agent execution trajectories
for training purposes, including tool calls, model responses, and outcomes.
"""

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from ..hooks import HookProvider, HookRegistry
from ..hooks.events import AfterInvocationEvent, BeforeInvocationEvent, MessageAddedEvent
from ..types.content import Message, Messages
from ..types.tools import ToolResult, ToolUse
from ..types.streaming import StopReason

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryStep:
    """A single step in an agent trajectory.
    
    Attributes:
        step_id: Unique identifier for this step
        timestamp: When this step occurred
        step_type: Type of step (model_inference, tool_call, tool_result, etc.)
        input_data: Input data for this step
        output_data: Output data from this step
        metadata: Additional metadata about the step
    """
    
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    step_type: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryData:
    """Complete trajectory data for an agent execution.
    
    Attributes:
        trajectory_id: Unique identifier for this trajectory
        agent_id: ID of the agent that generated this trajectory
        session_id: Session identifier if applicable
        start_time: When the trajectory started
        end_time: When the trajectory ended
        steps: List of steps in the trajectory
        final_result: Final result of the agent execution
        reward: Reward value for this trajectory (if applicable)
        metadata: Additional metadata about the trajectory
    """
    
    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    session_id: Optional[str] = None
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    steps: List[TrajectoryStep] = field(default_factory=list)
    final_result: Optional[Dict[str, Any]] = None
    reward: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, step: TrajectoryStep) -> None:
        """Add a step to the trajectory."""
        self.steps.append(step)
    
    def finalize(self, final_result: Optional[Dict[str, Any]] = None) -> None:
        """Mark the trajectory as complete."""
        self.end_time = datetime.now(timezone.utc)
        if final_result:
            self.final_result = final_result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trajectory to dictionary for serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert trajectory to JSON string."""
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False)


class TrajectoryCapture(HookProvider):
    """Captures agent execution trajectories for training purposes.
    
    This class implements the HookProvider interface to automatically capture
    agent execution data during normal operation. It can be added to any agent
    to enable trajectory collection.
    """
    
    def __init__(
        self,
        storage_backend: Optional[Any] = None,
        capture_tool_calls: bool = True,
        capture_model_responses: bool = True,
        capture_metadata: bool = True,
    ):
        """Initialize trajectory capture.
        
        Args:
            storage_backend: Optional storage backend for persisting trajectories
            capture_tool_calls: Whether to capture tool call details
            capture_model_responses: Whether to capture model response details
            capture_metadata: Whether to capture additional metadata
        """
        self.storage_backend = storage_backend
        self.capture_tool_calls = capture_tool_calls
        self.capture_model_responses = capture_model_responses
        self.capture_metadata = capture_metadata
        
        # Current trajectory being captured
        self.current_trajectory: Optional[TrajectoryData] = None
        self.agent_id: Optional[str] = None
        
        logger.debug(
            "trajectory_capture=<%s>, capture_tool_calls=<%s>, capture_model_responses=<%s> | initialized",
            self.__class__.__name__,
            capture_tool_calls,
            capture_model_responses,
        )
    
    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register hooks for trajectory capture."""
        # Start capturing at the beginning of each invocation
        registry.add_callback(
            BeforeInvocationEvent,
            self._on_before_invocation
        )
        
        # Capture messages as they're added
        registry.add_callback(
            MessageAddedEvent,
            self._on_message_added
        )
        
        # Finalize trajectory at the end of each invocation
        registry.add_callback(
            AfterInvocationEvent,
            self._on_after_invocation
        )
    
    def _on_before_invocation(self, event: BeforeInvocationEvent) -> None:
        """Handle before invocation event to start trajectory capture."""
        self.agent_id = event.agent.agent_id
        
        # Start new trajectory
        self.current_trajectory = TrajectoryData(
            agent_id=self.agent_id,
            session_id=getattr(event.agent, 'session_id', None),
        )
        
        # Add initial step
        initial_step = TrajectoryStep(
            step_type="invocation_start",
            input_data={
                "agent_id": self.agent_id,
                "system_prompt": event.agent.system_prompt,
                "tools": [tool.tool_name for tool in event.agent.tool_registry.registry.values()],
            },
            metadata={
                "agent_name": event.agent.name,
                "model_id": getattr(event.agent.model, 'model_id', None),
            }
        )
        self.current_trajectory.add_step(initial_step)
        
        logger.debug(
            "trajectory_id=<%s>, agent_id=<%s> | started trajectory capture",
            self.current_trajectory.trajectory_id,
            self.agent_id,
        )
    
    def _on_message_added(self, event: MessageAddedEvent) -> None:
        """Handle message added event to capture conversation steps."""
        if not self.current_trajectory:
            return
        
        message = event.message
        step_type = f"message_{message['role']}"
        
        # Extract content for capture
        content_data = {}
        tool_calls = []
        tool_results = []
        
        for content_block in message.get("content", []):
            if "text" in content_block:
                content_data["text"] = content_block["text"]
            elif "toolUse" in content_block and self.capture_tool_calls:
                tool_calls.append(content_block["toolUse"])
            elif "toolResult" in content_block and self.capture_tool_calls:
                tool_results.append(content_block["toolResult"])
        
        # Create step
        step = TrajectoryStep(
            step_type=step_type,
            input_data={
                "role": message["role"],
                "content": content_data,
            },
            output_data={
                "tool_calls": tool_calls,
                "tool_results": tool_results,
            },
            metadata={
                "message_length": len(str(message)),
                "content_blocks": len(message.get("content", [])),
            }
        )
        
        self.current_trajectory.add_step(step)
        
        logger.debug(
            "trajectory_id=<%s>, step_type=<%s>, step_id=<%s> | captured message step",
            self.current_trajectory.trajectory_id,
            step_type,
            step.step_id,
        )
    
    def _on_after_invocation(self, event: AfterInvocationEvent) -> None:
        """Handle after invocation event to finalize trajectory."""
        if not self.current_trajectory:
            return
        
        # Add final step
        final_step = TrajectoryStep(
            step_type="invocation_end",
            input_data={
                "agent_id": self.agent_id,
            },
            output_data={
                "stop_reason": getattr(event, 'stop_reason', None),
                "success": not hasattr(event, 'error') or event.error is None,
            },
            metadata={
                "total_steps": len(self.current_trajectory.steps),
                "execution_time": (
                    datetime.now(timezone.utc) - self.current_trajectory.start_time
                ).total_seconds(),
            }
        )
        
        self.current_trajectory.add_step(final_step)
        self.current_trajectory.finalize()
        
        # Store trajectory if backend is available
        if self.storage_backend:
            self._store_trajectory(self.current_trajectory)
        
        logger.debug(
            "trajectory_id=<%s>, total_steps=<%d> | completed trajectory capture",
            self.current_trajectory.trajectory_id,
            len(self.current_trajectory.steps),
        )
        
        # Reset for next trajectory
        self.current_trajectory = None
    
    def _store_trajectory(self, trajectory: TrajectoryData) -> None:
        """Store trajectory using the configured backend."""
        try:
            if hasattr(self.storage_backend, 'store_trajectory'):
                self.storage_backend.store_trajectory(trajectory)
            elif hasattr(self.storage_backend, 'write'):
                # Assume it's a file-like object
                self.storage_backend.write(trajectory.to_json() + "\n")
            else:
                logger.warning(
                    "storage_backend=<%s> | unsupported storage backend type",
                    type(self.storage_backend).__name__,
                )
        except Exception as e:
            logger.error(
                "trajectory_id=<%s>, error=<%s> | failed to store trajectory",
                trajectory.trajectory_id,
                e,
                exc_info=True,
            )
    
    def get_current_trajectory(self) -> Optional[TrajectoryData]:
        """Get the currently being captured trajectory."""
        return self.current_trajectory
    
    def set_reward(self, reward: float) -> None:
        """Set reward for the current trajectory."""
        if self.current_trajectory:
            self.current_trajectory.reward = reward
            logger.debug(
                "trajectory_id=<%s>, reward=<%f> | set trajectory reward",
                self.current_trajectory.trajectory_id,
                reward,
            )
