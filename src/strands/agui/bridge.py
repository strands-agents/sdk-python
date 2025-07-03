"""Strands to AG-UI Protocol Bridge with State Management.

This module converts Strands agent events to AG-UI protocol events,
including full state management support for AG-UI compatible frontends.
"""

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import uuid4

# Use relative import to avoid module name conflict
from .state_tools import StrandsStateManager, get_state_manager, setup_agent_state_management

logger = logging.getLogger(__name__)


class AGUIEventType(str, Enum):
    """AG-UI Protocol Event Types."""

    # Run lifecycle events
    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"

    # Message events
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"

    # Tool events
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    TOOL_CALL_END = "TOOL_CALL_END"

    # State events - KEY for AG-UI compatibility!
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"

    # Step events
    STEP_STARTED = "STEP_STARTED"
    STEP_FINISHED = "STEP_FINISHED"

    # Custom events
    CUSTOM = "CUSTOM"
    RAW = "RAW"


class StrandsAGUIBridge:
    """Bridge that converts Strands agent events to AG-UI protocol events with state management."""

    def __init__(self, agent: Any, state_manager: Optional[StrandsStateManager] = None) -> None:
        """Initialize the Strands AG-UI bridge.

        Args:
            agent: The Strands agent to bridge
            state_manager: Optional state manager, will use global one if not provided
        """
        self.agent = agent
        self.state_manager = state_manager or get_state_manager()
        self.current_run_id: Optional[str] = None
        self.current_thread_id: Optional[str] = None
        self.current_message_id: Optional[str] = None
        self.active_tool_calls: Dict[str, Dict[str, Any]] = {}
        self.message_started = False

        # Set up state change callback
        self.state_manager.add_callback(self._on_state_change)
        self._state_change_queue: List[Dict[str, Any]] = []

    async def stream_agui_events(
        self,
        prompt: str,
        thread_id: Optional[str] = None,
        run_id: Optional[str] = None,
        initial_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream AG-UI protocol events from a Strands agent execution."""
        self.current_run_id = run_id or str(uuid4())
        self.current_thread_id = thread_id or str(uuid4())
        self.current_message_id = str(uuid4())
        self.message_started = False

        # Set initial state if provided
        if initial_state:
            self.state_manager.set_state(initial_state)

        try:
            # Emit run started event
            yield self._create_agui_event(
                event_type=AGUIEventType.RUN_STARTED,
                data={"thread_id": self.current_thread_id, "run_id": self.current_run_id},
            )

            # Emit initial state snapshot
            current_state = self.state_manager.get_state()
            if current_state:
                yield self._create_agui_event(event_type=AGUIEventType.STATE_SNAPSHOT, data={"snapshot": current_state})

            # Stream agent execution and convert events
            async for strands_event in self.agent.stream_async(prompt, **kwargs):
                agui_events = self._convert_strands_event_to_agui(strands_event)
                for agui_event in agui_events:
                    yield agui_event

                # Emit any queued state changes
                while self._state_change_queue:
                    state_event = self._state_change_queue.pop(0)
                    yield state_event

            # Emit message end if we started one
            if self.message_started:
                yield self._create_agui_event(
                    event_type=AGUIEventType.TEXT_MESSAGE_END, data={"message_id": self.current_message_id}
                )

            # Emit run finished event
            yield self._create_agui_event(
                event_type=AGUIEventType.RUN_FINISHED,
                data={"thread_id": self.current_thread_id, "run_id": self.current_run_id},
            )

        except Exception as e:
            logger.error("Error in Strands-AG-UI bridge: %s", e)
            yield self._create_agui_event(
                event_type=AGUIEventType.RUN_ERROR, data={"message": str(e), "code": type(e).__name__}
            )

    def _convert_strands_event_to_agui(self, strands_event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert a Strands event to one or more AG-UI events."""
        agui_events = []

        # Handle text content streaming
        if "data" in strands_event and strands_event["data"]:
            # Start message if not already started
            if not self.message_started:
                agui_events.append(
                    self._create_agui_event(
                        event_type=AGUIEventType.TEXT_MESSAGE_START,
                        data={"message_id": self.current_message_id, "role": "assistant"},
                    )
                )
                self.message_started = True

            # Add content event
            agui_events.append(
                self._create_agui_event(
                    event_type=AGUIEventType.TEXT_MESSAGE_CONTENT,
                    data={"message_id": self.current_message_id, "delta": strands_event["data"]},
                )
            )

        # Handle tool execution
        if "current_tool_use" in strands_event:
            tool_use = strands_event["current_tool_use"]
            if tool_use and tool_use.get("toolUseId"):
                tool_id = tool_use["toolUseId"]
                tool_name = tool_use.get("name", "unknown")

                # Track tool start
                if tool_id not in self.active_tool_calls:
                    self.active_tool_calls[tool_id] = {"name": tool_name, "started": True}
                    agui_events.append(
                        self._create_agui_event(
                            event_type=AGUIEventType.TOOL_CALL_START,
                            data={
                                "tool_call_id": tool_id,
                                "tool_call_name": tool_name,
                                "parent_message_id": self.current_message_id,
                            },
                        )
                    )

                # Handle tool arguments
                if "input" in tool_use:
                    tool_input = tool_use["input"]
                    if isinstance(tool_input, dict):
                        tool_input = json.dumps(tool_input)
                    elif not isinstance(tool_input, str):
                        tool_input = str(tool_input)

                    agui_events.append(
                        self._create_agui_event(
                            event_type=AGUIEventType.TOOL_CALL_ARGS, data={"tool_call_id": tool_id, "delta": tool_input}
                        )
                    )

        # Handle completion events
        if strands_event.get("complete", False):
            # End any active tool calls
            for tool_id in list(self.active_tool_calls.keys()):
                agui_events.append(
                    self._create_agui_event(event_type=AGUIEventType.TOOL_CALL_END, data={"tool_call_id": tool_id})
                )
                del self.active_tool_calls[tool_id]

        return agui_events

    def _on_state_change(self, new_state: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Handle state changes from the state manager."""
        # Queue state delta event
        if updates:
            state_event = self._create_agui_event(
                event_type=AGUIEventType.STATE_DELTA, data={"delta": self._dict_to_json_patch(updates)}
            )
            self._state_change_queue.append(state_event)

    def _dict_to_json_patch(self, updates: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert dictionary updates to JSON Patch format."""
        patches = []
        for key, value in updates.items():
            if value is None:
                patches.append({"op": "remove", "path": f"/{key}"})
            else:
                patches.append({"op": "replace", "path": f"/{key}", "value": value})
        return patches

    def _create_agui_event(self, event_type: AGUIEventType, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a properly formatted AG-UI protocol event."""
        return {"type": event_type.value, "timestamp": int(datetime.now().timestamp() * 1000), **data}


class StrandsAGUIEndpoint:
    """HTTP endpoint that serves AG-UI events from Strands agents with state management."""

    def __init__(self, agents: Dict[str, Any]) -> None:
        """Initialize the Strands AG-UI endpoint.

        Args:
            agents: Dictionary of agent name to agent instance
        """
        self.agents = agents
        self.bridges: Dict[str, StrandsAGUIBridge] = {}

        # Create bridges for each agent
        for name, agent in agents.items():
            self.bridges[name] = StrandsAGUIBridge(agent)

    async def handle_request(self, request_data: Dict[str, Any]) -> AsyncIterator[str]:
        """Handle AG-UI protocol HTTP request and stream SSE responses."""
        agent_name = request_data.get("agent")
        messages = request_data.get("messages", [])
        thread_id = request_data.get("threadId")
        run_id = request_data.get("runId")
        frontend_state = request_data.get("state", {})

        if not agent_name or agent_name not in self.agents:
            yield f"data: {json.dumps({'type': 'RUN_ERROR', 'message': 'Agent not found'})}\n\n"
            return

        bridge = self.bridges[agent_name]

        # Extract the latest user message
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if not user_messages:
            yield f"data: {json.dumps({'type': 'RUN_ERROR', 'message': 'No user message found'})}\n\n"
            return

        latest_message = user_messages[-1]
        latest_prompt = latest_message.get("content", "")
        if isinstance(latest_prompt, list) and latest_prompt:
            latest_prompt = latest_prompt[0].get("text", "")

        try:
            async for event in bridge.stream_agui_events(
                prompt=latest_prompt, thread_id=thread_id, run_id=run_id, initial_state=frontend_state
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error("Error in AG-UI endpoint: %s", e)
            error_event = {"type": "RUN_ERROR", "message": str(e), "code": type(e).__name__}
            yield f"data: {json.dumps(error_event)}\n\n"


def create_strands_agui_setup(
    agents: Dict[str, Any], initial_states: Optional[Dict[str, Dict[str, Any]]] = None
) -> StrandsAGUIEndpoint:
    """Create a complete Strands + AG-UI setup with state management.

    Args:
        agents: Dictionary mapping agent names to agent instances
        initial_states: Optional dictionary mapping agent names to their initial states

    Returns:
        A configured StrandsAGUIEndpoint instance
    """
    # Set up state management for each agent
    for name, agent in agents.items():
        initial_state = initial_states.get(name, {}) if initial_states else {}
        setup_agent_state_management(agent, initial_state)

    return StrandsAGUIEndpoint(agents)
