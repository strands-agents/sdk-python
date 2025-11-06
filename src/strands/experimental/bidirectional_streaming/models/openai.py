"""OpenAI Realtime API provider for Strands bidirectional streaming.

Provides real-time audio and text communication through OpenAI's Realtime API
with WebSocket connections, voice activity detection, and function calling.
"""

import asyncio
import base64
import json
import logging
import os
import uuid
from typing import AsyncIterable, Union

import websockets
from websockets.exceptions import ConnectionClosed

from ....types.content import Messages
from ....types.tools import ToolResult, ToolSpec, ToolUse
from ....types._events import ToolResultEvent, ToolUseStreamEvent
from ..types.bidirectional_streaming import (
    AudioInputEvent,
    AudioStreamEvent,
    ConnectionCloseEvent,
    ConnectionStartEvent,
    ErrorEvent,
    ImageInputEvent,
    InterruptionEvent,
    UsageEvent,
    OutputEvent,
    TextInputEvent,
    TranscriptStreamEvent,
    TurnCompleteEvent,
    TurnStartEvent,
)
from .bidirectional_model import BidirectionalModel

logger = logging.getLogger(__name__)

# OpenAI Realtime API configuration
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-realtime"

AUDIO_FORMAT = {"type": "audio/pcm", "rate": 24000}

DEFAULT_SESSION_CONFIG = {
    "type": "realtime",
    "instructions": "You are a helpful assistant. Please speak in English and keep your responses clear and concise.",
    "output_modalities": ["audio"],
    "audio": {
        "input": {
            "format": AUDIO_FORMAT,
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500,
            }
        },
        "output": {"format": AUDIO_FORMAT, "voice": "alloy"},
    },
}


class OpenAIRealtimeModel(BidirectionalModel):
    """OpenAI Realtime API implementation for bidirectional streaming.
    
    Combines model configuration and connection state in a single class.
    Manages WebSocket connection to OpenAI's Realtime API with automatic VAD,
    function calling, and event conversion to Strands format.
    """

    def __init__(
        self, 
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        session_config: dict[str, any] | None = None,
        **kwargs
    ) -> None:
        """Initialize OpenAI Realtime bidirectional model.
        
        Args:
            model: OpenAI model identifier (default: gpt-realtime).
            api_key: OpenAI API key for authentication.
            organization: OpenAI organization ID for API requests.
            project: OpenAI project ID for API requests.
            session_config: Session configuration parameters (e.g., voice, turn_detection, modalities).
            **kwargs: Reserved for future parameters.
        """
        # Model configuration
        self.model = model
        self.api_key = api_key
        self.organization = organization
        self.project = project
        self.session_config = session_config or {}
        
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Connection state (initialized in connect())
        self.websocket = None
        self.connection_id = None
        self._active = False
        
        self._event_queue = None
        self._response_task = None
        self._function_call_buffer = {}
        
        logger.debug("OpenAI Realtime bidirectional model initialized: %s", model)

    async def connect(
        self,
        system_prompt: str | None = None,
        tools: list[ToolSpec] | None = None,
        messages: Messages | None = None,
        **kwargs,
    ) -> None:
        """Establish bidirectional connection to OpenAI Realtime API.
        
        Args:
            system_prompt: System instructions for the model.
            tools: List of tools available to the model.
            messages: Conversation history to initialize with.
            **kwargs: Additional configuration options.
        """
        if self._active:
            raise RuntimeError("Connection already active. Close the existing connection before creating a new one.")
        
        logger.info("Creating OpenAI Realtime connection...")
        
        try:
            # Initialize connection state
            self.connection_id = str(uuid.uuid4())
            self._active = True
            self._event_queue = asyncio.Queue()
            self._function_call_buffer = {}
            
            # Establish WebSocket connection
            url = f"{OPENAI_REALTIME_URL}?model={self.model}"
            
            headers = [("Authorization", f"Bearer {self.api_key}")]
            if self.organization:
                headers.append(("OpenAI-Organization", self.organization))
            if self.project:
                headers.append(("OpenAI-Project", self.project))
            
            self.websocket = await websockets.connect(url, additional_headers=headers)
            logger.info("WebSocket connected successfully")
            
            # Configure session
            session_config = self._build_session_config(system_prompt, tools)
            await self._send_event({"type": "session.update", "session": session_config})
            
            # Add conversation history if provided
            if messages:
                await self._add_conversation_history(messages)
            
            # Start background response processor
            self._response_task = asyncio.create_task(self._process_responses())
            logger.info("OpenAI Realtime connection established")
            
        except Exception as e:
            self._active = False
            logger.error("OpenAI connection error: %s", e)
            raise

    def _require_active(self) -> bool:
        """Check if session is active."""
        return self._active

    def _create_text_event(self, text: str, role: str) -> TranscriptStreamEvent:
        """Create standardized transcript event."""
        return TranscriptStreamEvent(
            text=text,
            source="user" if role == "user" else "assistant",
            is_final=True
        )

    def _create_voice_activity_event(self, activity_type: str) -> InterruptionEvent | None:
        """Create standardized interruption event for voice activity."""
        # Only speech_started triggers interruption
        if activity_type == "speech_started":
            return InterruptionEvent(reason="user_speech", turn_id=None)
        # Other voice activity events are logged but don't create events
        return None

    def _build_session_config(self, system_prompt: str | None, tools: list[ToolSpec] | None) -> dict:
        """Build session configuration for OpenAI Realtime API."""
        config = DEFAULT_SESSION_CONFIG.copy()
        
        if system_prompt:
            config["instructions"] = system_prompt
        
        if tools:
            config["tools"] = self._convert_tools_to_openai_format(tools)
        
        # Apply user-provided session configuration
        supported_params = {
            "type", "output_modalities", "instructions", "voice", "audio", 
            "tools", "tool_choice", "input_audio_format", "output_audio_format",
            "input_audio_transcription", "turn_detection"
        }
        
        for key, value in self.session_config.items():
            if key in supported_params:
                config[key] = value
            else:
                logger.warning("Ignoring unsupported session parameter: %s", key)
        
        return config

    def _convert_tools_to_openai_format(self, tools: list[ToolSpec]) -> list[dict]:
        """Convert Strands tool specifications to OpenAI Realtime API format."""
        openai_tools = []
        
        for tool in tools:
            input_schema = tool["inputSchema"]
            if "json" in input_schema:
                schema = json.loads(input_schema["json"]) if isinstance(input_schema["json"], str) else input_schema["json"]
            else:
                schema = input_schema
            
            # OpenAI Realtime API expects flat structure, not nested under "function"
            openai_tool = {
                "type": "function",
                "name": tool["name"],
                "description": tool["description"],
                "parameters": schema
            }
            openai_tools.append(openai_tool)
        
        return openai_tools

    async def _add_conversation_history(self, messages: Messages) -> None:
        """Add conversation history to the session."""
        for message in messages:
            conversation_item = {
                "type": "conversation.item.create",
                "item": {"type": "message", "role": message["role"], "content": []}
            }
            
            content = message.get("content", "")
            if isinstance(content, str):
                conversation_item["item"]["content"].append({"type": "input_text", "text": content})
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        conversation_item["item"]["content"].append({"type": "input_text", "text": item.get("text", "")})
            
            await self._send_event(conversation_item)

    async def _process_responses(self) -> None:
        """Process incoming WebSocket messages."""
        logger.debug("OpenAI Realtime response processor started")
        
        try:
            async for message in self.websocket:
                if not self._active:
                    break
                
                try:
                    event = json.loads(message)
                    await self._event_queue.put(event)
                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse OpenAI event: %s", e)
                    continue
                    
        except ConnectionClosed:
            logger.debug("OpenAI Realtime WebSocket connection closed")
        except Exception as e:
            logger.error("Error in OpenAI Realtime response processing: %s", e)
        finally:
            self._active = False
            logger.debug("OpenAI Realtime response processor stopped")

    async def receive(self) -> AsyncIterable[OutputEvent]:
        """Receive OpenAI events and convert to Strands TypedEvent format."""
        # Emit connection start event
        yield ConnectionStartEvent(
            connection_id=self.connection_id,
            model=self.model,
            capabilities=["audio", "tools"]
        )
        
        try:
            while self._active:
                try:
                    openai_event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                    for event in self._convert_openai_event(openai_event) or []: 
                        yield event
                except asyncio.TimeoutError:
                    continue
                    
        except Exception as e:
            logger.error("Error receiving OpenAI Realtime event: %s", e)
            yield ErrorEvent(error=e)
        finally:
            # Emit connection close event
            yield ConnectionCloseEvent(connection_id=self.connection_id, reason="complete")

    def _convert_openai_event(self, openai_event: dict[str, any]) -> list[OutputEvent] | None:
        """Convert OpenAI events to Strands TypedEvent format."""
        event_type = openai_event.get("type")
        
        # Turn start - response begins
        if event_type == "response.created":
            response = openai_event.get("response", {})
            response_id = response.get("id", str(uuid.uuid4()))
            return [TurnStartEvent(turn_id=response_id)]
        
        # Audio output
        elif event_type == "response.output_audio.delta":
            # Audio is already base64 string from OpenAI
            return [AudioStreamEvent(
                audio=openai_event["delta"],
                format="pcm",
                sample_rate=24000,
                channels=1
            )]
        
        # Assistant text output events - combine multiple similar events
        elif event_type in ["response.output_text.delta", "response.output_audio_transcript.delta"]:
            return [self._create_text_event(openai_event["delta"], "assistant")]
        
        # User transcription events - combine multiple similar events
        elif event_type in ["conversation.item.input_audio_transcription.delta", 
                           "conversation.item.input_audio_transcription.completed"]:
            text_key = "delta" if "delta" in event_type else "transcript"
            text = openai_event.get(text_key, "")
            return [self._create_text_event(text, "user")] if text.strip() else None
        
        elif event_type == "conversation.item.input_audio_transcription.segment":
            segment_data = openai_event.get("segment", {})
            text = segment_data.get("text", "")
            return [self._create_text_event(text, "user")] if text.strip() else None
        
        elif event_type == "conversation.item.input_audio_transcription.failed":
            error_info = openai_event.get("error", {})
            logger.warning("OpenAI transcription failed: %s", error_info.get("message", "Unknown error"))
            return None
        
        # Function call processing
        elif event_type == "response.function_call_arguments.delta":
            call_id = openai_event.get("call_id")
            delta = openai_event.get("delta", "")
            if call_id:
                if call_id not in self._function_call_buffer:
                    self._function_call_buffer[call_id] = {"call_id": call_id, "name": "", "arguments": delta}
                else:
                    self._function_call_buffer[call_id]["arguments"] += delta
            return None
        
        elif event_type == "response.function_call_arguments.done":
            call_id = openai_event.get("call_id")
            if call_id and call_id in self._function_call_buffer:
                function_call = self._function_call_buffer[call_id]
                try:
                    tool_use: ToolUse = {
                        "toolUseId": call_id,
                        "name": function_call["name"],
                        "input": json.loads(function_call["arguments"]) if function_call["arguments"] else {},
                    }
                    del self._function_call_buffer[call_id]
                    # Return ToolUseStreamEvent for consistency with standard agent
                    return [ToolUseStreamEvent(
                        delta={"toolUse": tool_use},
                        current_tool_use=tool_use
                    )]
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning("Error parsing function arguments for %s: %s", call_id, e)
                    del self._function_call_buffer[call_id]
            return None
        
        # Voice activity detection events - combine similar events using mapping
        elif event_type in ["input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped", 
                           "input_audio_buffer.timeout_triggered"]:
            # Map event types to activity types
            activity_map = {
                "input_audio_buffer.speech_started": "speech_started",
                "input_audio_buffer.speech_stopped": "speech_stopped", 
                "input_audio_buffer.timeout_triggered": "timeout"
            }
            event = self._create_voice_activity_event(activity_map[event_type])
            return [event] if event else None
        
        # Turn complete and usage - response finished
        elif event_type == "response.done":
            response = openai_event.get("response", {})
            response_id = response.get("id", "unknown")
            status = response.get("status", "completed")
            usage = response.get("usage")
            
            # Map OpenAI status to our stop_reason
            stop_reason_map = {
                "completed": "complete",
                "cancelled": "interrupted",
                "failed": "error",
                "incomplete": "interrupted"
            }
            
            # Build list of events to return
            events = []
            
            # Always add turn complete event
            events.append(TurnCompleteEvent(
                turn_id=response_id,
                stop_reason=stop_reason_map.get(status, "complete")
            ))
            
            # Add usage event if available
            if usage:
                input_details = usage.get("input_token_details", {})
                output_details = usage.get("output_token_details", {})
                
                # Build modality details
                modality_details = []
                
                # Text modality
                text_input = input_details.get("text_tokens", 0)
                text_output = output_details.get("text_tokens", 0)
                if text_input > 0 or text_output > 0:
                    modality_details.append({
                        "modality": "text",
                        "input_tokens": text_input,
                        "output_tokens": text_output
                    })
                
                # Audio modality
                audio_input = input_details.get("audio_tokens", 0)
                audio_output = output_details.get("audio_tokens", 0)
                if audio_input > 0 or audio_output > 0:
                    modality_details.append({
                        "modality": "audio",
                        "input_tokens": audio_input,
                        "output_tokens": audio_output
                    })
                
                # Image modality
                image_input = input_details.get("image_tokens", 0)
                if image_input > 0:
                    modality_details.append({
                        "modality": "image",
                        "input_tokens": image_input,
                        "output_tokens": 0
                    })
                
                # Cached tokens
                cached_tokens = input_details.get("cached_tokens", 0)
                
                # Add usage event
                events.append(UsageEvent(
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                    modality_details=modality_details if modality_details else None,
                    cache_read_input_tokens=cached_tokens if cached_tokens > 0 else None
                ))
            
            # Return list of events
            return events
        
        # Lifecycle events (log only) - combine multiple similar events
        elif event_type in ["conversation.item.retrieve", "conversation.item.added"]:
            item = openai_event.get("item", {})
            action = "retrieved" if "retrieve" in event_type else "added"
            logger.debug("OpenAI conversation item %s: %s", action, item.get("id"))
            return None
            
        elif event_type == "conversation.item.done":
            logger.debug("OpenAI conversation item done: %s", openai_event.get("item", {}).get("id"))
            return None
        
        # Response output events - combine similar events
        elif event_type in ["response.output_item.added", "response.output_item.done", 
                           "response.content_part.added", "response.content_part.done"]:
            item_data = openai_event.get("item") or openai_event.get("part")
            logger.debug("OpenAI %s: %s", event_type, item_data.get("id") if item_data else "unknown")
            
            # Track function call names from response.output_item.added
            if event_type == "response.output_item.added":
                item = openai_event.get("item", {})
                if item.get("type") == "function_call":
                    call_id = item.get("call_id")
                    function_name = item.get("name")
                    if call_id and function_name:
                        if call_id not in self._function_call_buffer:
                            self._function_call_buffer[call_id] = {"call_id": call_id, "name": function_name, "arguments": ""}
                        else:
                            self._function_call_buffer[call_id]["name"] = function_name
            return None
        
        # Session/buffer events - combine simple log-only events
        elif event_type in ["input_audio_buffer.committed", "input_audio_buffer.cleared",
                           "session.created", "session.updated"]:
            logger.debug("OpenAI %s event", event_type)
            return None
        
        elif event_type == "error":
            logger.error("OpenAI Realtime error: %s", openai_event.get("error", {}))
            return None
        
        else:
            logger.debug("Unhandled OpenAI event type: %s", event_type)
            return None

    async def send(
        self,
        content: Union[TextInputEvent, AudioInputEvent, ImageInputEvent, ToolResultEvent],
    ) -> None:
        """Unified send method for all content types. Sends the given content to OpenAI.
        
        Dispatches to appropriate internal handler based on content type.
        
        Args:
            content: Typed event (TextInputEvent, AudioInputEvent, ImageInputEvent, or ToolResultEvent).
        """
        if not self._require_active():
            return
        
        try:
            # Note: TypedEvent inherits from dict, so isinstance checks for TypedEvent must come first
            if isinstance(content, TextInputEvent):
                await self._send_text_content(content.text)
            elif isinstance(content, AudioInputEvent):
                await self._send_audio_content(content)
            elif isinstance(content, ImageInputEvent):
                # ImageInputEvent - not supported by OpenAI Realtime yet
                logger.warning("Image input not supported by OpenAI Realtime API")
            elif isinstance(content, ToolResultEvent):
                tool_result = content.get("tool_result")
                if tool_result:
                    await self._send_tool_result(tool_result)
            else:
                logger.warning(f"Unknown content type: {type(content).__name__}")
        except Exception as e:
            logger.error(f"Error sending content: {e}")
            raise  # Propagate exception for debugging in experimental code

    async def _send_audio_content(self, audio_input: AudioInputEvent) -> None:
        """Internal: Send audio content to OpenAI for processing."""
        # Audio is already base64 encoded in the event
        await self._send_event({"type": "input_audio_buffer.append", "audio": audio_input.audio})

    async def _send_text_content(self, text: str) -> None:
        """Internal: Send text content to OpenAI for processing."""
        item_data = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}]
        }
        await self._send_event({"type": "conversation.item.create", "item": item_data})
        await self._send_event({"type": "response.create"})

    async def _send_interrupt(self) -> None:
        """Internal: Send interruption signal to OpenAI."""
        await self._send_event({"type": "response.cancel"})

    async def _send_tool_result(self, tool_result: ToolResult) -> None:
        """Internal: Send tool result back to OpenAI."""
        tool_use_id = tool_result.get("toolUseId")
        
        logger.debug("OpenAI tool result send: %s", tool_use_id)
        
        # Extract result content
        result_data = {}
        if "content" in tool_result:
            # Extract text from content blocks
            for block in tool_result["content"]:
                if "text" in block:
                    result_data = block["text"]
                    break
        
        result_text = json.dumps(result_data) if not isinstance(result_data, str) else result_data
        
        item_data = {
            "type": "function_call_output",
            "call_id": tool_use_id,
            "output": result_text
        }
        await self._send_event({"type": "conversation.item.create", "item": item_data})
        await self._send_event({"type": "response.create"})

    async def close(self) -> None:
        """Close session and cleanup resources."""
        if not self._active:
            return
        
        logger.debug("OpenAI Realtime cleanup - starting connection close")
        self._active = False
        
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass
        
        try:
            await self.websocket.close()
        except Exception as e:
            logger.warning("Error closing OpenAI Realtime WebSocket: %s", e)
        
        logger.debug("OpenAI Realtime connection closed")

    async def _send_event(self, event: dict[str, any]) -> None:
        """Send event to OpenAI via WebSocket."""
        try:
            message = json.dumps(event)
            await self.websocket.send(message)
            logger.debug("Sent OpenAI event: %s", event.get("type"))
        except Exception as e:
            logger.error("Error sending OpenAI event: %s", e)
            raise


