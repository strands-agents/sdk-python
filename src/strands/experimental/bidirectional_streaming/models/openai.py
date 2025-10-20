"""OpenAI Realtime API provider for Strands bidirectional streaming.

Provides real-time audio and text communication through OpenAI's Realtime API
with WebSocket connections, voice activity detection, and function calling.
"""

import asyncio
import base64
import json
import logging
import uuid
from typing import Any, AsyncIterable, Dict, List, Optional

import websockets
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosed

from ....types._events import ToolResultEvent, ToolUseStreamEvent
from ....types.content import Messages
from ....types.tools import ToolSpec, ToolUse
from ..types.bidirectional_streaming import (
    AudioInputEvent,
    AudioStreamEvent,
    BidirectionalConnectionEndEvent,
    BidirectionalConnectionStartEvent,
    BidirectionalStreamEvent,
    ErrorEvent,
    ImageInputEvent,
    InterruptionEvent,
    ModalityUsage,
    MultimodalUsage,
    SessionEndEvent,
    SessionStartEvent,
    TranscriptStreamEvent,
    TurnCompleteEvent,
    TurnStartEvent,
)
from ..utils.event_logger import EventLogger
from .bidirectional_model import BidirectionalModel, BidirectionalModelSession

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


class OpenAIRealtimeSession(BidirectionalModelSession):
    """OpenAI Realtime API session for real-time audio/text streaming.
    
    Manages WebSocket connection to OpenAI's Realtime API with automatic VAD,
    function calling, and event conversion to Strands format.
    """

    def __init__(self, websocket: WebSocketClientProtocol, config: Dict[str, Any]) -> None:
        """Initialize OpenAI Realtime session."""
        self.websocket = websocket
        self.config = config
        self.session_id = str(uuid.uuid4())
        self._active = True
        
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._response_task: Optional[asyncio.Task] = None
        self._function_call_buffer: Dict[str, Dict[str, Any]] = {}
        self._max_buffer_size = 100
        self._error_count = 0
        self._max_errors = 10
        self.event_logger = EventLogger("openai")
        
        logger.debug("OpenAI Realtime session initialized: %s", self.session_id)

    def _require_active(self) -> bool:
        """Check if session is active."""
        return self._active

    async def _create_conversation_item(self, item_data: dict) -> None:
        """Create conversation item and trigger response."""
        await self._send_openai_event({"type": "conversation.item.create", "item": item_data})
        await self._send_openai_event({"type": "response.create"})

    async def initialize(
        self,
        system_prompt: str | None = None,
        tools: list[ToolSpec] | None = None,
        messages: Messages | None = None,
    ) -> None:
        """Initialize session with configuration."""
        try:
            session_config = self._build_session_config(system_prompt, tools)
            await self._send_openai_event({"type": "session.update", "session": session_config})
            
            if messages:
                await self._add_conversation_history(messages)
            
            self._response_task = asyncio.create_task(self._process_responses())
            logger.info("OpenAI Realtime session initialized successfully")
            
        except Exception as e:
            logger.error("Error during OpenAI Realtime initialization: %s", e)
            raise

    def _build_session_config(self, system_prompt: Optional[str], tools: Optional[List[ToolSpec]]) -> Dict[str, Any]:
        """Build session configuration for OpenAI Realtime API."""
        config = DEFAULT_SESSION_CONFIG.copy()
        
        if system_prompt:
            config["instructions"] = system_prompt
        
        if tools:
            config["tools"] = self._convert_tools_to_openai_format(tools)
        
        custom_config = self.config.get("session", {})
        supported_params = {
            "type", "output_modalities", "instructions", "voice", "audio", 
            "tools", "tool_choice", "input_audio_format", "output_audio_format",
            "input_audio_transcription", "turn_detection"
        }
        
        for key, value in custom_config.items():
            if key in supported_params:
                config[key] = value
            else:
                logger.warning("Ignoring unsupported session parameter: %s", key)
        
        return config

    def _convert_tools_to_openai_format(self, tools: List[ToolSpec]) -> List[Dict[str, Any]]:
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
            
            await self._send_openai_event(conversation_item)

    async def _process_responses(self) -> None:
        """Process incoming WebSocket messages."""
        logger.debug("OpenAI Realtime response processor started")
        
        try:
            async for message in self.websocket:
                if not self._active:
                    break
                
                try:
                    event = json.loads(message)
                    
                    # Log incoming event
                    self.event_logger.log_incoming("openai_raw", event)
                    
                    await self._event_queue.put(event)
                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse OpenAI event: %s", e)
                    continue
                    
        except ConnectionClosed:
            logger.debug("OpenAI Realtime WebSocket connection closed")
        except Exception as e:
            self._error_count += 1
            logger.warning("OpenAI Realtime response error (%d/%d): %s", self._error_count, self._max_errors, e)
            if self._error_count >= self._max_errors:
                logger.error("Max error count reached, stopping response processor")
            else:
                # Continue processing if under error limit
                logger.debug("Continuing after error (will retry)")
                return
        finally:
            self._active = False
            logger.debug("OpenAI Realtime response processor stopped")

    async def receive_events(self) -> AsyncIterable[BidirectionalStreamEvent]:
        """Receive OpenAI events and convert to Strands format."""
        # Emit SessionStartEvent
        session_start = SessionStartEvent(
            session_id=self.session_id,
            model=self.config.get("model", DEFAULT_MODEL),
            capabilities=["audio", "tools"]
        )
        yield session_start
        
        try:
            while self._active:
                try:
                    openai_event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                    provider_event = self._convert_openai_event(openai_event)
                    if provider_event:
                        yield provider_event
                except asyncio.TimeoutError:
                    continue
                    
        except Exception as e:
            logger.error("Error receiving OpenAI Realtime event: %s", e)
            # Emit ErrorEvent
            error_event = ErrorEvent(
                error=e,
                code="receive_error",
                details={"exception_type": type(e).__name__}
            )
            yield {"error": error_event}
        finally:
            # Emit SessionEndEvent
            session_end = SessionEndEvent(reason="client_disconnect")
            yield {"session_end": session_end}

    def _convert_openai_event(self, openai_event: Dict[str, Any]) -> Optional[BidirectionalStreamEvent]:
        """Convert OpenAI events to Strands format.
        
        Dispatches to specific handler methods for complex event types.
        """
        event_type = openai_event.get("type")
        
        # Simple events handled inline
        if event_type == "session.created":
            logger.debug("OpenAI session.created event (already handled)")
            return None
        
        if event_type == "response.created":
            response_id = openai_event.get("response", {}).get("id", str(uuid.uuid4()))
            return {"turn_start": TurnStartEvent(turn_id=response_id)}
        
        if event_type == "response.output_audio.delta":
            audio_data = base64.b64decode(openai_event["delta"])
            audio_stream = AudioStreamEvent(
                audio=audio_data,
                format="pcm",
                sample_rate=24000,
                channels=1
            )
            return {"audio_stream": audio_stream}
        
        # Assistant audio transcript -> TranscriptStreamEvent
        if event_type == "response.output_audio_transcript.delta":
            text = openai_event.get("delta", "")
            if text.strip():
                transcript = TranscriptStreamEvent(
                    text=text,
                    source="assistant",
                    is_final=False
                )
                return {"transcript_stream": transcript}
            return None
        
        if event_type == "response.output_audio_transcript.done":
            text = openai_event.get("transcript", "")
            if text.strip():
                transcript = TranscriptStreamEvent(
                    text=text,
                    source="assistant",
                    is_final=True
                )
                return {"transcript_stream": transcript}
            return None
        
        # User transcription -> TranscriptStreamEvent
        if event_type == "conversation.item.input_audio_transcription.delta":
            transcript_delta = openai_event.get("delta", "")
            if transcript_delta.strip():
                transcript = TranscriptStreamEvent(
                    text=transcript_delta,
                    source="user",
                    is_final=False
                )
                return {"transcript_stream": transcript}
            return None
        
        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript = openai_event.get("transcript", "")
            if transcript.strip():
                transcript_event = TranscriptStreamEvent(
                    text=transcript,
                    source="user",
                    is_final=True
                )
                return {"transcript_stream": transcript_event}
            return None
        
        if event_type == "conversation.item.input_audio_transcription.failed":
            error_info = openai_event.get("error", {})
            error_message = error_info.get("message", "Unknown transcription error")
            logger.warning("OpenAI transcription failed: %s", error_message)
            
            error_exception = Exception(error_message)
            error_event = ErrorEvent(
                error=error_exception,
                code="transcription_failed",
                details=error_info
            )
            return {"error": error_event}
        
        # Function call processing - delegate to helper methods
        if event_type == "response.function_call_arguments.delta":
            return self._handle_function_call_delta(openai_event)
        
        if event_type == "response.function_call_arguments.done":
            return self._handle_function_call_done(openai_event)
        
        # Voice activity detection -> InterruptionEvent
        if event_type == "input_audio_buffer.speech_started":
            interruption = InterruptionEvent(
                reason="user_speech",
                turn_id=None  # OpenAI doesn't provide turn_id in this event
            )
            return {"interruption": interruption}
        
        if event_type == "input_audio_buffer.speech_stopped":
            logger.debug("OpenAI speech stopped (no event emitted)")
            return None
        
        if event_type == "input_audio_buffer.timeout_triggered":
            logger.debug("OpenAI audio buffer timeout")
            return None
        
        # Response done -> TurnCompleteEvent
        if event_type == "response.done":
            response = openai_event.get("response", {})
            response_id = response.get("id", "unknown")
            status = response.get("status", "completed")
            
            # Map OpenAI status to stop_reason
            stop_reason_map = {
                "completed": "complete",
                "cancelled": "interrupted",
                "failed": "error",
                "incomplete": "error"
            }
            stop_reason = stop_reason_map.get(status, "complete")
            
            # Check if there are function calls (tool use)
            output = response.get("output", [])
            has_function_call = any(item.get("type") == "function_call" for item in output)
            if has_function_call:
                stop_reason = "tool_use"
            
            turn_complete = TurnCompleteEvent(
                turn_id=response_id,
                stop_reason=stop_reason
            )
            return {"turn_complete": turn_complete}
        
        # Usage tracking -> UsageEvent
        if event_type == "response.output_item.done":
            item = openai_event.get("item", {})
            # Check if usage information is available
            # OpenAI may provide usage in rate_limits.updated or in response metadata
            # For now, we'll handle it in rate_limits.updated
            return None
        
        if event_type == "rate_limits.updated":
            rate_limits = openai_event.get("rate_limits", [])
            
            # Extract token usage from rate limits
            total_input = 0
            total_output = 0
            details: List[ModalityUsage] = []
            
            for limit in rate_limits:
                name = limit.get("name", "")
                remaining = limit.get("remaining", 0)
                limit_val = limit.get("limit", 0)
                
                # Calculate tokens used (limit - remaining)
                used = limit_val - remaining if limit_val > 0 else 0
                
                if "input_tokens" in name:
                    if "text" in name:
                        details.append({"modality": "text", "input_tokens": used, "output_tokens": 0})
                        total_input += used
                    elif "audio" in name:
                        details.append({"modality": "audio", "input_tokens": used, "output_tokens": 0})
                        total_input += used
                elif "output_tokens" in name:
                    if "text" in name:
                        # Find or update existing text entry
                        for d in details:
                            if d["modality"] == "text":
                                d["output_tokens"] = used
                                break
                        else:
                            details.append({"modality": "text", "input_tokens": 0, "output_tokens": used})
                        total_output += used
                    elif "audio" in name:
                        # Find or update existing audio entry
                        for d in details:
                            if d["modality"] == "audio":
                                d["output_tokens"] = used
                                break
                        else:
                            details.append({"modality": "audio", "input_tokens": 0, "output_tokens": used})
                        total_output += used
            
            if total_input > 0 or total_output > 0:
                total_tokens = total_input + total_output
                usage_event = MultimodalUsage(
                    input_tokens=total_input,
                    output_tokens=total_output,
                    total_tokens=total_tokens,
                    modality_details=details if details else None
                )
                return {"usage": usage_event}
            return None
        
        # Lifecycle events (log only)
        if event_type == "conversation.item.retrieve":
            item = openai_event.get("item", {})
            logger.debug("OpenAI conversation item retrieved: %s", item.get("id"))
            return None
        
        if event_type == "conversation.item.added":
            logger.debug("OpenAI conversation item added: %s", openai_event.get("item", {}).get("id"))
            return None
            
        if event_type == "conversation.item.done":
            logger.debug("OpenAI conversation item done: %s", openai_event.get("item", {}).get("id"))
            return None
        
        if event_type in ["response.output_item.added", "response.content_part.added", "response.content_part.done"]:
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
        
        if event_type in ["input_audio_buffer.committed", "input_audio_buffer.cleared",
                           "session.updated"]:
            logger.debug("OpenAI %s event", event_type)
            return None
        
        # Error event -> ErrorEvent
        if event_type == "error":
            error_data = openai_event.get("error", {})
            error_code = error_data.get("code", "unknown_error")
            error_message = error_data.get("message", "Unknown error occurred")
            
            error_exception = Exception(error_message)
            error_event = ErrorEvent(
                error=error_exception,
                code=error_code,
                details=error_data
            )
            return {"error": error_event}
        
        # Unhandled event types
        logger.debug("Unhandled OpenAI event type: %s", event_type)
        return None

    async def send(self, event: AudioInputEvent | ImageInputEvent | ToolResultEvent) -> None:
        """Send input event to the model using unified interface.
        
        This is the preferred method for sending all types of input to the model.
        
        Args:
            event: Input event to send (AudioInputEvent, ImageInputEvent, or ToolResultEvent).
            
        Raises:
            NotImplementedError: If the provider doesn't support the event type.
            ValueError: If the event type is not recognized.
        """
        if not self._require_active():
            return
        
        if isinstance(event, AudioInputEvent):
            # Convert to OpenAI's input_audio_buffer.append format
            self.event_logger.log_outgoing("audio_input", {
                "format": event.format,
                "sample_rate": event.sample_rate,
                "channels": event.channels,
                "audio": f"<{len(event.audio)} bytes>"
            })
            
            audio_base64 = base64.b64encode(event.audio).decode("utf-8")
            await self._send_openai_event({"type": "input_audio_buffer.append", "audio": audio_base64})
            
        elif isinstance(event, ImageInputEvent):
            raise NotImplementedError("OpenAI Realtime API does not support image input")
            
        elif isinstance(event, ToolResultEvent):
            # Convert to OpenAI's conversation.item.create with function_call_output
            tool_result = event.tool_result
            tool_use_id = tool_result.get("toolUseId", "unknown")
            logger.debug("OpenAI tool result send: %s", tool_use_id)
            
            # Extract the actual result content
            content = tool_result.get("content", [])
            if content and isinstance(content, list) and len(content) > 0:
                result_text = content[0].get("text", "")
            else:
                result_text = json.dumps(tool_result)
            
            item_data = {
                "type": "function_call_output",
                "call_id": tool_use_id,
                "output": result_text
            }
            await self._create_conversation_item(item_data)
            
        else:
            raise ValueError(f"Unsupported event type: {type(event)}")

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

    async def _send_openai_event(self, event: Dict[str, Any]) -> None:
        """Send event to OpenAI via WebSocket."""
        try:
            message = json.dumps(event)
            await self.websocket.send(message)
            logger.debug("Sent OpenAI event: %s", event.get("type"))
        except Exception as e:
            logger.error("Error sending OpenAI event: %s", e)
            raise
    
    def _handle_function_call_delta(self, openai_event: Dict[str, Any]) -> None:
        """Handle streaming function call arguments."""
        call_id = openai_event.get("call_id")
        delta = openai_event.get("delta", "")
        if call_id:
            if call_id not in self._function_call_buffer:
                if len(self._function_call_buffer) >= self._max_buffer_size:
                    logger.warning("Function call buffer full, clearing oldest entries")
                    oldest_keys = list(self._function_call_buffer.keys())[:10]
                    for key in oldest_keys:
                        del self._function_call_buffer[key]
                self._function_call_buffer[call_id] = {"call_id": call_id, "name": "", "arguments": delta}
            else:
                self._function_call_buffer[call_id]["arguments"] += delta
        return None
    
    def _handle_function_call_done(self, openai_event: Dict[str, Any]) -> Optional[BidirectionalStreamEvent]:
        """Handle completed function call."""
        call_id = openai_event.get("call_id")
        if call_id and call_id in self._function_call_buffer:
            function_call = self._function_call_buffer[call_id]
            try:
                # OpenAI streams tool arguments, but we emit complete tool use on .done
                delta: Dict[str, Any] = {
                    "toolUse": {
                        "input": function_call["arguments"]  # JSON string
                    }
                }
                
                current_tool_use = {
                    "toolUseId": call_id,
                    "name": function_call["name"],
                    "input": function_call["arguments"]  # JSON string
                }
                
                tool_use_event = ToolUseStreamEvent(delta, current_tool_use)
                del self._function_call_buffer[call_id]
                return {"tool_use": tool_use_event}
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Error parsing function arguments for %s: %s", call_id, e)
                del self._function_call_buffer[call_id]
                error_event = ErrorEvent(
                    error=e,
                    code="function_parse_error",
                    details={"call_id": call_id, "error": str(e)}
                )
                return {"error": error_event}
        return None


class OpenAIRealtimeBidirectionalModel(BidirectionalModel):
    """OpenAI Realtime API provider for Strands bidirectional streaming.
    
    Provides real-time audio/text communication through OpenAI's Realtime API
    with WebSocket connections, voice activity detection, and function calling.
    """

    def __init__(
        self, 
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        **config: Any
    ) -> None:
        """Initialize OpenAI Realtime bidirectional model."""
        self.model = model
        self.api_key = api_key
        self.config = config
        
        import os
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        logger.debug("OpenAI Realtime bidirectional model initialized: %s", model)

    async def create_bidirectional_connection(
        self,
        system_prompt: Optional[str] = None,
        tools: Optional[List[ToolSpec]] = None,
        messages: Optional[Messages] = None,
        **kwargs: Any,
    ) -> BidirectionalModelSession:
        """Create bidirectional connection to OpenAI Realtime API."""
        logger.info("Creating OpenAI Realtime connection...")
        
        try:
            url = f"{OPENAI_REALTIME_URL}?model={self.model}"
            
            headers = [("Authorization", f"Bearer {self.api_key}")]
            if "organization" in self.config:
                headers.append(("OpenAI-Organization", self.config["organization"]))
            if "project" in self.config:
                headers.append(("OpenAI-Project", self.config["project"]))
            
            websocket = await websockets.connect(url, additional_headers=headers)
            logger.info("WebSocket connected successfully")
            
            session = OpenAIRealtimeSession(websocket, self.config)
            await session.initialize(system_prompt, tools, messages)
            
            logger.info("OpenAI Realtime connection established")
            return session
            
        except Exception as e:
            logger.error("OpenAI connection error: %s", e)
            raise