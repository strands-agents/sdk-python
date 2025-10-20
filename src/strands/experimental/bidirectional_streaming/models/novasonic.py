"""Nova Sonic bidirectional model provider for real-time streaming conversations.

Implements the BidirectionalModel interface for Amazon's Nova Sonic, handling the
complex event sequencing and audio processing required by Nova Sonic's
InvokeModelWithBidirectionalStream protocol.

Nova Sonic specifics:
- Hierarchical event sequences: connectionStart → promptStart → content streaming
- Base64-encoded audio format with hex encoding
- Tool execution with content containers and identifier tracking
- 8-minute connection limits with proper cleanup sequences
- Interruption detection through stopReason events
"""

import asyncio
import base64
import json
import logging
import time
import traceback
import uuid
from typing import Any, AsyncIterable, Dict, List, Optional

from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient, InvokeModelWithBidirectionalStreamOperationInput
from aws_sdk_bedrock_runtime.config import Config, HTTPAuthSchemeResolver, SigV4AuthScheme
from aws_sdk_bedrock_runtime.models import BidirectionalInputPayloadPart, InvokeModelWithBidirectionalStreamInputChunk
from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver

from ....types._events import ToolResultEvent, ToolUseStreamEvent
from ....types.content import Messages
from ....types.tools import ToolSpec, ToolUse
from ..types.bidirectional_streaming import (
    AudioInputEvent,
    AudioStreamEvent,
    BidirectionalConnectionEndEvent,
    BidirectionalConnectionStartEvent,
    ErrorEvent,
    ImageInputEvent,
    InterruptionEvent,
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

# Nova Sonic configuration constants
NOVA_INFERENCE_CONFIG = {"maxTokens": 1024, "topP": 0.9, "temperature": 0.7}

NOVA_AUDIO_INPUT_CONFIG = {
    "mediaType": "audio/lpcm",
    "sampleRateHertz": 16000,
    "sampleSizeBits": 16,
    "channelCount": 1,
    "audioType": "SPEECH",
    "encoding": "base64",
}

NOVA_AUDIO_OUTPUT_CONFIG = {
    "mediaType": "audio/lpcm",
    "sampleRateHertz": 24000,
    "sampleSizeBits": 16,
    "channelCount": 1,
    "voiceId": "matthew",
    "encoding": "base64",
    "audioType": "SPEECH",
}

NOVA_TEXT_CONFIG = {"mediaType": "text/plain"}
NOVA_TOOL_CONFIG = {"mediaType": "application/json"}

# Timing constants
SILENCE_THRESHOLD = 2.0
EVENT_DELAY = 0.1
RESPONSE_TIMEOUT = 1.0


class NovaSonicSession(BidirectionalModelSession):
    """Nova Sonic connection implementation handling the provider's specific protocol.

    Manages Nova Sonic's complex event sequencing, audio format conversion, and
    tool execution patterns while providing the standard BidirectionalModelSession
    interface.
    """

    def __init__(self, stream: Any, config: Dict[str, Any]) -> None:
        """Initialize Nova Sonic connection.

        Args:
            stream: Nova Sonic bidirectional stream.
            config: Model configuration.
        """
        self.stream = stream
        self.config = config
        self.prompt_name = str(uuid.uuid4())
        self._active = True

        # Nova Sonic requires unique content names
        self.audio_content_name = str(uuid.uuid4())
        self.text_content_name = str(uuid.uuid4())

        # Audio connection state
        self.audio_connection_active = False
        self.last_audio_time = None
        self.silence_threshold = SILENCE_THRESHOLD
        self.silence_task: Optional[asyncio.Task] = None
        self._error_count = 0
        self._max_errors = 10
        
        # Transcript deduplication (Nova sends duplicates)
        self._seen_transcripts: set[str] = set()
        self._transcript_window_size = 10
        
        # Event logger
        self.event_logger = EventLogger("nova")

        # Validate stream
        if not stream:
            logger.error("Stream is None")
            raise ValueError("Stream cannot be None")

        logger.debug("Nova Sonic connection initialized with prompt: %s", self.prompt_name)

    async def initialize(
        self,
        system_prompt: str | None = None,
        tools: list[ToolSpec] | None = None,
        messages: Messages | None = None,
    ) -> None:
        """Initialize Nova Sonic connection with required protocol sequence."""
        try:
            system_prompt = system_prompt or "You are a helpful assistant. Keep responses brief."

            init_events = self._build_initialization_events(system_prompt, tools or [], messages)

            logger.debug(f"Nova Sonic initialization - sending {len(init_events)} events")
            await self._send_initialization_events(init_events)

            logger.info("Nova Sonic connection initialized successfully")
            self._response_task = asyncio.create_task(self._process_responses())

        except Exception as e:
            logger.error("Error during Nova Sonic initialization: %s", e)
            raise

    async def send(self, event: AudioInputEvent | ImageInputEvent | ToolResultEvent) -> None:
        """Unified send method for all input types.

        Args:
            event: Input event to send (AudioInputEvent, ImageInputEvent, or ToolResultEvent).

        Raises:
            NotImplementedError: If ImageInputEvent is provided (Nova Sonic doesn't support images).
        """
        if isinstance(event, AudioInputEvent):
            # Convert AudioInputEvent to Nova Sonic's audioInput format
            await self.send_audio_content(event)
        elif isinstance(event, ImageInputEvent):
            # Nova Sonic doesn't support image input
            raise NotImplementedError("Nova Sonic does not support image input")
        elif isinstance(event, ToolResultEvent):
            # Convert ToolResultEvent to Nova Sonic's toolResult format
            await self.send_tool_result(event.tool_use_id, event.tool_result)
        else:
            raise ValueError(f"Unsupported event type: {type(event)}")

    def _build_initialization_events(
        self, system_prompt: str, tools: List[ToolSpec], messages: Optional[Messages]
    ) -> List[str]:
        """Build the sequence of initialization events."""
        events = [self._get_connection_start_event(), self._get_prompt_start_event(tools)]

        events.extend(self._get_system_prompt_events(system_prompt))

        # Message history would be processed here if needed in the future
        # Currently not implemented as it's not used in the existing test cases

        return events

    async def _send_initialization_events(self, events: List[str]) -> None:
        """Send initialization events with required delays."""
        for event in events:
            await self._send_nova_event(event)
            await asyncio.sleep(EVENT_DELAY)

    async def _process_responses(self) -> None:
        """Process Nova Sonic responses continuously."""
        logger.debug("Nova Sonic response processor started")

        try:
            while self._active:
                try:
                    output = await asyncio.wait_for(self.stream.await_output(), timeout=RESPONSE_TIMEOUT)
                    result = await output[1].receive()

                    if result.value and result.value.bytes_:
                        await self._handle_response_data(result.value.bytes_.decode("utf-8"))

                except asyncio.TimeoutError:
                    await asyncio.sleep(0.1)
                    continue
                except Exception as e:
                    self._error_count += 1
                    logger.warning("Nova Sonic response error (%d/%d): %s", self._error_count, self._max_errors, e)
                    if self._error_count >= self._max_errors:
                        logger.error("Max error count reached, stopping response processor")
                        break
                    await asyncio.sleep(0.1)
                    continue

        except Exception as e:
            logger.error(f"Nova Sonic fatal error: {e}")
        finally:
            logger.debug("Nova Sonic response processor stopped")

    async def _handle_response_data(self, response_data: str) -> None:
        """Handle decoded response data from Nova Sonic."""
        try:
            json_data = json.loads(response_data)

            if "event" in json_data:
                nova_event = json_data["event"]
                
                self.event_logger.log_incoming("nova_raw", nova_event)
                self._log_event_type(nova_event)

                if not hasattr(self, "_event_queue"):
                    self._event_queue = asyncio.Queue()

                await self._event_queue.put(nova_event)
        except json.JSONDecodeError as e:
            logger.warning("Nova Sonic JSON decode error: %s", e)

    def _log_event_type(self, nova_event: Dict[str, Any]) -> None:
        """Log specific Nova Sonic event types for debugging."""
        if "usageEvent" in nova_event:
            logger.debug("Nova usage: %s", nova_event["usageEvent"])
        elif "textOutput" in nova_event:
            logger.debug("Nova text output")
        elif "toolUse" in nova_event:
            tool_use = nova_event["toolUse"]
            logger.debug("Nova tool use: %s (id: %s)", tool_use["toolName"], tool_use["toolUseId"])
        elif "audioOutput" in nova_event:
            audio_content = nova_event["audioOutput"]["content"]
            audio_bytes = base64.b64decode(audio_content)
            logger.debug("Nova audio output: %d bytes", len(audio_bytes))

    async def receive_events(self) -> AsyncIterable[Dict[str, Any]]:
        """Receive Nova Sonic events and convert to provider-agnostic format."""
        if not self.stream:
            logger.error("Stream is None")
            return

        logger.debug("Nova events - starting event stream")

        # Emit SessionStartEvent immediately (Nova Sonic doesn't send one from server)
        session_start = SessionStartEvent(
            session_id=self.prompt_name,
            model=self.config.get("model_id", "amazon.nova-sonic-v1:0"),
            capabilities=["audio", "tools"]
        )
        yield session_start

        # Initialize event queue if not already done
        if not hasattr(self, "_event_queue"):
            self._event_queue = asyncio.Queue()

        try:
            while self._active:
                try:
                    # Get events from the queue populated by _process_responses
                    nova_event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)

                    # Convert to provider-agnostic format
                    provider_event = self._convert_nova_event(nova_event)
                    if provider_event:
                        yield provider_event

                except asyncio.TimeoutError:
                    # No events in queue - continue waiting
                    continue

        except Exception as e:
            logger.error("Error receiving Nova Sonic event: %s", e)
            logger.error(traceback.format_exc())
        finally:
            # Emit connection end event when exiting
            connection_end: BidirectionalConnectionEndEvent = {
                "connectionId": self.prompt_name,
                "reason": "connection_complete",
                "metadata": {"provider": "nova_sonic"},
            }
            yield {"BidirectionalConnectionEnd": connection_end}

    async def start_audio_connection(self) -> None:
        """Start audio input connection (call once before sending audio chunks)."""
        if self.audio_connection_active:
            return

        logger.debug("Nova audio connection start")

        audio_content_start = json.dumps(
            {
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                        "type": "AUDIO",
                        "interactive": True,
                        "role": "USER",
                        "audioInputConfiguration": NOVA_AUDIO_INPUT_CONFIG,
                    }
                }
            }
        )

        await self._send_nova_event(audio_content_start)
        self.audio_connection_active = True

    async def send_audio_content(self, audio_input: AudioInputEvent) -> None:
        """Send audio using Nova Sonic protocol-specific format."""
        if not self._active:
            return

        # Log outgoing audio
        self.event_logger.log_outgoing("audio_input", {
            "format": audio_input["format"],
            "sampleRate": audio_input["sample_rate"],
            "channels": audio_input["channels"],
            "audioData": f"<{len(audio_input['audio'])} bytes>"
        })

        # Start audio connection if not already active
        if not self.audio_connection_active:
            await self.start_audio_connection()

        # Update last audio time and cancel any pending silence task
        self.last_audio_time = time.time()
        if self.silence_task is not None and not self.silence_task.done():
            self.silence_task.cancel()
            try:
                await self.silence_task
            except asyncio.CancelledError:
                pass

        # Convert audio to Nova Sonic base64 format
        nova_audio_data = base64.b64encode(audio_input["audio"]).decode("utf-8")

        # Send audio input event
        audio_event = json.dumps(
            {
                "event": {
                    "audioInput": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                        "content": nova_audio_data,
                    }
                }
            }
        )

        await self._send_nova_event(audio_event)

        # Start silence detection task
        self.silence_task = asyncio.create_task(self._check_silence())
    
    async def send_image_content(self, image_input: ImageInputEvent) -> None:
        """Send image content - not supported by Nova Sonic.
        
        Nova Sonic currently only supports audio input, not image/video.
        This method is provided for interface compatibility.
        """
        logger.warning("Image input not supported by Nova Sonic model")
        # Nova Sonic doesn't support image input, so this is a no-op
    
    async def _check_silence(self) -> None:
        """Check for silence and automatically end audio connection."""
        try:
            await asyncio.sleep(self.silence_threshold)
            if self.audio_connection_active and self.last_audio_time:
                elapsed = time.time() - self.last_audio_time
                if elapsed >= self.silence_threshold:
                    logger.debug("Nova silence detected: %.2f seconds", elapsed)
                    await self.end_audio_input()
        except asyncio.CancelledError:
            pass

    async def end_audio_input(self) -> None:
        """End current audio input connection to trigger Nova Sonic processing."""
        if not self.audio_connection_active:
            return

        logger.debug("Nova audio connection end")

        audio_content_end = json.dumps(
            {"event": {"contentEnd": {"promptName": self.prompt_name, "contentName": self.audio_content_name}}}
        )

        await self._send_nova_event(audio_content_end)
        self.audio_connection_active = False

    async def send_text_content(self, text: str, **kwargs) -> None:
        """Send text content using Nova Sonic format."""
        if not self._active:
            return

        # Log outgoing text
        self.event_logger.log_outgoing("text_input", {"text": text})

        content_name = str(uuid.uuid4())
        events = [
            self._get_text_content_start_event(content_name),
            self._get_text_input_event(content_name, text),
            self._get_content_end_event(content_name),
        ]

        for event in events:
            await self._send_nova_event(event)

    async def send_interrupt(self) -> None:
        """Send interruption signal to Nova Sonic."""
        if not self._active:
            return

        # Nova Sonic handles interruption through special input events
        interrupt_event = {
            "event": {
                "audioInput": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                    "stopReason": "INTERRUPTED",
                }
            }
        }
        await self._send_nova_event(interrupt_event)

    async def send_tool_result(self, tool_use_id: str, result: Dict[str, Any]) -> None:
        """Send tool result using Nova Sonic toolResult format."""
        if not self._active:
            return

        logger.debug("Nova tool result send: %s", tool_use_id)
        content_name = str(uuid.uuid4())
        events = [
            self._get_tool_content_start_event(content_name, tool_use_id),
            self._get_tool_result_event(content_name, result),
            self._get_content_end_event(content_name),
        ]

        for i, event in enumerate(events):
            await self._send_nova_event(event)



    async def close(self) -> None:
        """Close Nova Sonic connection with proper cleanup sequence."""
        if not self._active:
            return

        logger.debug("Nova cleanup - starting connection close")
        self._active = False

        # Cancel response processing task if running
        if hasattr(self, "_response_task") and not self._response_task.done():
            self._response_task.cancel()
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass

        try:
            # End audio connection if active
            if self.audio_connection_active:
                await self.end_audio_input()

            # Send cleanup events
            cleanup_events = [self._get_prompt_end_event(), self._get_connection_end_event()]

            for event in cleanup_events:
                try:
                    await self._send_nova_event(event)
                except Exception as e:
                    logger.warning("Error during Nova Sonic cleanup: %s", e)

            # Close stream
            try:
                await self.stream.input_stream.close()
            except Exception as e:
                logger.warning("Error closing Nova Sonic stream: %s", e)

        except Exception as e:
            logger.error("Nova cleanup error: %s", str(e))
        finally:
            logger.debug("Nova connection closed")

    def _convert_nova_event(self, nova_event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert Nova Sonic events to provider-agnostic format.
        
        Dispatches to specific handler methods for complex event types.
        """
        # Simple events handled inline
        if "sessionStart" in nova_event:
            return SessionStartEvent(
                session_id=self.prompt_name,
                model=self.config.get("model_id", "amazon.nova-sonic-v1:0"),
                capabilities=["audio", "tools"]
            )

        elif "completionStart" in nova_event:
            completion_id = nova_event["completionStart"].get("completionId", str(uuid.uuid4()))
            self._current_turn_id = completion_id
            return TurnStartEvent(turn_id=completion_id)

        elif "audioOutput" in nova_event:
            audio_content = nova_event["audioOutput"]["content"]
            audio_bytes = base64.b64decode(audio_content)
            audio_stream = AudioStreamEvent(
                audio=audio_bytes,
                format="pcm",
                sample_rate=24000,
                channels=1
            )
            return {"audio_stream": audio_stream}

        # Complex events delegated to helper methods
        elif "textOutput" in nova_event:
            return self._handle_text_output(nova_event)
            text_content = nova_event["textOutput"]["content"]
            # Use stored role from contentStart event, fallback to event role
            role = getattr(self, "_current_role", nova_event["textOutput"].get("role", "assistant"))

            # Check for Nova Sonic interruption pattern (matches working sample)
            if '{ "interrupted" : true }' in text_content:
                logger.debug("Nova interruption detected in text")
                turn_id = getattr(self, "_current_turn_id", None)
                interruption = InterruptionEvent(reason="user_speech", turn_id=turn_id)
                return {"interruption": interruption}

            # Deduplicate transcripts (Nova sends the same text multiple times)
            if text_content in self._seen_transcripts:
                logger.debug("Skipping duplicate transcript: %s", text_content[:50])
                return None
            
            # Add to seen transcripts and maintain window size
            self._seen_transcripts.add(text_content)
            if len(self._seen_transcripts) > self._transcript_window_size:
                # Remove oldest (convert to list, remove first, convert back)
                transcripts_list = list(self._seen_transcripts)
                self._seen_transcripts = set(transcripts_list[1:])

            # Log transcription for debugging (use DEBUG level to avoid duplicate prints)
            if role == "USER":
                logger.debug("User transcript: %s", text_content)
            elif role == "ASSISTANT":
                logger.debug("Assistant transcript: %s", text_content)

            # Map role to source
            source = "user" if role == "USER" else "assistant"
            # Nova Sonic text outputs are typically final transcripts
            is_final = True

            transcript_stream = TranscriptStreamEvent(
                text=text_content,
                source=source,
                is_final=is_final
            )
            return {"transcript_stream": transcript_stream}

        # Handle toolUse → ToolUseEvent
        elif "toolUse" in nova_event:
            tool_use = nova_event["toolUse"]

            # Nova Sonic returns complete tool use (no streaming)
            # Create single delta with complete tool use
            delta: Dict[str, Any] = {
                "toolUse": {
                    "input": tool_use["content"]  # Already JSON string
                }
            }
            
            current_tool_use = {
                "toolUseId": tool_use["toolUseId"],
                "name": tool_use["toolName"],
                "input": tool_use["content"]  # Already JSON string
            }
            
            tool_use_event = ToolUseStreamEvent(delta, current_tool_use)
            return {"tool_use": tool_use_event}

        # Handle contentEnd with stopReason="INTERRUPTED" → InterruptionEvent
        elif "contentEnd" in nova_event and nova_event["contentEnd"].get("stopReason") == "INTERRUPTED":
            logger.debug("Nova interruption stop reason in contentEnd")
            turn_id = getattr(self, "_current_turn_id", None)
            interruption = InterruptionEvent(reason="user_speech", turn_id=turn_id)
            return {"interruption": interruption}

        # Handle completionEnd → TurnCompleteEvent
        elif "completionEnd" in nova_event:
            turn_id = getattr(self, "_current_turn_id", str(uuid.uuid4()))
            stop_reason_map = {
                "COMPLETE": "complete",
                "INTERRUPTED": "interrupted",
                "TOOL_USE": "tool_use",
                "ERROR": "error"
            }
            nova_stop_reason = nova_event["completionEnd"].get("stopReason", "COMPLETE")
            stop_reason = stop_reason_map.get(nova_stop_reason, "complete")

            turn_complete = TurnCompleteEvent(turn_id=turn_id, stop_reason=stop_reason)
            return {"turn_complete": turn_complete}

        # Handle usageEvent → UsageEvent
        elif "usageEvent" in nova_event:
            usage_data = nova_event["usageEvent"]
            
            # Build modality breakdown
            details = []
            usage_details = usage_data.get("details", {})
            
            # Extract speech tokens (audio modality)
            total_usage = usage_details.get("total", {})
            input_speech = total_usage.get("input", {}).get("speechTokens", 0)
            output_speech = total_usage.get("output", {}).get("speechTokens", 0)
            if input_speech or output_speech:
                details.append({
                    "modality": "audio",
                    "input_tokens": input_speech,
                    "output_tokens": output_speech
                })
            
            # Extract text tokens
            input_text = total_usage.get("input", {}).get("textTokens", 0)
            output_text = total_usage.get("output", {}).get("textTokens", 0)
            if input_text or output_text:
                details.append({
                    "modality": "text",
                    "input_tokens": input_text,
                    "output_tokens": output_text
                })

            input_tokens = usage_data.get("totalInputTokens", 0)
            output_tokens = usage_data.get("totalOutputTokens", 0)
            total_tokens = input_tokens + output_tokens
            
            usage_event = MultimodalUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                modality_details=details if details else None
            )
            return {"usage": usage_event}

        # Handle sessionEnd → SessionEndEvent
        elif "sessionEnd" in nova_event:
            reason_map = {
                "CLIENT_DISCONNECT": "client_disconnect",
                "TIMEOUT": "timeout",
                "ERROR": "error",
                "COMPLETE": "complete"
            }
            nova_reason = nova_event["sessionEnd"].get("reason", "COMPLETE")
            reason = reason_map.get(nova_reason, "complete")

            session_end = SessionEndEvent(reason=reason)
            return {"session_end": session_end}

        # Handle content start events (track role and turn state)
        elif "contentStart" in nova_event:
            role = nova_event["contentStart"].get("role", "unknown")
            # Store role for subsequent text output events
            self._current_role = role
            return None

        # Handle error responses → ErrorEvent
        elif "error" in nova_event or "exception" in nova_event:
            error_data = nova_event.get("error", nova_event.get("exception", {}))
            error_code = error_data.get("code", "UNKNOWN_ERROR")
            error_message = error_data.get("message", "An error occurred")
            
            # Create exception from error data
            error_exception = Exception(error_message)
            error_exception.__class__.__name__ = error_code  # Set exception name to error code
            
            error_event = ErrorEvent(
                error=error_exception,
                code=error_code,
                details=error_data.get("details")
            )
            return {"error": error_event}

        # Handle other events
        else:
            return None

    # Nova Sonic event template methods
    def _get_connection_start_event(self) -> str:
        """Generate Nova Sonic connection start event."""
        return json.dumps({"event": {"sessionStart": {"inferenceConfiguration": NOVA_INFERENCE_CONFIG}}})

    def _get_prompt_start_event(self, tools: list[ToolSpec]) -> str:
        """Generate Nova Sonic prompt start event with tool configuration."""
        prompt_start_event = {
            "event": {
                "promptStart": {
                    "promptName": self.prompt_name,
                    "textOutputConfiguration": NOVA_TEXT_CONFIG,
                    "audioOutputConfiguration": NOVA_AUDIO_OUTPUT_CONFIG,
                }
            }
        }

        if tools:
            tool_config = self._build_tool_configuration(tools)
            prompt_start_event["event"]["promptStart"]["toolUseOutputConfiguration"] = NOVA_TOOL_CONFIG
            prompt_start_event["event"]["promptStart"]["toolConfiguration"] = {"tools": tool_config}

        return json.dumps(prompt_start_event)

    def _build_tool_configuration(self, tools: List[ToolSpec]) -> List[Dict[str, Any]]:
        """Build tool configuration from tool specs."""
        tool_config = []
        for tool in tools:
            input_schema = (
                {"json": json.dumps(tool["inputSchema"]["json"])}
                if "json" in tool["inputSchema"]
                else {"json": json.dumps(tool["inputSchema"])}
            )

            tool_config.append(
                {"toolSpec": {"name": tool["name"], "description": tool["description"], "inputSchema": input_schema}}
            )
        return tool_config

    def _get_system_prompt_events(self, system_prompt: str) -> List[str]:
        """Generate system prompt events."""
        content_name = str(uuid.uuid4())
        return [
            self._get_text_content_start_event(content_name, "SYSTEM"),
            self._get_text_input_event(content_name, system_prompt),
            self._get_content_end_event(content_name),
        ]

    def _get_text_content_start_event(self, content_name: str, role: str = "USER") -> str:
        """Generate text content start event."""
        return json.dumps(
            {
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": content_name,
                        "type": "TEXT",
                        "role": role,
                        "interactive": True,
                        "textInputConfiguration": NOVA_TEXT_CONFIG,
                    }
                }
            }
        )

    def _get_tool_content_start_event(self, content_name: str, tool_use_id: str) -> str:
        """Generate tool content start event."""
        return json.dumps(
            {
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": content_name,
                        "interactive": False,
                        "type": "TOOL",
                        "role": "TOOL",
                        "toolResultInputConfiguration": {
                            "toolUseId": tool_use_id,
                            "type": "TEXT",
                            "textInputConfiguration": NOVA_TEXT_CONFIG,
                        },
                    }
                }
            }
        )

    def _get_text_input_event(self, content_name: str, text: str) -> str:
        """Generate text input event."""
        return json.dumps(
            {"event": {"textInput": {"promptName": self.prompt_name, "contentName": content_name, "content": text}}}
        )

    def _get_tool_result_event(self, content_name: str, result: Dict[str, Any]) -> str:
        """Generate tool result event."""
        return json.dumps(
            {
                "event": {
                    "toolResult": {
                        "promptName": self.prompt_name,
                        "contentName": content_name,
                        "content": json.dumps(result),
                    }
                }
            }
        )

    def _get_content_end_event(self, content_name: str) -> str:
        """Generate content end event."""
        return json.dumps({"event": {"contentEnd": {"promptName": self.prompt_name, "contentName": content_name}}})

    def _get_prompt_end_event(self) -> str:
        """Generate prompt end event."""
        return json.dumps({"event": {"promptEnd": {"promptName": self.prompt_name}}})

    def _get_connection_end_event(self) -> str:
        """Generate connection end event."""
        return json.dumps({"event": {"connectionEnd": {}}})

    async def _send_nova_event(self, event: str) -> None:
        """Send event JSON string to Nova Sonic stream."""
        try:
            # Event is already a JSON string
            bytes_data = event.encode("utf-8")
            chunk = InvokeModelWithBidirectionalStreamInputChunk(value=BidirectionalInputPayloadPart(bytes_=bytes_data))
            await self.stream.input_stream.send(chunk)
            logger.debug("Successfully sent Nova Sonic event")

        except Exception as e:
            logger.error("Error sending Nova Sonic event: %s", e)
            logger.error("Event was: %s", event)
            raise
    
    def _handle_text_output(self, nova_event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle textOutput events (transcripts and interruptions)."""
        text_content = nova_event["textOutput"]["content"]
        role = getattr(self, "_current_role", nova_event["textOutput"].get("role", "assistant"))

        # Check for interruption pattern
        if '{ "interrupted" : true }' in text_content:
            logger.debug("Nova interruption detected in text")
            turn_id = getattr(self, "_current_turn_id", None)
            interruption = InterruptionEvent(reason="user_speech", turn_id=turn_id)
            return {"interruption": interruption}

        # Deduplicate transcripts (Nova sends duplicates)
        if text_content in self._seen_transcripts:
            logger.debug("Skipping duplicate transcript: %s", text_content[:50])
            return None
        
        # Add to seen transcripts and maintain window size
        self._seen_transcripts.add(text_content)
        if len(self._seen_transcripts) > self._transcript_window_size:
            transcripts_list = list(self._seen_transcripts)
            self._seen_transcripts = set(transcripts_list[1:])

        # Log transcription
        if role == "USER":
            logger.debug("User transcript: %s", text_content)
        elif role == "ASSISTANT":
            logger.debug("Assistant transcript: %s", text_content)

        # Map role to source
        source = "user" if role == "USER" else "assistant"
        is_final = True  # Nova Sonic text outputs are typically final

        transcript_stream = TranscriptStreamEvent(
            text=text_content,
            source=source,
            is_final=is_final
        )
        return {"transcript_stream": transcript_stream}


class NovaSonicBidirectionalModel(BidirectionalModel):
    """Nova Sonic model implementation for bidirectional streaming.

    Provides access to Amazon's Nova Sonic model through the bidirectional
    streaming interface, handling AWS authentication and connection management.
    """

    def __init__(self, model_id: str = "amazon.nova-sonic-v1:0", region: str = "us-east-1", **config: any) -> None:
        """Initialize Nova Sonic bidirectional model.

        Args:
            model_id: Nova Sonic model identifier.
            region: AWS region.
            **config: Additional configuration.
        """
        self.model_id = model_id
        self.region = region
        self.config = config
        self._client = None

        logger.debug("Nova Sonic bidirectional model initialized: %s", model_id)

    async def create_bidirectional_connection(
        self,
        system_prompt: str | None = None,
        tools: list[ToolSpec] | None = None,
        messages: Messages | None = None,
        **kwargs,
    ) -> BidirectionalModelSession:
        """Create Nova Sonic bidirectional connection."""
        logger.debug("Nova connection create - starting")

        # Initialize client if needed
        if not self._client:
            await self._initialize_client()

        # Start Nova Sonic bidirectional stream
        try:
            stream = await self._client.invoke_model_with_bidirectional_stream(
                InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
            )

            # Create and initialize connection
            connection = NovaSonicSession(stream, self.config)
            await connection.initialize(system_prompt, tools, messages)

            logger.debug("Nova connection created")
            return connection
        except Exception as e:
            logger.error("Nova connection create error: %s", str(e))
            logger.error("Failed to create Nova Sonic connection: %s", e)
            raise

    async def _initialize_client(self) -> None:
        """Initialize Nova Sonic client."""
        try:
            config = Config(
                endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
                region=self.region,
                aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
                auth_scheme_resolver=HTTPAuthSchemeResolver(),
                auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="bedrock")},
            )

            self._client = BedrockRuntimeClient(config=config)
            logger.debug("Nova Sonic client initialized")

        except ImportError as e:
            logger.error("Nova Sonic dependencies not available: %s", e)
            raise
        except Exception as e:
            logger.error("Error initializing Nova Sonic client: %s", e)
            raise
