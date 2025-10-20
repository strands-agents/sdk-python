"""Gemini Live API bidirectional model provider using official Google GenAI SDK.

Implements the BidirectionalModel interface for Google's Gemini Live API using the
official Google GenAI SDK for simplified and robust WebSocket communication.

Key improvements over custom WebSocket implementation:
- Uses official google-genai SDK with native Live API support
- Simplified session management with client.aio.live.connect()
- Built-in tool integration and event handling
- Automatic WebSocket connection management and error handling
- Native support for audio/text streaming and interruption
"""

import asyncio
import base64
import json
import logging
import uuid
from typing import Any, AsyncIterable, Dict, List, Optional, Union

from google import genai
from google.genai import types as genai_types
from google.genai.types import LiveServerMessage, LiveServerContent

from ....types._events import ToolResultEvent, ToolUseStreamEvent
from ....types.content import Messages
from ....types.tools import ToolSpec, ToolUse
from ..types.bidirectional_streaming import (
    AudioInputEvent,
    AudioOutputEvent,
    BidirectionalConnectionEndEvent,
    BidirectionalConnectionStartEvent,
    ImageInputEvent,
    InterruptionDetectedEvent,
    TextOutputEvent,
    TranscriptEvent,
)
from ..utils.event_logger import EventLogger
from .bidirectional_model import BidirectionalModel, BidirectionalModelSession

logger = logging.getLogger(__name__)

# Audio format constants
GEMINI_INPUT_SAMPLE_RATE = 16000
GEMINI_OUTPUT_SAMPLE_RATE = 24000
GEMINI_CHANNELS = 1


class GeminiLiveSession(BidirectionalModelSession):
    """Gemini Live API session using official Google GenAI SDK.
    
    Provides a clean interface to Gemini Live API using the official SDK,
    eliminating custom WebSocket handling and providing robust error handling.
    """
    
    def __init__(self, client: genai.Client, model_id: str, config: Dict[str, Any]):
        """Initialize Gemini Live API session.
        
        Args:
            client: Gemini client instance
            model_id: Model identifier
            config: Model configuration including live config
        """
        self.client = client
        self.model_id = model_id
        self.config = config
        self.session_id = str(uuid.uuid4())
        self._active = True
        self.live_session = None
        self.live_session_cm = None
        self.event_logger = EventLogger("gemini")
        

    
    async def initialize(
        self,
        system_prompt: Optional[str] = None,
        tools: Optional[List[ToolSpec]] = None,
        messages: Optional[Messages] = None
    ) -> None:
        """Initialize Gemini Live API session by creating the connection."""
        
        try:
            # Build live config
            live_config = self.config.get("live_config")
            
            if live_config is None:
                raise ValueError("live_config is required but not found in session config")
            
            # Create the context manager
            self.live_session_cm = self.client.aio.live.connect(
                model=self.model_id,
                config=live_config
            )
            
            # Enter the context manager
            self.live_session = await self.live_session_cm.__aenter__()
            
            # Send initial message history if provided
            if messages:
                await self._send_message_history(messages)
            
            
        except Exception as e:
            logger.error("Error initializing Gemini Live session: %s", e)
            raise
    
    async def _send_message_history(self, messages: Messages) -> None:
        """Send conversation history to Gemini Live API.
        
        Sends each message as a separate turn with the correct role to maintain
        proper conversation context. Follows the same pattern as the non-bidirectional
        Gemini model implementation.
        """
        if not messages:
            return
        
        # Convert each message to Gemini format and send separately
        for message in messages:
            content_parts = []
            for content_block in message["content"]:
                if "text" in content_block:
                    content_parts.append(genai_types.Part(text=content_block["text"]))
            
            if content_parts:
                # Map role correctly - Gemini uses "user" and "model" roles
                # "assistant" role from Messages format maps to "model" in Gemini
                role = "model" if message["role"] == "assistant" else message["role"]
                content = genai_types.Content(role=role, parts=content_parts)
                await self.live_session.send_client_content(turns=content)
    
    async def receive_events(self) -> AsyncIterable[Dict[str, Any]]:
        """Receive Gemini Live API events and convert to new TypedEvent format."""
        
        # Emit SessionStartEvent on connection
        from ..types.bidirectional_streaming import SessionStartEvent
        session_start = SessionStartEvent(
            session_id=self.session_id,
            model=self.model_id,
            capabilities=["audio", "text", "tools", "images"]
        )
        yield session_start
        
        # Track turn state for TurnStartEvent emission
        current_turn_id: Optional[str] = None
        
        restart_count = 0
        max_restarts = 100
        
        try:
            # Wrap in while loop to restart after turn_complete (SDK limitation workaround)
            while self._active and restart_count < max_restarts:
                try:
                    async for message in self.live_session.receive():
                        if not self._active:
                            break
                        
                        # Convert to new TypedEvent format
                        events = self._convert_gemini_live_event(message, current_turn_id)
                        for event in events:
                            # Track turn state
                            from ..types.bidirectional_streaming import TurnStartEvent, TurnCompleteEvent
                            if isinstance(event, TurnStartEvent):
                                current_turn_id = event.turn_id
                            elif isinstance(event, TurnCompleteEvent):
                                current_turn_id = None
                            
                            yield event
                    
                    # SDK exits receive loop after turn_complete - restart automatically
                    if self._active:
                        restart_count += 1
                        logger.debug("Restarting receive loop after turn completion (%d/%d)", restart_count, max_restarts)
                    
                except Exception as e:
                    logger.error("Error in receive iteration: %s", e)
                    # Convert exception to ErrorEvent
                    from ..types.bidirectional_streaming import ErrorEvent
                    error_event = ErrorEvent(
                        error=e,
                        code="receive_error",
                        details={"exception_type": type(e).__name__}
                    )
                    yield error_event
                    # Small delay before retrying to avoid tight error loops
                    await asyncio.sleep(0.1)
            
            if restart_count >= max_restarts:
                logger.warning("Max restart count reached, ending receive loop")
                    
        except Exception as e:
            logger.error("Fatal error in receive loop: %s", e)
            # Convert fatal exception to ErrorEvent
            from ..types.bidirectional_streaming import ErrorEvent
            error_event = ErrorEvent(
                error=e,
                code="fatal_error",
                details={"exception_type": type(e).__name__}
            )
            yield error_event
        finally:
            # Emit SessionEndEvent when exiting
            from ..types.bidirectional_streaming import SessionEndEvent
            session_end = SessionEndEvent(reason="complete")
            yield session_end
    
    def _convert_gemini_live_event(self, message: LiveServerMessage, current_turn_id: Optional[str]) -> List[Any]:
        """Convert Gemini Live API events to new TypedEvent format.
        
        Handles different types of text output:
        - inputTranscription: User's speech transcribed to text (emitted as TranscriptStreamEvent)
        - outputTranscription: Model's audio transcribed to text (emitted as TranscriptStreamEvent)
        - modelTurn text: Actual text response from the model (emitted as TranscriptStreamEvent)
        
        Returns:
            List of TypedEvent instances (may be empty if event should be ignored)
        """
        from ....types._events import ToolUseStreamEvent
        from ..types.bidirectional_streaming import (
            TurnStartEvent,
            AudioStreamEvent,
            TranscriptStreamEvent,
            InterruptionEvent,
            TurnCompleteEvent,
            MultimodalUsage,
            ModalityUsage,
        )
        
        events: List[Any] = []
        
        try:
            # Log raw incoming event
            raw_event = {
                "text": message.text if hasattr(message, 'text') else None,
                "data": f"<{len(message.data)} bytes>" if hasattr(message, 'data') and message.data else None,
                "tool_call": str(message.tool_call) if hasattr(message, 'tool_call') and message.tool_call else None,
                "server_content": str(message.server_content) if hasattr(message, 'server_content') and message.server_content else None,
            }
            self.event_logger.log_incoming("gemini_raw", raw_event)
            
            # Detect first content and emit TurnStartEvent if not already in a turn
            has_content = (
                (message.text and message.text.strip()) or
                (message.data and len(message.data) > 0) or
                (message.tool_call and message.tool_call.function_calls)
            )
            
            if has_content and current_turn_id is None:
                turn_id = str(uuid.uuid4())
                events.append(TurnStartEvent(turn_id=turn_id))
                # Update current_turn_id for subsequent events
                current_turn_id = turn_id
            
            # Handle interruption (from server_content)
            if message.server_content and message.server_content.interrupted:
                events.append(InterruptionEvent(
                    reason="user_speech",
                    turn_id=current_turn_id
                ))
            
            # Handle input transcription (user's speech) - emit as TranscriptStreamEvent
            if message.server_content and message.server_content.input_transcription:
                input_transcript = message.server_content.input_transcription
                # Check if the transcription object has text content
                if hasattr(input_transcript, 'text') and input_transcript.text:
                    transcription_text = input_transcript.text
                    logger.debug(f"Input transcription detected: {transcription_text}")
                    events.append(TranscriptStreamEvent(
                        text=transcription_text,
                        source="user",
                        is_final=True  # Gemini provides final transcripts
                    ))
            
            # Handle output transcription (model's audio) - emit as TranscriptStreamEvent
            if message.server_content and message.server_content.output_transcription:
                output_transcript = message.server_content.output_transcription
                # Check if the transcription object has text content
                if hasattr(output_transcript, 'text') and output_transcript.text:
                    transcription_text = output_transcript.text
                    logger.debug(f"Output transcription detected: {transcription_text}")
                    events.append(TranscriptStreamEvent(
                        text=transcription_text,
                        source="assistant",
                        is_final=True  # Gemini provides final transcripts
                    ))
            
            # Handle actual text output from model (not transcription)
            # The SDK's message.text property accesses modelTurn.parts[].text
            if message.text:
                events.append(TranscriptStreamEvent(
                    text=message.text,
                    source="assistant",
                    is_final=True
                ))
            
            # Handle audio output using SDK's built-in data property
            if message.data:
                events.append(AudioStreamEvent(
                    audio=message.data,
                    format="pcm",
                    sample_rate=GEMINI_OUTPUT_SAMPLE_RATE,
                    channels=GEMINI_CHANNELS
                ))
            
            # Handle tool calls
            if message.tool_call and message.tool_call.function_calls:
                for func_call in message.tool_call.function_calls:
                    # Gemini returns complete tool use (no streaming)
                    # Create single delta with complete tool use
                    tool_input_json = json.dumps(func_call.args or {})
                    
                    delta: Dict[str, Any] = {
                        "toolUse": {
                            "input": tool_input_json
                        }
                    }
                    
                    current_tool_use = {
                        "toolUseId": func_call.id,
                        "name": func_call.name,
                        "input": tool_input_json
                    }
                    
                    events.append(ToolUseStreamEvent(delta, current_tool_use))
            
            # Handle turn_complete or generation_complete
            if message.server_content and (
                message.server_content.turn_complete or 
                hasattr(message.server_content, 'generation_complete')
            ):
                if current_turn_id:
                    # Determine stop reason
                    stop_reason = "complete"
                    if message.server_content.interrupted:
                        stop_reason = "interrupted"
                    elif message.tool_call and message.tool_call.function_calls:
                        stop_reason = "tool_use"
                    
                    events.append(TurnCompleteEvent(
                        turn_id=current_turn_id,
                        stop_reason=stop_reason
                    ))
            
            # Handle usage metadata
            if hasattr(message, 'usage_metadata') and message.usage_metadata:
                usage_meta = message.usage_metadata
                total_input = getattr(usage_meta, 'prompt_token_count', 0) or 0
                total_output = getattr(usage_meta, 'candidates_token_count', 0) or 0
                
                # Gemini may not provide detailed modality breakdown
                details: List[ModalityUsage] = []
                if total_input > 0 or total_output > 0:
                    # Create a generic entry since Gemini doesn't break down by modality
                    details.append({
                        "modality": "text",
                        "input_tokens": total_input,
                        "output_tokens": total_output
                    })
                
                total_tokens = total_input + total_output
                events.append(MultimodalUsage(
                    input_tokens=total_input,
                    output_tokens=total_output,
                    total_tokens=total_tokens,
                    modality_details=details if details else None
                ))
            
            return events
            
        except Exception as e:
            logger.error("Error converting Gemini Live event: %s", e)
            logger.error("Message type: %s", type(message).__name__)
            logger.error("Message attributes: %s", [attr for attr in dir(message) if not attr.startswith('_')])
            return []
    
    async def send(self, event: Union[AudioInputEvent, ImageInputEvent, ToolResultEvent]) -> None:
        """Unified send method for all input event types.
        
        Args:
            event: Input event to send (AudioInputEvent, ImageInputEvent, or ToolResultEvent)
        
        Raises:
            NotImplementedError: If the event type is not supported by this provider
        """
        from ....types._events import ToolResultEvent
        from ..types.bidirectional_streaming import AudioInputEvent, ImageInputEvent
        
        if isinstance(event, AudioInputEvent):
            # Convert to Gemini's send_realtime_input with Blob type
            await self._send_audio(event)
        elif isinstance(event, ImageInputEvent):
            # Convert to Gemini's send() with inline_data
            await self._send_image(event)
        elif isinstance(event, ToolResultEvent):
            # Convert to Gemini's send_tool_response with FunctionResponse list
            await self._send_tool_result(event)
        else:
            raise NotImplementedError(f"Unsupported event type: {type(event).__name__}")
    
    async def _send_audio(self, event: AudioInputEvent) -> None:
        """Internal method to send audio input."""
        if not self._active:
            return
        
        try:
            # Log outgoing audio
            self.event_logger.log_outgoing("audio_input", {
                "format": event.format,
                "sampleRate": event.sample_rate,
                "channels": event.channels,
                "audioData": f"<{len(event.audio)} bytes>"
            })
            
            # Create audio blob for the SDK
            audio_blob = genai_types.Blob(
                data=event.audio,
                mime_type=f"audio/pcm;rate={GEMINI_INPUT_SAMPLE_RATE}"
            )
            
            # Send real-time audio input - this automatically handles VAD and interruption
            await self.live_session.send_realtime_input(audio=audio_blob)
            
        except Exception as e:
            logger.error("Error sending audio content: %s", e)
    
    async def _send_image(self, event: ImageInputEvent) -> None:
        """Internal method to send image input."""
        if not self._active:
            return
        
        try:
            # Log outgoing image
            image_data_preview = event.image
            if isinstance(image_data_preview, bytes):
                image_data_preview = f"<{len(image_data_preview)} bytes>"
            elif isinstance(image_data_preview, str) and len(image_data_preview) > 100:
                image_data_preview = image_data_preview[:100] + f"... (total: {len(image_data_preview)} chars)"
            
            self.event_logger.log_outgoing("image_input", {
                "mimeType": event.mime_type,
                "encoding": event.encoding,
                "imageData": image_data_preview
            })
            
            # Prepare the message based on encoding
            if event.encoding == "base64":
                # Data is already base64 encoded
                if isinstance(event.image, bytes):
                    data_str = event.image.decode()
                else:
                    data_str = event.image
            else:
                # Raw bytes - need to base64 encode
                data_str = base64.b64encode(event.image).decode()
            
            # Create the message in the format expected by Gemini Live
            msg = {
                "mime_type": event.mime_type,
                "data": data_str
            }
            
            # Send using the same method as the GitHub example
            await self.live_session.send(input=msg)
            
        except Exception as e:
            logger.error("Error sending image content: %s", e)
    
    async def _send_tool_result(self, event: ToolResultEvent) -> None:
        """Internal method to send tool result."""
        if not self._active:
            return
        
        try:
            # Create function response
            func_response = genai_types.FunctionResponse(
                id=event.tool_use_id,
                name=event.tool_use_id,  # Gemini uses name as identifier
                response=event.tool_result  # Use tool_result property, not result
            )
            
            # Send tool response
            await self.live_session.send_tool_response(function_responses=[func_response])
        except Exception as e:
            logger.error("Error sending tool result: %s", e)
    
    async def close(self) -> None:
        """Close Gemini Live API connection."""
        if not self._active:
            return
        
        self._active = False
        
        try:
            # Exit the context manager properly
            if self.live_session_cm:
                await self.live_session_cm.__aexit__(None, None, None)
        except Exception as e:
            logger.error("Error closing Gemini Live session: %s", e)
            raise


class GeminiLiveBidirectionalModel(BidirectionalModel):
    """Gemini Live API model implementation using official Google GenAI SDK.
    
    Provides access to Google's Gemini Live API through the bidirectional
    streaming interface, using the official SDK for robust and simple integration.
    """
    
    def __init__(
        self,
        model_id: str = "models/gemini-2.0-flash-live-preview-04-09",
        api_key: Optional[str] = None,
        **config
    ):
        """Initialize Gemini Live API bidirectional model.
        
        Args:
            model_id: Gemini Live model identifier.
            api_key: Google AI API key for authentication.
            **config: Additional configuration.
        """
        self.model_id = model_id
        self.api_key = api_key
        self.config = config
        
        # Create Gemini client with proper API version
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        
        # Use v1alpha for Live API as it has better model support
        client_kwargs["http_options"] = {"api_version": "v1alpha"}
        
        self.client = genai.Client(**client_kwargs)
    
    async def create_bidirectional_connection(
        self,
        system_prompt: Optional[str] = None,
        tools: Optional[List[ToolSpec]] = None,
        messages: Optional[Messages] = None,
        **kwargs
    ) -> BidirectionalModelSession:
        """Create Gemini Live API bidirectional connection using official SDK."""
        
        try:
            # Build configuration
            live_config = self._build_live_config(system_prompt, tools, **kwargs)
            
            # Create session config
            session_config = self._get_session_config()
            session_config["live_config"] = live_config
            
            # Create and initialize session wrapper
            session = GeminiLiveSession(self.client, self.model_id, session_config)
            await session.initialize(system_prompt, tools, messages)
            
            return session
            
        except Exception as e:
            logger.error("Failed to create Gemini Live connection: %s", e)
            raise
    
    def _build_live_config(
        self,
        system_prompt: Optional[str] = None,
        tools: Optional[List[ToolSpec]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Build LiveConnectConfig for the official SDK.
        
        Simply passes through all config parameters from params, allowing users
        to configure any Gemini Live API parameter directly.
        """
        # Start with user config from params
        config_dict = {}
        if "params" in self.config:
            config_dict.update(self.config["params"])
        
        # Override with any kwargs
        config_dict.update(kwargs)
        
        # Add system instruction if provided
        if system_prompt:
            config_dict["system_instruction"] = system_prompt
        
        # Add tools if provided
        if tools:
            config_dict["tools"] = self._format_tools_for_live_api(tools)
        
        return config_dict
    
    def _format_tools_for_live_api(self, tool_specs: List[ToolSpec]) -> List[genai_types.Tool]:
        """Format tool specs for Gemini Live API."""
        if not tool_specs:
            return []
        
        # Debug: Log the tool specs being formatted
        for tool_spec in tool_specs:
            logger.debug(f"Formatting tool: {tool_spec['name']}")
            logger.debug(f"Tool schema: {tool_spec['inputSchema']['json']}")
            
        return [
            genai_types.Tool(
                function_declarations=[
                    genai_types.FunctionDeclaration(
                        description=tool_spec["description"],
                        name=tool_spec["name"],
                        parameters_json_schema=tool_spec["inputSchema"]["json"],
                    )
                    for tool_spec in tool_specs
                ],
            ),
        ]
    
    def _get_session_config(self) -> Dict[str, Any]:
        """Get session configuration for Gemini Live API."""
        return {
            "model_id": self.model_id,
            "params": self.config.get("params"),
            **self.config
        }
