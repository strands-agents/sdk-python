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
import logging
import uuid
from typing import Any, AsyncIterable, Dict, List, Optional, Union

from google import genai
from google.genai import types as genai_types
from google.genai.types import LiveServerMessage, LiveServerContent

from ....types.content import Messages
from ....types.tools import ToolResult, ToolSpec, ToolUse
from ....types._events import ToolResultEvent
from ..types.bidirectional_streaming import (
    AudioInputEvent,
    AudioStreamEvent,
    ErrorEvent,
    ImageInputEvent,
    InterruptionEvent,
    SessionEndEvent,
    SessionStartEvent,
    TextInputEvent,
    TranscriptStreamEvent,
    TurnCompleteEvent,
    TurnStartEvent,
)
from .bidirectional_model import BidirectionalModel

logger = logging.getLogger(__name__)

# Audio format constants
GEMINI_INPUT_SAMPLE_RATE = 16000
GEMINI_OUTPUT_SAMPLE_RATE = 24000
GEMINI_CHANNELS = 1


class GeminiLiveBidirectionalModel(BidirectionalModel):
    """Gemini Live API implementation using official Google GenAI SDK.
    
    Combines model configuration and connection state in a single class.
    Provides a clean interface to Gemini Live API using the official SDK,
    eliminating custom WebSocket handling and providing robust error handling.
    """
    
    def __init__(
        self,
        model_id: str = "models/gemini-2.0-flash-live-preview-04-09",
        api_key: Optional[str] = None,
        live_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize Gemini Live API bidirectional model.
        
        Args:
            model_id: Gemini Live model identifier.
            api_key: Google AI API key for authentication.
            live_config: Gemini Live API configuration parameters (e.g., response_modalities, speech_config).
            **kwargs: Reserved for future parameters.
        """
        # Model configuration
        self.model_id = model_id
        self.api_key = api_key
        self.live_config = live_config or {}
        
        # Create Gemini client with proper API version
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        
        # Use v1alpha for Live API as it has better model support
        client_kwargs["http_options"] = {"api_version": "v1alpha"}
        
        self.client = genai.Client(**client_kwargs)
        
        # Connection state (initialized in connect())
        self.live_session = None
        self.live_session_cm = None
        self.session_id = None
        self._active = False
    
    async def connect(
        self,
        system_prompt: Optional[str] = None,
        tools: Optional[List[ToolSpec]] = None,
        messages: Optional[Messages] = None,
        **kwargs
    ) -> None:
        """Establish bidirectional connection with Gemini Live API.
        
        Args:
            system_prompt: System instructions for the model.
            tools: List of tools available to the model.
            messages: Conversation history to initialize with.
            **kwargs: Additional configuration options.
        """
        if self._active:
            raise RuntimeError("Connection already active. Close the existing connection before creating a new one.")
        
        try:
            # Initialize connection state
            self.session_id = str(uuid.uuid4())
            self._active = True
            
            # Build live config
            live_config = self._build_live_config(system_prompt, tools, **kwargs)
            
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
            logger.error("Error connecting to Gemini Live: %s", e)
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
    
    async def receive(self) -> AsyncIterable[Dict[str, Any]]:
        """Receive Gemini Live API events and convert to provider-agnostic format."""
        
        # Emit session start event
        yield SessionStartEvent(
            session_id=self.session_id,
            model=self.model_id,
            capabilities=["audio", "tools", "images"]
        )
        
        try:
            # Wrap in while loop to restart after turn_complete (SDK limitation workaround)
            while self._active:
                try:
                    async for message in self.live_session.receive():
                        if not self._active:
                            break
                        
                        # Convert to provider-agnostic format
                        provider_event = self._convert_gemini_live_event(message)
                        if provider_event:
                            yield provider_event
                    
                    # SDK exits receive loop after turn_complete - restart automatically
                    if self._active:
                        logger.debug("Restarting receive loop after turn completion")
                    
                except Exception as e:
                    logger.error("Error in receive iteration: %s", e)
                    # Small delay before retrying to avoid tight error loops
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            logger.error("Fatal error in receive loop: %s", e)
            yield ErrorEvent(error=e)
        finally:
            # Emit session end event when exiting
            yield SessionEndEvent(reason="complete")
    
    def _convert_gemini_live_event(self, message: LiveServerMessage) -> Optional[Dict[str, Any]]:
        """Convert Gemini Live API events to provider-agnostic format.
        
        Handles different types of content:
        - inputTranscription: User's speech transcribed to text
        - outputTranscription: Model's audio transcribed to text
        - modelTurn text: Text response from the model
        """
        try:
            # Handle interruption first (from server_content)
            if message.server_content and message.server_content.interrupted:
                return InterruptionEvent(reason="user_speech", turn_id=None)
            
            # Handle input transcription (user's speech) - emit as transcript event
            if message.server_content and message.server_content.input_transcription:
                input_transcript = message.server_content.input_transcription
                # Check if the transcription object has text content
                if hasattr(input_transcript, 'text') and input_transcript.text:
                    transcription_text = input_transcript.text
                    logger.debug(f"Input transcription detected: {transcription_text}")
                    return TranscriptStreamEvent(
                        text=transcription_text,
                        source="user",
                        is_final=True
                    )
            
            # Handle output transcription (model's audio) - emit as transcript event
            if message.server_content and message.server_content.output_transcription:
                output_transcript = message.server_content.output_transcription
                # Check if the transcription object has text content
                if hasattr(output_transcript, 'text') and output_transcript.text:
                    transcription_text = output_transcript.text
                    logger.debug(f"Output transcription detected: {transcription_text}")
                    return TranscriptStreamEvent(
                        text=transcription_text,
                        source="assistant",
                        is_final=True
                    )
            
            # Handle text output from model
            if message.text:
                logger.debug(f"Text output as transcript: {message.text}")
                return TranscriptStreamEvent(
                    text=message.text,
                    source="assistant",
                    is_final=True
                )
            
            # Handle audio output using SDK's built-in data property
            if message.data:
                return AudioStreamEvent(
                    audio=message.data,
                    format="pcm",
                    sample_rate=GEMINI_OUTPUT_SAMPLE_RATE,
                    channels=GEMINI_CHANNELS
                )
            
            # Handle tool calls
            if message.tool_call and message.tool_call.function_calls:
                for func_call in message.tool_call.function_calls:
                    tool_use_event: ToolUse = {
                        "toolUseId": func_call.id,
                        "name": func_call.name,
                        "input": func_call.args or {}
                    }
                    return {"toolUse": tool_use_event}
            
            # Silently ignore setup_complete, turn_complete, generation_complete, and usage_metadata messages
            return None
            
        except Exception as e:
            logger.error("Error converting Gemini Live event: %s", e)
            logger.error("Message type: %s", type(message).__name__)
            logger.error("Message attributes: %s", [attr for attr in dir(message) if not attr.startswith('_')])
            return None
    
    async def send(
        self,
        content: Union[TextInputEvent, AudioInputEvent, ImageInputEvent, ToolResultEvent],
    ) -> None:
        """Unified send method for all content types. Sends the given inputs to Google Live API
        
        Dispatches to appropriate internal handler based on content type.
        
        Args:
            content: Typed event (TextInputEvent, AudioInputEvent, ImageInputEvent, or ToolResultEvent).
        """
        if not self._active:
            return
        
        try:
            if isinstance(content, TextInputEvent):
                await self._send_text_content(content.text)
            elif isinstance(content, AudioInputEvent):
                await self._send_audio_content(content)
            elif isinstance(content, ImageInputEvent):
                await self._send_image_content(content)
            elif isinstance(content, ToolResultEvent):
                tool_result = content.get("tool_result")
                if tool_result:
                    await self._send_tool_result(tool_result)
            else:
                logger.warning(f"Unknown content type: {type(content)}")
        except Exception as e:
            logger.error(f"Error sending content: {e}")
    
    async def _send_audio_content(self, audio_input: AudioInputEvent) -> None:
        """Internal: Send audio content using Gemini Live API.
        
        Gemini Live expects continuous audio streaming via send_realtime_input.
        This automatically triggers VAD and can interrupt ongoing responses.
        """
        try:
            # Create audio blob for the SDK
            audio_blob = genai_types.Blob(
                data=audio_input.audio,
                mime_type=f"audio/pcm;rate={GEMINI_INPUT_SAMPLE_RATE}"
            )
            
            # Send real-time audio input - this automatically handles VAD and interruption
            await self.live_session.send_realtime_input(audio=audio_blob)
            
        except Exception as e:
            logger.error("Error sending audio content: %s", e)
    
    async def _send_image_content(self, image_input: ImageInputEvent) -> None:
        """Internal: Send image content using Gemini Live API.
        
        Sends image frames following the same pattern as the GitHub example.
        Images are sent as base64-encoded data with MIME type.
        """
        try:
            # Prepare the message based on encoding
            if image_input.encoding == "base64":
                # Data is already base64 encoded
                if isinstance(image_input.image, bytes):
                    data_str = image_input.image.decode()
                else:
                    data_str = image_input.image
            else:
                # Raw bytes - need to base64 encode
                data_str = base64.b64encode(image_input.image).decode()
            
            # Create the message in the format expected by Gemini Live
            msg = {
                "mime_type": image_input.mime_type,
                "data": data_str
            }
            
            # Send using the same method as the GitHub example
            await self.live_session.send(input=msg)
            
        except Exception as e:
            logger.error("Error sending image content: %s", e)
    
    async def _send_text_content(self, text: str) -> None:
        """Internal: Send text content using Gemini Live API."""
        try:
            # Create content with text
            content = genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=text)]
            )
            
            # Send as client content
            await self.live_session.send_client_content(turns=content)
            
        except Exception as e:
            logger.error("Error sending text content: %s", e)
    
    async def _send_tool_result(self, tool_result: ToolResult) -> None:
        """Internal: Send tool result using Gemini Live API."""
        try:
            tool_use_id = tool_result.get("toolUseId")
            
            # Extract result content
            result_data = {}
            if "content" in tool_result:
                # Extract text from content blocks
                for block in tool_result["content"]:
                    if "text" in block:
                        result_data = {"result": block["text"]}
                        break
            
            # Create function response
            func_response = genai_types.FunctionResponse(
                id=tool_use_id,
                name=tool_use_id,  # Gemini uses name as identifier
                response=result_data
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
            logger.error("Error closing Gemini Live connection: %s", e)
            raise
    
    def _build_live_config(
        self,
        system_prompt: Optional[str] = None,
        tools: Optional[List[ToolSpec]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Build LiveConnectConfig for the official SDK.
        
        Simply passes through all config parameters from live_config, allowing users
        to configure any Gemini Live API parameter directly.
        """
        # Start with user-provided live_config
        config_dict = {}
        if self.live_config:
            config_dict.update(self.live_config)
        
        # Override with any kwargs from connect()
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