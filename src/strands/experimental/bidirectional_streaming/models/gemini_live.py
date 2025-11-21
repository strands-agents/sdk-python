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
from typing import Any, AsyncIterable, Dict, List, Optional

from google import genai
from google.genai import types as genai_types
from google.genai.types import LiveServerMessage, LiveServerContent

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
from .base_model import BaseModel

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
        """Receive Gemini Live API events and convert to provider-agnostic format."""
        
        # Emit connection start event
        connection_start: BidirectionalConnectionStartEvent = {
            "connectionId": self.session_id,
            "metadata": {"provider": "gemini_live", "model_id": self.config.get("model_id")}
        }
        yield {"BidirectionalConnectionStart": connection_start}
        
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
        finally:
            # Emit connection end event when exiting
            connection_end: BidirectionalConnectionEndEvent = {
                "connectionId": self.session_id,
                "reason": "connection_complete",
                "metadata": {"provider": "gemini_live"}
            }
            yield {"BidirectionalConnectionEnd": connection_end}
    
    def _convert_gemini_live_event(self, message: LiveServerMessage) -> Optional[Dict[str, Any]]:
        """Convert Gemini Live API events to provider-agnostic format.
        
        Handles different types of text output:
        - inputTranscription: User's speech transcribed to text (emitted as transcript event)
        - outputTranscription: Model's audio transcribed to text (emitted as transcript event)
        - modelTurn text: Actual text response from the model (emitted as textOutput)
        """
        try:
            # Handle interruption first (from server_content)
            if message.server_content and message.server_content.interrupted:
                interruption: InterruptionDetectedEvent = {
                    "reason": "user_input"
                }
                return {"interruptionDetected": interruption}
            
            # Handle input transcription (user's speech) - emit as transcript event
            if message.server_content and message.server_content.input_transcription:
                input_transcript = message.server_content.input_transcription
                # Check if the transcription object has text content
                if hasattr(input_transcript, 'text') and input_transcript.text:
                    transcription_text = input_transcript.text
                    logger.debug(f"Input transcription detected: {transcription_text}")
                    transcript: TranscriptEvent = {
                        "text": transcription_text,
                        "role": "user",
                        "type": "input"
                    }
                    return {"transcript": transcript}
            
            # Handle output transcription (model's audio) - emit as transcript event
            if message.server_content and message.server_content.output_transcription:
                output_transcript = message.server_content.output_transcription
                # Check if the transcription object has text content
                if hasattr(output_transcript, 'text') and output_transcript.text:
                    transcription_text = output_transcript.text
                    logger.debug(f"Output transcription detected: {transcription_text}")
                    transcript: TranscriptEvent = {
                        "text": transcription_text,
                        "role": "assistant",
                        "type": "output"
                    }
                    return {"transcript": transcript}
            
            # Handle actual text output from model (not transcription)
            # The SDK's message.text property accesses modelTurn.parts[].text
            if message.text:
                text_output: TextOutputEvent = {
                    "text": message.text,
                    "role": "assistant"
                }
                return {"textOutput": text_output}
            
            # Handle audio output using SDK's built-in data property
            if message.data:
                audio_output: AudioOutputEvent = {
                    "audioData": message.data,
                    "format": "pcm",
                    "sampleRate": GEMINI_OUTPUT_SAMPLE_RATE,
                    "channels": GEMINI_CHANNELS,
                    "encoding": "raw"
                }
                return {"audioOutput": audio_output}
            
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
    
    async def send_audio_content(self, audio_input: AudioInputEvent) -> None:
        """Send audio content using Gemini Live API.
        
        Gemini Live expects continuous audio streaming via send_realtime_input.
        This automatically triggers VAD and can interrupt ongoing responses.
        """
        if not self._active:
            return
        
        try:
            # Create audio blob for the SDK
            audio_blob = genai_types.Blob(
                data=audio_input["audioData"],
                mime_type=f"audio/pcm;rate={GEMINI_INPUT_SAMPLE_RATE}"
            )
            
            # Send real-time audio input - this automatically handles VAD and interruption
            await self.live_session.send_realtime_input(audio=audio_blob)
            
        except Exception as e:
            logger.error("Error sending audio content: %s", e)
    
    async def send_image_content(self, image_input: ImageInputEvent) -> None:
        """Send image content using Gemini Live API.
        
        Sends image frames following the same pattern as the GitHub example.
        Images are sent as base64-encoded data with MIME type.
        """
        if not self._active:
            return
        
        try:
            # Prepare the message based on encoding
            if image_input["encoding"] == "base64":
                # Data is already base64 encoded
                if isinstance(image_input["imageData"], bytes):
                    data_str = image_input["imageData"].decode()
                else:
                    data_str = image_input["imageData"]
            else:
                # Raw bytes - need to base64 encode
                data_str = base64.b64encode(image_input["imageData"]).decode()
            
            # Create the message in the format expected by Gemini Live
            msg = {
                "mime_type": image_input["mimeType"],
                "data": data_str
            }
            
            # Send using the same method as the GitHub example
            await self.live_session.send(input=msg)
            
        except Exception as e:
            logger.error("Error sending image content: %s", e)
    
    async def send_text_content(self, text: str, **kwargs) -> None:
        """Send text content using Gemini Live API."""
        if not self._active:
            return
        
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
    
    async def send_interrupt(self) -> None:
        """Send interruption signal to Gemini Live API.
        
        Gemini Live uses automatic VAD-based interruption. When new audio input
        is detected, it automatically interrupts the ongoing generation.
        We don't need to send explicit interrupt signals like Nova Sonic.
        """
        if not self._active:
            return
        
        try:
            # Gemini Live handles interruption automatically through VAD
            # When new audio input is sent via send_realtime_input, it automatically
            # interrupts any ongoing generation. No explicit interrupt signal needed.
            logger.debug("Interrupt requested - Gemini Live handles this automatically via VAD")
            
        except Exception as e:
            logger.error("Error in interrupt handling: %s", e)
    
    async def send_tool_result(self, tool_use_id: str, result: Dict[str, Any]) -> None:
        """Send tool result using Gemini Live API."""
        if not self._active:
            return
        
        try:
            # Create function response
            func_response = genai_types.FunctionResponse(
                id=tool_use_id,
                name=tool_use_id,  # Gemini uses name as identifier
                response=result
            )
            
            # Send tool response
            await self.live_session.send_tool_response(function_responses=[func_response])
        except Exception as e:
            logger.error("Error sending tool result: %s", e)
    
    async def send_tool_error(self, tool_use_id: str, error: str) -> None:
        """Send tool error using Gemini Live API."""
        error_result = {"error": error}
        await self.send_tool_result(tool_use_id, error_result)
    
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