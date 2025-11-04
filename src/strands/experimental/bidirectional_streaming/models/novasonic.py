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
from typing import AsyncIterable, Union

from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient, InvokeModelWithBidirectionalStreamOperationInput
from aws_sdk_bedrock_runtime.config import Config, HTTPAuthSchemeResolver, SigV4AuthScheme
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
    InvokeModelWithBidirectionalStreamOperationOutput,
)
from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver

from ....types.content import Messages
from ....types.tools import ToolResult, ToolSpec, ToolUse
from ..types.bidirectional_streaming import (
    AudioInputEvent,
    AudioOutputEvent,
    BidirectionalConnectionEndEvent,
    BidirectionalConnectionStartEvent,
    ImageInputEvent,
    InterruptionDetectedEvent,
    TextInputEvent,
    TextOutputEvent,
    UsageMetricsEvent,
)
from .bidirectional_model import BidirectionalModel

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


class NovaSonicModel(BidirectionalModel):
    """Nova Sonic implementation for bidirectional streaming.

    Combines model configuration and connection state in a single class.
    Manages Nova Sonic's complex event sequencing, audio format conversion, and
    tool execution patterns while providing the standard BidirectionalModel interface.
    """

    def __init__(
        self,
        model_id: str = "amazon.nova-sonic-v1:0",
        region: str = "us-east-1",
        **kwargs
    ) -> None:
        """Initialize Nova Sonic bidirectional model.

        Args:
            model_id: Nova Sonic model identifier.
            region: AWS region.
            **kwargs: Reserved for future parameters.
        """
        # Model configuration
        self.model_id = model_id
        self.region = region
        self.client = None

        # Connection state (initialized in connect())
        self.stream = None
        self.session_id = None
        self._active = False

        # Nova Sonic requires unique content names
        self.audio_content_name = None

        # Audio connection state
        self.audio_connection_active = False
        self.last_audio_time = None
        self.silence_threshold = SILENCE_THRESHOLD
        self.silence_task = None

        # Background task and event queue
        self._response_task = None
        self._event_queue = None

        logger.debug("Nova Sonic bidirectional model initialized: %s", model_id)

    async def connect(
        self,
        system_prompt: str | None = None,
        tools: list[ToolSpec] | None = None,
        messages: Messages | None = None,
        **kwargs,
    ) -> None:
        """Establish bidirectional connection to Nova Sonic.

        Args:
            system_prompt: System instructions for the model.
            tools: List of tools available to the model.
            messages: Conversation history to initialize with.
            **kwargs: Additional configuration options.
        """
        if self._active:
            raise RuntimeError("Connection already active. Close the existing connection before creating a new one.")
        
        logger.debug("Nova connection create - starting")

        try:
            # Initialize client if needed
            if not self.client:
                await self._initialize_client()

            # Initialize connection state
            self.session_id = str(uuid.uuid4())
            self._active = True
            self.audio_content_name = str(uuid.uuid4())
            self._event_queue = asyncio.Queue()

            # Start Nova Sonic bidirectional stream
            self.stream = await self.client.invoke_model_with_bidirectional_stream(
                InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
            )

            # Validate stream
            if not self.stream:
                logger.error("Stream is None")
                raise ValueError("Stream cannot be None")

            logger.debug("Nova Sonic connection initialized with session: %s", self.session_id)

            # Send initialization events
            system_prompt = system_prompt or "You are a helpful assistant. Keep responses brief."
            init_events = self._build_initialization_events(system_prompt, tools or [], messages)

            logger.debug("Nova Sonic initialization - sending %d events", len(init_events))
            await self._send_initialization_events(init_events)

            # Start background response processor
            self._response_task = asyncio.create_task(self._process_responses())

            logger.info("Nova Sonic connection established successfully")

        except Exception as e:
            self._active = False
            logger.error("Nova connection create error: %s", str(e))
            raise

    def _build_initialization_events(
        self, system_prompt: str, tools: list[ToolSpec], messages: Messages | None
    ) -> list[str]:
        """Build the sequence of initialization events."""
        events = [self._get_connection_start_event(), self._get_prompt_start_event(tools)]

        events.extend(self._get_system_prompt_events(system_prompt))

        # Message history would be processed here if needed in the future
        # Currently not implemented as it's not used in the existing test cases

        return events

    async def _send_initialization_events(self, events: list[str]) -> None:
        """Send initialization events with required delays."""
        for _i, event in enumerate(events):
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
                    logger.warning("Nova Sonic response error: %s", e)
                    await asyncio.sleep(0.1)
                    continue

        except Exception as e:
            logger.error("Nova Sonic fatal error: %s", e)
        finally:
            logger.debug("Nova Sonic response processor stopped")

    async def _handle_response_data(self, response_data: str) -> None:
        """Handle decoded response data from Nova Sonic."""
        try:
            json_data = json.loads(response_data)

            if "event" in json_data:
                nova_event = json_data["event"]
                self._log_event_type(nova_event)

                if not hasattr(self, "_event_queue"):
                    self._event_queue = asyncio.Queue()

                await self._event_queue.put(nova_event)
        except json.JSONDecodeError as e:
            logger.warning("Nova Sonic JSON decode error: %s", e)

    def _log_event_type(self, nova_event: dict[str, any]) -> None:
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

    async def receive(self) -> AsyncIterable[dict[str, any]]:
        """Receive Nova Sonic events and convert to provider-agnostic format."""
        if not self.stream:
            logger.error("Stream is None")
            return

        logger.debug("Nova events - starting event stream")

        # Emit connection start event to Strands event system
        connection_start: BidirectionalConnectionStartEvent = {
            "connectionId": self.session_id,
            "metadata": {"provider": "nova_sonic", "model_id": self.model_id},
        }
        yield {"BidirectionalConnectionStart": connection_start}

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
                "connectionId": self.session_id,
                "reason": "connection_complete",
                "metadata": {"provider": "nova_sonic"},
            }
            yield {"BidirectionalConnectionEnd": connection_end}

    async def send(self, content: Union[TextInputEvent, ImageInputEvent, AudioInputEvent, ToolResult]) -> None:
        """Unified send method for all content types. Sends the given content to Nova Sonic.

        Dispatches to appropriate internal handler based on content type.

        Args:
            content: Typed event (TextInputEvent, ImageInputEvent, AudioInputEvent, or ToolResult).
        """
        if not self._active:
            return

        try:
            if isinstance(content, dict):
                # Dispatch based on content structure
                if "text" in content and "role" in content:
                    # TextInputEvent
                    await self._send_text_content(content["text"])
                elif "audioData" in content:
                    # AudioInputEvent
                    await self._send_audio_content(content)
                elif "imageData" in content or "image_url" in content:
                    # ImageInputEvent - not supported by Nova Sonic
                    logger.warning("Image input not supported by Nova Sonic")
                elif "toolUseId" in content and "status" in content:
                    # ToolResult
                    await self._send_tool_result(content)
                else:
                    logger.warning(f"Unknown content type with keys: {content.keys()}")
        except Exception as e:
            logger.error(f"Error sending content: {e}")
            raise  # Propagate exception for debugging in experimental code

    async def _start_audio_connection(self) -> None:
        """Internal: Start audio input connection (call once before sending audio chunks)."""
        if self.audio_connection_active:
            return

        logger.debug("Nova audio connection start")

        audio_content_start = json.dumps(
            {
                "event": {
                    "contentStart": {
                        "promptName": self.session_id,
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

    async def _send_audio_content(self, audio_input: AudioInputEvent) -> None:
        """Internal: Send audio using Nova Sonic protocol-specific format."""
        # Start audio connection if not already active
        if not self.audio_connection_active:
            await self._start_audio_connection()

        # Update last audio time and cancel any pending silence task
        self.last_audio_time = time.time()
        if self.silence_task and not self.silence_task.done():
            self.silence_task.cancel()

        # Convert audio to Nova Sonic base64 format
        nova_audio_data = base64.b64encode(audio_input["audioData"]).decode("utf-8")

        # Send audio input event
        audio_event = json.dumps(
            {
                "event": {
                    "audioInput": {
                        "promptName": self.session_id,
                        "contentName": self.audio_content_name,
                        "content": nova_audio_data,
                    }
                }
            }
        )

        await self._send_nova_event(audio_event)

        # Start silence detection task
        self.silence_task = asyncio.create_task(self._check_silence())

    async def _check_silence(self) -> None:
        """Internal: Check for silence and automatically end audio connection."""
        try:
            await asyncio.sleep(self.silence_threshold)
            if self.audio_connection_active and self.last_audio_time:
                elapsed = time.time() - self.last_audio_time
                if elapsed >= self.silence_threshold:
                    logger.debug("Nova silence detected: %.2f seconds", elapsed)
                    await self._end_audio_input()
        except asyncio.CancelledError:
            pass

    async def _end_audio_input(self) -> None:
        """Internal: End current audio input connection to trigger Nova Sonic processing."""
        if not self.audio_connection_active:
            return

        logger.debug("Nova audio connection end")

        audio_content_end = json.dumps(
            {"event": {"contentEnd": {"promptName": self.session_id, "contentName": self.audio_content_name}}}
        )

        await self._send_nova_event(audio_content_end)
        self.audio_connection_active = False

    async def _send_text_content(self, text: str) -> None:
        """Internal: Send text content using Nova Sonic format."""
        content_name = str(uuid.uuid4())
        events = [
            self._get_text_content_start_event(content_name),
            self._get_text_input_event(content_name, text),
            self._get_content_end_event(content_name),
        ]

        for event in events:
            await self._send_nova_event(event)

    async def _send_interrupt(self) -> None:
        """Internal: Send interruption signal to Nova Sonic."""
        # Nova Sonic handles interruption through special input events
        interrupt_event = json.dumps(
            {
                "event": {
                    "audioInput": {
                        "promptName": self.session_id,
                        "contentName": self.audio_content_name,
                        "stopReason": "INTERRUPTED",
                    }
                }
            }
        )
        await self._send_nova_event(interrupt_event)

    async def _send_tool_result(self, tool_result: ToolResult) -> None:
        """Internal: Send tool result using Nova Sonic toolResult format."""
        tool_use_id = tool_result.get("toolUseId")

        logger.debug("Nova tool result send: %s", tool_use_id)

        # Extract result content
        result_data = {}
        if "content" in tool_result:
            # Extract text from content blocks
            for block in tool_result["content"]:
                if "text" in block:
                    result_data = {"result": block["text"]}
                    break

        content_name = str(uuid.uuid4())
        events = [
            self._get_tool_content_start_event(content_name, tool_use_id),
            self._get_tool_result_event(content_name, result_data),
            self._get_content_end_event(content_name),
        ]

        for event in events:
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
                await self._end_audio_input()

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

    def _convert_nova_event(self, nova_event: dict[str, any]) -> dict[str, any] | None:
        """Convert Nova Sonic events to provider-agnostic format."""
        # Handle audio output
        if "audioOutput" in nova_event:
            audio_content = nova_event["audioOutput"]["content"]
            audio_bytes = base64.b64decode(audio_content)

            audio_output: AudioOutputEvent = {
                "audioData": audio_bytes,
                "format": "pcm",
                "sampleRate": 24000,
                "channels": 1,
                "encoding": "base64",
            }

            return {"audioOutput": audio_output}

        # Handle text output
        elif "textOutput" in nova_event:
            text_content = nova_event["textOutput"]["content"]
            # Use stored role from contentStart event, fallback to event role
            role = getattr(self, "_current_role", nova_event["textOutput"].get("role", "assistant"))

            # Check for Nova Sonic interruption pattern (matches working sample)
            if '{ "interrupted" : true }' in text_content:
                logger.debug("Nova interruption detected in text")
                interruption: InterruptionDetectedEvent = {"reason": "user_input"}
                return {"interruptionDetected": interruption}

            # Show transcription for user speech - ALWAYS show these regardless of DEBUG flag
            if role == "USER":
                print(f"User: {text_content}")
            elif role == "ASSISTANT":
                print(f"Assistant: {text_content}")

            text_output: TextOutputEvent = {"text": text_content, "role": role.lower()}

            return {"textOutput": text_output}

        # Handle tool use
        elif "toolUse" in nova_event:
            tool_use = nova_event["toolUse"]

            tool_use_event: ToolUse = {
                "toolUseId": tool_use["toolUseId"],
                "name": tool_use["toolName"],
                "input": json.loads(tool_use["content"]),
            }

            return {"toolUse": tool_use_event}

        # Handle interruption
        elif nova_event.get("stopReason") == "INTERRUPTED":
            logger.debug("Nova interruption stop reason")

            interruption: InterruptionDetectedEvent = {"reason": "user_input"}

            return {"interruptionDetected": interruption}

        # Handle usage events - convert to standardized format
        elif "usageEvent" in nova_event:
            usage_data = nova_event["usageEvent"]
            usage_metrics: UsageMetricsEvent = {
                "totalTokens": usage_data.get("totalTokens", 0),
                "inputTokens": usage_data.get("totalInputTokens", 0),
                "outputTokens": usage_data.get("totalOutputTokens", 0),
                "audioTokens": usage_data.get("details", {}).get("total", {}).get("output", {}).get("speechTokens", 0)
            }
            return {"usageMetrics": usage_metrics}

        # Handle content start events (track role)
        elif "contentStart" in nova_event:
            role = nova_event["contentStart"].get("role", "unknown")
            # Store role for subsequent text output events
            self._current_role = role
            return None

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
                    "promptName": self.session_id,
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

    def _build_tool_configuration(self, tools: list[ToolSpec]) -> list[dict]:
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

    def _get_system_prompt_events(self, system_prompt: str) -> list[str]:
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
                        "promptName": self.session_id,
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
                        "promptName": self.session_id,
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
            {"event": {"textInput": {"promptName": self.session_id, "contentName": content_name, "content": text}}}
        )

    def _get_tool_result_event(self, content_name: str, result: dict[str, any]) -> str:
        """Generate tool result event."""
        return json.dumps(
            {
                "event": {
                    "toolResult": {
                        "promptName": self.session_id,
                        "contentName": content_name,
                        "content": json.dumps(result),
                    }
                }
            }
        )

    def _get_content_end_event(self, content_name: str) -> str:
        """Generate content end event."""
        return json.dumps({"event": {"contentEnd": {"promptName": self.session_id, "contentName": content_name}}})

    def _get_prompt_end_event(self) -> str:
        """Generate prompt end event."""
        return json.dumps({"event": {"promptEnd": {"promptName": self.session_id}}})

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

            self.client = BedrockRuntimeClient(config=config)
            logger.debug("Nova Sonic client initialized")

        except ImportError as e:
            logger.error("Nova Sonic dependencies not available: %s", e)
            raise
        except Exception as e:
            logger.error("Error initializing Nova Sonic client: %s", e)
            raise
