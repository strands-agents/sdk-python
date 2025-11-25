"""OpenAI Realtime API provider for Strands bidirectional streaming.

Provides real-time audio and text communication through OpenAI's Realtime API
with WebSocket connections, voice activity detection, and function calling.
"""

import json
import logging
import os
import uuid
from typing import Any, AsyncGenerator, cast, Literal

import websockets
from websockets import ClientConnection

from ....types._events import ToolResultEvent, ToolUseStreamEvent
from ....types.content import Messages
from ....types.tools import ToolResult, ToolSpec, ToolUse
from .._async import stop_all
from ..types.events import (
    BidiAudioInputEvent,
    BidiAudioStreamEvent,
    BidiConnectionStartEvent,
    BidiInputEvent,
    BidiInterruptionEvent,
    BidiOutputEvent,
    BidiResponseCompleteEvent,
    BidiResponseStartEvent,
    BidiTextInputEvent,
    BidiTranscriptStreamEvent,
    BidiUsageEvent,
    ModalityUsage,
    Role,
    SampleRate,
    StopReason,
)
from ..types.bidi_model import AudioConfig
from .bidi_model import BidiModel

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
            "transcription": {"model": "gpt-4o-transcribe"},
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500,
            },
        },
        "output": {"format": AUDIO_FORMAT, "voice": "alloy"},
    },
}


class BidiOpenAIRealtimeModel(BidiModel):
    """OpenAI Realtime API implementation for bidirectional streaming.

    Combines model configuration and connection state in a single class.
    Manages WebSocket connection to OpenAI's Realtime API with automatic VAD,
    function calling, and event conversion to Strands format.
    """

    _websocket: ClientConnection

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        session_config: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenAI Realtime bidirectional model.

        Args:
            model: OpenAI model identifier (default: gpt-realtime).
            api_key: OpenAI API key for authentication.
            organization: OpenAI organization ID for API requests.
            project: OpenAI project ID for API requests.
            session_config: Session configuration parameters (e.g., voice, turn_detection, modalities).
            config: Optional configuration dictionary with structure {"audio": AudioConfig, ...}.
                   If not provided or if "audio" key is missing, uses OpenAI Realtime API's default audio configuration.
            **kwargs: Reserved for future parameters.
        """
        # Model configuration
        self.model_id = model_id
        self.api_key = api_key
        self.organization = organization
        self.project = project
        self.session_config = session_config or {}

        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
                )

        # Connection state (initialized in start())
        self._connection_id: str | None = None

        self._function_call_buffer: dict[str, Any] = {}

        logger.debug("model=<%s> | openai realtime model initialized", model_id)

        # Extract audio config from config dict if provided
        user_audio_config = config.get("audio", {}) if config else {}

        # Extract voice from session_config if provided
        session_config_voice = "alloy"
        if self.session_config and "audio" in self.session_config:
            audio_settings = self.session_config["audio"]
            if isinstance(audio_settings, dict) and "output" in audio_settings:
                output_settings = audio_settings["output"]
                if isinstance(output_settings, dict):
                    session_config_voice = output_settings.get("voice", "alloy")

        # Define default audio configuration
        default_audio_config: AudioConfig = {
            "input_rate": cast(int, AUDIO_FORMAT["rate"]),
            "output_rate": cast(int, AUDIO_FORMAT["rate"]),
            "channels": 1,
            "format": "pcm",
            "voice": session_config_voice,
        }

        # Merge user config with defaults (user values take precedence)
        merged_audio_config = cast(AudioConfig, {**default_audio_config, **user_audio_config})

        # Store config with audio defaults always populated
        self.config: dict[str, Any] = {"audio": merged_audio_config}

        if user_audio_config:
            logger.debug("audio_config | merged user-provided config with defaults")
        else:
            logger.debug("audio_config | using default OpenAI Realtime audio configuration")

    async def start(
        self,
        system_prompt: str | None = None,
        tools: list[ToolSpec] | None = None,
        messages: Messages | None = None,
        **kwargs: Any,
    ) -> None:
        """Establish bidirectional connection to OpenAI Realtime API.

        Args:
            system_prompt: System instructions for the model.
            tools: List of tools available to the model.
            messages: Conversation history to initialize with.
            **kwargs: Additional configuration options.
        """
        if self._connection_id:
            raise RuntimeError("model already started | call stop before starting again")

        logger.info("openai realtime connection starting")

        # Initialize connection state
        self._connection_id = str(uuid.uuid4())

        self._function_call_buffer = {}

        # Establish WebSocket connection
        url = f"{OPENAI_REALTIME_URL}?model={self.model_id}"

        headers = [("Authorization", f"Bearer {self.api_key}")]
        if self.organization:
            headers.append(("OpenAI-Organization", self.organization))
        if self.project:
            headers.append(("OpenAI-Project", self.project))

        self._websocket = await websockets.connect(url, additional_headers=headers)
        logger.info("connection_id=<%s> | websocket connected successfully", self._connection_id)

        # Configure session
        session_config = self._build_session_config(system_prompt, tools)
        await self._send_event({"type": "session.update", "session": session_config})

        # Add conversation history if provided
        if messages:
            await self._add_conversation_history(messages)

    def _create_text_event(self, text: str, role: str, is_final: bool = True) -> BidiTranscriptStreamEvent:
        """Create standardized transcript event.

        Args:
            text: The transcript text
            role: The role (will be normalized to lowercase)
            is_final: Whether this is the final transcript
        """
        # Normalize role to lowercase and ensure it's either "user" or "assistant"
        normalized_role = role.lower() if isinstance(role, str) else "assistant"
        if normalized_role not in ["user", "assistant"]:
            normalized_role = "assistant"

        return BidiTranscriptStreamEvent(
            delta={"text": text},
            text=text,
            role=cast(Role, normalized_role),
            is_final=is_final,
            current_transcript=text if is_final else None,
        )

    def _create_voice_activity_event(self, activity_type: str) -> BidiInterruptionEvent | None:
        """Create standardized interruption event for voice activity."""
        # Only speech_started triggers interruption
        if activity_type == "speech_started":
            return BidiInterruptionEvent(reason="user_speech")
        # Other voice activity events are logged but don't create events
        return None

    def _build_session_config(self, system_prompt: str | None, tools: list[ToolSpec] | None) -> dict[str, Any]:
        """Build session configuration for OpenAI Realtime API."""
        config: dict[str, Any] = DEFAULT_SESSION_CONFIG.copy()

        if system_prompt:
            config["instructions"] = system_prompt

        if tools:
            config["tools"] = self._convert_tools_to_openai_format(tools)

        # Apply user-provided session configuration
        supported_params = {
            "type",
            "output_modalities",
            "instructions",
            "voice",
            "audio",
            "tools",
            "tool_choice",
            "input_audio_format",
            "output_audio_format",
            "input_audio_transcription",
            "turn_detection",
        }

        for key, value in self.session_config.items():
            if key in supported_params:
                config[key] = value
            else:
                logger.warning("parameter=<%s> | ignoring unsupported session parameter", key)

        # Override voice with config value if present (config takes precedence)
        if "voice" in self.config["audio"]:
            config.setdefault("audio", {}).setdefault("output", {})["voice"] = self.config["audio"]["voice"]

        return config

    def _convert_tools_to_openai_format(self, tools: list[ToolSpec]) -> list[dict]:
        """Convert Strands tool specifications to OpenAI Realtime API format."""
        openai_tools = []

        for tool in tools:
            input_schema = tool["inputSchema"]
            if "json" in input_schema:
                schema = (
                    json.loads(input_schema["json"]) if isinstance(input_schema["json"], str) else input_schema["json"]
                )
            else:
                schema = input_schema

            # OpenAI Realtime API expects flat structure, not nested under "function"
            openai_tool = {
                "type": "function",
                "name": tool["name"],
                "description": tool["description"],
                "parameters": schema,
            }
            openai_tools.append(openai_tool)

        return openai_tools

    async def _add_conversation_history(self, messages: Messages) -> None:
        """Add conversation history to the session."""
        for message in messages:
            conversation_item: dict[Any, Any] = {
                "type": "conversation.item.create",
                "item": {"type": "message", "role": message["role"], "content": []},
            }

            content = message.get("content", "")
            if isinstance(content, str):
                conversation_item["item"]["content"].append({"type": "input_text", "text": content})
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        conversation_item["item"]["content"].append(
                            {"type": "input_text", "text": item.get("text", "")}
                        )

            await self._send_event(conversation_item)

    async def receive(self) -> AsyncGenerator[BidiOutputEvent, None]:
        """Receive OpenAI events and convert to Strands TypedEvent format."""
        if not self._connection_id:
            raise RuntimeError("model not started | call start before receiving")

        yield BidiConnectionStartEvent(connection_id=self._connection_id, model=self.model_id)

        async for message in self._websocket:
            openai_event = json.loads(message)

            for event in self._convert_openai_event(openai_event) or []:
                yield event

    def _convert_openai_event(self, openai_event: dict[str, Any]) -> list[BidiOutputEvent] | None:
        """Convert OpenAI events to Strands TypedEvent format."""
        event_type = openai_event.get("type")

        # Turn start - response begins
        if event_type == "response.created":
            response = openai_event.get("response", {})
            response_id = response.get("id", str(uuid.uuid4()))
            return [BidiResponseStartEvent(response_id=response_id)]

        # Audio output
        elif event_type == "response.output_audio.delta":
            # Audio is already base64 string from OpenAI
            # Channels from config is guaranteed to be 1 or 2
            channels = cast(Literal[1, 2], self.config["audio"]["channels"])
            return [
                BidiAudioStreamEvent(
                    audio=openai_event["delta"],
                    format="pcm",
                    sample_rate=cast(SampleRate, AUDIO_FORMAT["rate"]),
                    channels=1,
                )
            ]

        # Assistant text output events - combine multiple similar events
        elif event_type in ["response.output_text.delta", "response.output_audio_transcript.delta"]:
            role = openai_event.get("role", "assistant")
            return [
                self._create_text_event(
                    openai_event["delta"], role.lower() if isinstance(role, str) else "assistant", is_final=False
                )
            ]

        elif event_type in ["response.output_audio_transcript.done"]:
            role = openai_event.get("role", "assistant").lower()
            return [self._create_text_event(openai_event["transcript"], role)]

        elif event_type in ["response.output_text.done"]:
            role = openai_event.get("role", "assistant").lower()
            return [self._create_text_event(openai_event["text"], role)]

        # User transcription events - combine multiple similar events
        elif event_type in [
            "conversation.item.input_audio_transcription.delta",
            "conversation.item.input_audio_transcription.completed",
        ]:
            text_key = "delta" if "delta" in event_type else "transcript"
            text = openai_event.get(text_key, "")
            role = openai_event.get("role", "user")
            is_final = "completed" in event_type
            return (
                [self._create_text_event(text, role.lower() if isinstance(role, str) else "user", is_final=is_final)]
                if text.strip()
                else None
            )

        elif event_type == "conversation.item.input_audio_transcription.segment":
            segment_data = openai_event.get("segment", {})
            text = segment_data.get("text", "")
            role = segment_data.get("role", "user")
            return (
                [self._create_text_event(text, role.lower() if isinstance(role, str) else "user")]
                if text.strip()
                else None
            )

        elif event_type == "conversation.item.input_audio_transcription.failed":
            error_info = openai_event.get("error", {})
            logger.warning("error=<%s> | openai transcription failed", error_info.get("message", "unknown error"))
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
                    return [ToolUseStreamEvent(delta={"toolUse": tool_use}, current_tool_use=dict(tool_use))]
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning("call_id=<%s>, error=<%s> | error parsing function arguments", call_id, e)
                    del self._function_call_buffer[call_id]
            return None

        # Voice activity detection - speech_started triggers interruption
        elif event_type == "input_audio_buffer.speech_started":
            # This is the primary interruption signal - handle it first
            return [BidiInterruptionEvent(reason="user_speech")]

        # Response cancelled - handle interruption
        elif event_type == "response.cancelled":
            response = openai_event.get("response", {})
            response_id = response.get("id", "unknown")
            logger.debug("response_id=<%s> | openai response cancelled", response_id)
            return [BidiResponseCompleteEvent(response_id=response_id, stop_reason="interrupted")]

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
                "incomplete": "interrupted",
            }

            # Build list of events to return
            events: list[Any] = []

            # Always add response complete event
            events.append(
                BidiResponseCompleteEvent(
                    response_id=response_id,
                    stop_reason=cast(StopReason, stop_reason_map.get(status, "complete")),
                ),
            )

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
                    modality_details.append(
                        {"modality": "text", "input_tokens": text_input, "output_tokens": text_output}
                    )

                # Audio modality
                audio_input = input_details.get("audio_tokens", 0)
                audio_output = output_details.get("audio_tokens", 0)
                if audio_input > 0 or audio_output > 0:
                    modality_details.append(
                        {"modality": "audio", "input_tokens": audio_input, "output_tokens": audio_output}
                    )

                # Image modality
                image_input = input_details.get("image_tokens", 0)
                if image_input > 0:
                    modality_details.append({"modality": "image", "input_tokens": image_input, "output_tokens": 0})

                # Cached tokens
                cached_tokens = input_details.get("cached_tokens", 0)

                # Add usage event
                events.append(
                    BidiUsageEvent(
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        total_tokens=usage.get("total_tokens", 0),
                        modality_details=cast(list[ModalityUsage], modality_details) if modality_details else None,
                        cache_read_input_tokens=cached_tokens if cached_tokens > 0 else None,
                    )
                )

            # Return list of events
            return events

        # Lifecycle events (log only) - combine multiple similar events
        elif event_type in ["conversation.item.retrieve", "conversation.item.added"]:
            item = openai_event.get("item", {})
            action = "retrieved" if "retrieve" in event_type else "added"
            logger.debug("action=<%s>, item_id=<%s> | openai conversation item event", action, item.get("id"))
            return None

        elif event_type == "conversation.item.done":
            logger.debug("item_id=<%s> | openai conversation item done", openai_event.get("item", {}).get("id"))
            return None

        # Response output events - combine similar events
        elif event_type in [
            "response.output_item.added",
            "response.output_item.done",
            "response.content_part.added",
            "response.content_part.done",
        ]:
            item_data = openai_event.get("item") or openai_event.get("part")
            logger.debug(
                "event_type=<%s>, item_id=<%s> | openai output event",
                event_type,
                item_data.get("id") if item_data else "unknown",
            )

            # Track function call names from response.output_item.added
            if event_type == "response.output_item.added":
                item = openai_event.get("item", {})
                if item.get("type") == "function_call":
                    call_id = item.get("call_id")
                    function_name = item.get("name")
                    if call_id and function_name:
                        if call_id not in self._function_call_buffer:
                            self._function_call_buffer[call_id] = {
                                "call_id": call_id,
                                "name": function_name,
                                "arguments": "",
                            }
                        else:
                            self._function_call_buffer[call_id]["name"] = function_name
            return None

        # Session/buffer events - combine simple log-only events
        elif event_type in [
            "input_audio_buffer.committed",
            "input_audio_buffer.cleared",
            "session.created",
            "session.updated",
        ]:
            logger.debug("event_type=<%s> | openai event received", event_type)
            return None

        elif event_type == "error":
            error_data = openai_event.get("error", {})
            error_code = error_data.get("code", "")

            # Suppress expected errors that don't affect session state
            if error_code == "response_cancel_not_active":
                # This happens when trying to cancel a response that's not active
                # It's safe to ignore as the session remains functional
                logger.debug("openai response cancel attempted when no response active")
                return None

            # Log other errors
            logger.error("error=<%s> | openai realtime error", error_data)
            return None

        else:
            logger.debug("event_type=<%s> | unhandled openai event type", event_type)
            return None

    async def send(
        self,
        content: BidiInputEvent | ToolResultEvent,
    ) -> None:
        """Unified send method for all content types. Sends the given content to OpenAI.

        Dispatches to appropriate internal handler based on content type.

        Args:
            content: Typed event (BidiTextInputEvent, BidiAudioInputEvent, BidiImageInputEvent, or ToolResultEvent).

        Raises:
            ValueError: If content type not supported (e.g., image content).
        """
        if not self._connection_id:
            raise RuntimeError("model not started | call start before sending")

        # Note: TypedEvent inherits from dict, so isinstance checks for TypedEvent must come first
        if isinstance(content, BidiTextInputEvent):
            await self._send_text_content(content.text)
        elif isinstance(content, BidiAudioInputEvent):
            await self._send_audio_content(content)
        elif isinstance(content, ToolResultEvent):
            tool_result = content.get("tool_result")
            if tool_result:
                await self._send_tool_result(tool_result)
        else:
            raise ValueError(f"content_type={type(content)} | content not supported")

    async def _send_audio_content(self, audio_input: BidiAudioInputEvent) -> None:
        """Internal: Send audio content to OpenAI for processing."""
        # Audio is already base64 encoded in the event
        await self._send_event({"type": "input_audio_buffer.append", "audio": audio_input.audio})

    async def _send_text_content(self, text: str) -> None:
        """Internal: Send text content to OpenAI for processing."""
        item_data = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}
        await self._send_event({"type": "conversation.item.create", "item": item_data})
        await self._send_event({"type": "response.create"})

    async def _send_interrupt(self) -> None:
        """Internal: Send interruption signal to OpenAI."""
        await self._send_event({"type": "response.cancel"})

    async def _send_tool_result(self, tool_result: ToolResult) -> None:
        """Internal: Send tool result back to OpenAI."""
        tool_use_id = tool_result.get("toolUseId")

        logger.debug("tool_use_id=<%s> | sending openai tool result", tool_use_id)

        # TODO: We need to extract all content and content types
        result_data: dict[Any, Any] | str = {}
        if "content" in tool_result:
            # Extract text from content blocks
            for block in tool_result["content"]:
                if "text" in block:
                    result_data = block["text"]
                    break

        result_text = json.dumps(result_data) if not isinstance(result_data, str) else result_data

        item_data = {"type": "function_call_output", "call_id": tool_use_id, "output": result_text}
        await self._send_event({"type": "conversation.item.create", "item": item_data})
        await self._send_event({"type": "response.create"})

    async def stop(self) -> None:
        """Close session and cleanup resources."""
        logger.debug("openai realtime connection cleanup starting")

        async def stop_websocket() -> None:
            if not hasattr(self, "_websocket"):
                return

            await self._websocket.close()

        async def stop_connection() -> None:
            self._connection_id = None

        await stop_all(stop_websocket, stop_connection)

        logger.debug("openai realtime connection closed")

    async def _send_event(self, event: dict[str, Any]) -> None:
        """Send event to OpenAI via WebSocket."""
        message = json.dumps(event)
        await self._websocket.send(message)
        logger.debug("event_type=<%s> | openai event sent", event.get("type"))
