"""Gemini Live API bidirectional model provider using official Google GenAI SDK.

Implements the BidiModel interface for Google's Gemini Live API using the
official Google GenAI SDK for simplified and robust WebSocket communication.

Key improvements over custom WebSocket implementation:

- Uses official google-genai SDK with native Live API support
- Simplified session management with client.aio.live.connect()
- Built-in tool integration and event handling
- Automatic WebSocket connection management and error handling
- Native support for audio/text streaming and interruption
"""

import base64
import logging
import uuid
from typing import Any, AsyncGenerator, cast

from google import genai
from google.genai import types as genai_types
from google.genai.types import LiveConnectConfigOrDict, LiveServerMessage

from ....types._events import ToolResultEvent, ToolUseStreamEvent
from ....types.content import Messages
from ....types.tools import ToolResult, ToolSpec, ToolUse
from .._async import stop_all
from ..types.events import (
    AudioChannel,
    AudioSampleRate,
    BidiAudioInputEvent,
    BidiAudioStreamEvent,
    BidiConnectionStartEvent,
    BidiImageInputEvent,
    BidiInputEvent,
    BidiInterruptionEvent,
    BidiOutputEvent,
    BidiTextInputEvent,
    BidiTranscriptStreamEvent,
    BidiUsageEvent,
    ModalityUsage,
)
from ..types.model import AudioConfig
from .model import BidiModel, BidiModelTimeoutError

logger = logging.getLogger(__name__)

# Audio format constants
GEMINI_INPUT_SAMPLE_RATE: AudioSampleRate = 16000
GEMINI_OUTPUT_SAMPLE_RATE: AudioSampleRate = 24000
GEMINI_CHANNELS: AudioChannel = 1


class BidiGeminiLiveModel(BidiModel):
    """Gemini Live API implementation using official Google GenAI SDK.

    Combines model configuration and connection state in a single class.
    Provides a clean interface to Gemini Live API using the official SDK,
    eliminating custom WebSocket handling and providing robust error handling.
    """

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash-native-audio-preview-09-2025",
        provider_config: dict[str, Any] | None = None,
        client_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Initialize Gemini Live API bidirectional model.

        Args:
            model_id: Model identifier (default: gemini-2.5-flash-native-audio-preview-09-2025)
            provider_config: Model behavior (audio, inference)
            client_config: Authentication (api_key, http_options)
            **kwargs: Reserved for future parameters.

        """
        # Store model ID
        self.model_id = model_id

        # Resolve client config with defaults
        self._client_config = self._resolve_client_config(client_config or {})

        # Resolve provider config with defaults
        self.config = self._resolve_provider_config(provider_config or {})

        # Store API key for later use
        self.api_key = self._client_config.get("api_key")

        # Create Gemini client
        self._client = genai.Client(**self._client_config)

        # Connection state (initialized in start())
        self._live_session: Any = None
        self._live_session_context_manager: Any = None
        self._live_session_handle: str | None = None
        self._connection_id: str | None = None

    def _resolve_client_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Resolve client config (sets default http_options if not provided)."""
        resolved = config.copy()

        # Set default http_options if not provided
        if "http_options" not in resolved:
            resolved["http_options"] = {"api_version": "v1alpha"}

        return resolved

    def _resolve_provider_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Merge user config with defaults (user takes precedence)."""
        default_audio: AudioConfig = {
            "input_rate": GEMINI_INPUT_SAMPLE_RATE,
            "output_rate": GEMINI_OUTPUT_SAMPLE_RATE,
            "channels": GEMINI_CHANNELS,
            "format": "pcm",
        }
        default_inference = {
            "response_modalities": ["AUDIO"],
            "outputAudioTranscription": {},
            "inputAudioTranscription": {},
        }

        resolved = {
            "audio": {
                **default_audio,
                **config.get("audio", {}),
            },
            "inference": {
                **default_inference,
                **config.get("inference", {}),
            },
        }
        return resolved

    async def start(
        self,
        system_prompt: str | None = None,
        tools: list[ToolSpec] | None = None,
        messages: Messages | None = None,
        **kwargs: Any,
    ) -> None:
        """Establish bidirectional connection with Gemini Live API.

        Args:
            system_prompt: System instructions for the model.
            tools: List of tools available to the model.
            messages: Conversation history to initialize with.
            **kwargs: Additional configuration options.
        """
        if self._connection_id:
            raise RuntimeError("model already started | call stop before starting again")

        self._connection_id = str(uuid.uuid4())

        # Build live config
        live_config = self._build_live_config(system_prompt, tools, **kwargs)

        # Create the context manager and session
        self._live_session_context_manager = self._client.aio.live.connect(
            model=self.model_id, config=cast(LiveConnectConfigOrDict, live_config)
        )
        self._live_session = await self._live_session_context_manager.__aenter__()

        # Gemini itself restores message history when resuming from session
        if messages and "live_session_handle" not in kwargs:
            await self._send_message_history(messages)

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
                await self._live_session.send_client_content(turns=content)

    async def receive(self) -> AsyncGenerator[BidiOutputEvent, None]:
        """Receive Gemini Live API events and convert to provider-agnostic format."""
        if not self._connection_id:
            raise RuntimeError("model not started | call start before receiving")

        yield BidiConnectionStartEvent(connection_id=self._connection_id, model=self.model_id)

        # Wrap in while loop to restart after turn_complete (SDK limitation workaround)
        while True:
            async for message in self._live_session.receive():
                for event in self._convert_gemini_live_event(message):
                    yield event

    def _convert_gemini_live_event(self, message: LiveServerMessage) -> list[BidiOutputEvent]:
        """Convert Gemini Live API events to provider-agnostic format.

        Handles different types of content:

        - inputTranscription: User's speech transcribed to text
        - outputTranscription: Model's audio transcribed to text
        - modelTurn text: Text response from the model
        - usageMetadata: Token usage information

        Returns:
            List of event dicts (empty list if no events to emit).

        Raises:
            BidiModelTimeoutError: If gemini responds with go away message.
        """
        if message.go_away:
            raise BidiModelTimeoutError(
                message.go_away.model_dump_json(), live_session_handle=self._live_session_handle
            )

        if message.session_resumption_update:
            resumption_update = message.session_resumption_update
            if resumption_update.resumable and resumption_update.new_handle:
                self._live_session_handle = resumption_update.new_handle
                logger.debug("session_handle=<%s> | updating gemini session handle", self._live_session_handle)
            return []

        # Handle interruption first (from server_content)
        if message.server_content and message.server_content.interrupted:
            return [BidiInterruptionEvent(reason="user_speech")]

        # Handle input transcription (user's speech) - emit as transcript event
        if message.server_content and message.server_content.input_transcription:
            input_transcript = message.server_content.input_transcription
            # Check if the transcription object has text content
            if hasattr(input_transcript, "text") and input_transcript.text:
                transcription_text = input_transcript.text
                logger.debug("text_length=<%d> | gemini input transcription detected", len(transcription_text))
                return [
                    BidiTranscriptStreamEvent(
                        delta={"text": transcription_text},
                        text=transcription_text,
                        role="user",
                        # TODO: https://github.com/googleapis/python-genai/issues/1504
                        is_final=bool(input_transcript.finished),
                        current_transcript=transcription_text,
                    )
                ]

        # Handle output transcription (model's audio) - emit as transcript event
        if message.server_content and message.server_content.output_transcription:
            output_transcript = message.server_content.output_transcription
            # Check if the transcription object has text content
            if hasattr(output_transcript, "text") and output_transcript.text:
                transcription_text = output_transcript.text
                logger.debug("text_length=<%d> | gemini output transcription detected", len(transcription_text))
                return [
                    BidiTranscriptStreamEvent(
                        delta={"text": transcription_text},
                        text=transcription_text,
                        role="assistant",
                        # TODO: https://github.com/googleapis/python-genai/issues/1504
                        is_final=bool(output_transcript.finished),
                        current_transcript=transcription_text,
                    )
                ]

        # Handle audio output using SDK's built-in data property
        # Check this BEFORE text to avoid triggering warning on mixed content
        if message.data:
            # Convert bytes to base64 string for JSON serializability
            audio_b64 = base64.b64encode(message.data).decode("utf-8")
            return [
                BidiAudioStreamEvent(
                    audio=audio_b64,
                    format="pcm",
                    sample_rate=cast(AudioSampleRate, self.config["audio"]["output_rate"]),
                    channels=cast(AudioChannel, self.config["audio"]["channels"]),
                )
            ]

        # Handle text output from model_turn (avoids warning by checking parts directly)
        if message.server_content and message.server_content.model_turn:
            model_turn = message.server_content.model_turn
            if model_turn.parts:
                # Concatenate all text parts (Gemini may send multiple parts)
                text_parts = []
                for part in model_turn.parts:
                    # Check if part has text attribute and it's not empty
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)

                if text_parts:
                    full_text = " ".join(text_parts)
                    return [
                        BidiTranscriptStreamEvent(
                            delta={"text": full_text},
                            text=full_text,
                            role="assistant",
                            is_final=True,
                            current_transcript=full_text,
                        )
                    ]

        # Handle tool calls - return list to support multiple tool calls
        if message.tool_call and message.tool_call.function_calls:
            tool_events: list[BidiOutputEvent] = []
            for func_call in message.tool_call.function_calls:
                tool_use_event: ToolUse = {
                    "toolUseId": cast(str, func_call.id),
                    "name": cast(str, func_call.name),
                    "input": func_call.args or {},
                }
                # Create ToolUseStreamEvent for consistency with standard agent
                tool_events.append(
                    ToolUseStreamEvent(delta={"toolUse": tool_use_event}, current_tool_use=dict(tool_use_event))
                )
            return tool_events

        # Handle usage metadata
        if hasattr(message, "usage_metadata") and message.usage_metadata:
            usage = message.usage_metadata

            # Build modality details from token details
            modality_details = []

            # Process prompt tokens details
            if usage.prompt_tokens_details:
                for detail in usage.prompt_tokens_details:
                    if detail.modality and detail.token_count:
                        modality_details.append(
                            {
                                "modality": str(detail.modality).lower(),
                                "input_tokens": detail.token_count,
                                "output_tokens": 0,
                            }
                        )

            # Process response tokens details
            if usage.response_tokens_details:
                for detail in usage.response_tokens_details:
                    if detail.modality and detail.token_count:
                        # Find or create modality entry
                        modality_str = str(detail.modality).lower()
                        existing = next((m for m in modality_details if m["modality"] == modality_str), None)
                        if existing:
                            existing["output_tokens"] = detail.token_count
                        else:
                            modality_details.append(
                                {"modality": modality_str, "input_tokens": 0, "output_tokens": detail.token_count}
                            )

            return [
                BidiUsageEvent(
                    input_tokens=usage.prompt_token_count or 0,
                    output_tokens=usage.response_token_count or 0,
                    total_tokens=usage.total_token_count or 0,
                    modality_details=cast(list[ModalityUsage], modality_details) if modality_details else None,
                    cache_read_input_tokens=usage.cached_content_token_count
                    if usage.cached_content_token_count
                    else None,
                )
            ]

        # Silently ignore setup_complete and generation_complete messages
        return []

    async def send(
        self,
        content: BidiInputEvent | ToolResultEvent,
    ) -> None:
        """Unified send method for all content types. Sends the given inputs to Google Live API.

        Dispatches to appropriate internal handler based on content type.

        Args:
            content: Typed event (BidiTextInputEvent, BidiAudioInputEvent, BidiImageInputEvent, or ToolResultEvent).

        Raises:
            ValueError: If content type not supported (e.g., image content).
        """
        if not self._connection_id:
            raise RuntimeError("model not started | call start before sending/receiving")

        if isinstance(content, BidiTextInputEvent):
            await self._send_text_content(content.text)
        elif isinstance(content, BidiAudioInputEvent):
            await self._send_audio_content(content)
        elif isinstance(content, BidiImageInputEvent):
            await self._send_image_content(content)
        elif isinstance(content, ToolResultEvent):
            tool_result = content.get("tool_result")
            if tool_result:
                await self._send_tool_result(tool_result)
        else:
            raise ValueError(f"content_type={type(content)} | content not supported")

    async def _send_audio_content(self, audio_input: BidiAudioInputEvent) -> None:
        """Internal: Send audio content using Gemini Live API.

        Gemini Live expects continuous audio streaming via send_realtime_input.
        This automatically triggers VAD and can interrupt ongoing responses.
        """
        # Decode base64 audio to bytes for SDK
        audio_bytes = base64.b64decode(audio_input.audio)

        # Create audio blob for the SDK
        mime_type = f"audio/pcm;rate={self.config['audio']['input_rate']}"
        audio_blob = genai_types.Blob(data=audio_bytes, mime_type=mime_type)

        # Send real-time audio input - this automatically handles VAD and interruption
        await self._live_session.send_realtime_input(audio=audio_blob)

    async def _send_image_content(self, image_input: BidiImageInputEvent) -> None:
        """Internal: Send image content using Gemini Live API.

        Sends image frames following the same pattern as the GitHub example.
        Images are sent as base64-encoded data with MIME type.
        """
        # Image is already base64 encoded in the event
        msg = {"mime_type": image_input.mime_type, "data": image_input.image}

        # Send using the same method as the GitHub example
        await self._live_session.send(input=msg)

    async def _send_text_content(self, text: str) -> None:
        """Internal: Send text content using Gemini Live API."""
        # Create content with text
        content = genai_types.Content(role="user", parts=[genai_types.Part(text=text)])

        # Send as client content
        await self._live_session.send_client_content(turns=content)

    async def _send_tool_result(self, tool_result: ToolResult) -> None:
        """Internal: Send tool result using Gemini Live API."""
        tool_use_id = tool_result.get("toolUseId")
        content = tool_result.get("content", [])

        # Validate all content types are supported
        for block in content:
            if "text" not in block and "json" not in block:
                # Unsupported content type - raise error
                raise ValueError(
                    f"tool_use_id=<{tool_use_id}>, content_types=<{list(block.keys())}> | "
                    f"Content type not supported by Gemini Live API"
                )

        # Optimize for single content item - unwrap the array
        if len(content) == 1:
            result_data = cast(dict[str, Any], content[0])
        else:
            # Multiple items - send as array
            result_data = {"result": content}

        # Create function response
        func_response = genai_types.FunctionResponse(
            id=tool_use_id,
            name=tool_use_id,  # Gemini uses name as identifier
            response=result_data,
        )

        # Send tool response
        await self._live_session.send_tool_response(function_responses=[func_response])

    async def stop(self) -> None:
        """Close Gemini Live API connection."""

        async def stop_session() -> None:
            if not self._live_session_context_manager:
                return

            await self._live_session_context_manager.__aexit__(None, None, None)

        async def stop_connection() -> None:
            self._connection_id = None

        await stop_all(stop_session, stop_connection)

    def _build_live_config(
        self, system_prompt: str | None = None, tools: list[ToolSpec] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Build LiveConnectConfig for the official SDK.

        Simply passes through all config parameters from provider_config, allowing users
        to configure any Gemini Live API parameter directly.
        """
        config_dict: dict[str, Any] = self.config["inference"].copy()

        config_dict["session_resumption"] = {"handle": kwargs.get("live_session_handle")}

        # Add system instruction if provided
        if system_prompt:
            config_dict["system_instruction"] = system_prompt

        # Add tools if provided
        if tools:
            config_dict["tools"] = self._format_tools_for_live_api(tools)

        if "voice" in self.config["audio"]:
            config_dict.setdefault("speech_config", {}).setdefault("voice_config", {}).setdefault(
                "prebuilt_voice_config", {}
            )["voice_name"] = self.config["audio"]["voice"]

        return config_dict

    def _format_tools_for_live_api(self, tool_specs: list[ToolSpec]) -> list[genai_types.Tool]:
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
